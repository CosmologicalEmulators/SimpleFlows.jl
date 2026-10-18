using Reactant, Enzyme, Optimisers
import ForwardDiff

# Unconditional tests of train_flow!(...; backend=:reactant): Lux TrainState + AutoEnzyme,
# device-resident data, in-compiled batch gathering with padded exact partial batches.

const RExt = Base.get_extension(SimpleFlows, :ReactantExt)

function _rt_flow(arch, T; dims=3, seed=1, hidden=[8], n_transforms=2, identity_init=false)
    kw = (; architecture=arch, n_transforms, dist_dims=dims, hidden_layer_sizes=hidden,
          activation=tanh, identity_init, rng=Random.MersenneTwister(seed))
    return arch == :NSF ? FlowDistribution(T; identity_init=false, kw..., K=4, tail_bound=3.0) : FlowDistribution(T; identity_init=false, kw...)
end

_rt_data(T, n; dims=3, seed=2) =
    T.([1.0 0.3 0.0; -0.2 0.8 0.1; 0.0 0.2 1.5][1:dims, 1:dims] * randn(Random.MersenneTwister(seed), dims, n) .+ [1.0, -1.0, 0.5][1:dims])

# Independent flatten / rebuild (ForwardDiff needs a rebuild that does not cast back to the stored eltype).
_rt_vec(x::AbstractArray) = vec(Array(x))
_rt_vec(x::NamedTuple) = reduce(vcat, map(_rt_vec, values(x)))
function _rt_rebuild(v, template::AbstractArray, offset)
    n = length(template)
    return reshape(v[offset+1:offset+n], size(template)), offset + n
end
function _rt_rebuild(v, template::NamedTuple, offset)
    vals = Any[]
    for x in values(template)
        val, offset = _rt_rebuild(v, x, offset)
        push!(vals, val)
    end
    return NamedTuple{keys(template)}(Tuple(vals)), offset
end
_rt_is_cpu(x::AbstractArray) = x isa Array
_rt_is_cpu(x::NamedTuple) = all(_rt_is_cpu, values(x))
_rt_host(x) = Lux.cpu_device()(x)

@testset "Reactant training: weight gradients vs Zygote and ForwardDiff" begin
    Reactant.set_default_backend("cpu")
    for T in (Float32, Float64), arch in (:RealNVP, :NSF, :MAF)
        @testset "$arch $T" begin
            flow = _rt_flow(arch, T)
            data = _rt_data(T, 20)
            nrm = MinMaxNormalizer(data)
            xn = SimpleFlows.normalize(nrm, data)
            e = SimpleFlows._make_reactant_engine(flow, xn, nothing, nrm; lr=nothing, batch_size=5, opt=nothing)
            rtol, atol = T == Float64 ? (1e-8, 1e-10) : (2e-3, 2e-5)
            function check(idx, ps_host)
                data_b = RExt._padded_batch(e, e.xtrain, idx)
                grads, loss, _, ts = Lux.Training.compute_gradients(AutoEnzyme(), e.obj, data_b, e.ts)
                e.ts = ts
                xb = xn[:, idx]
                lossf(ps) = -mean(SimpleFlows.log_prob(flow.model, ps, flow.st, xb))
                gz = Zygote.gradient(lossf, ps_host)[1]
                @test Float64(loss) ≈ Float64(lossf(ps_host)) rtol=rtol
                @test _rt_vec(_rt_host(grads)) ≈ _rt_vec(gz) rtol=rtol atol=atol
                v = _rt_vec(ps_host)
                gfd = ForwardDiff.gradient(v) do w
                    psd, _ = _rt_rebuild(w, ps_host, 0)
                    -mean(SimpleFlows.log_prob(flow.model, psd, flow.st, xn[:, idx] .+ zero(first(w))))
                end
                @test _rt_vec(_rt_host(grads)) ≈ gfd rtol=(T == Float64 ? 1e-7 : 5e-3) atol=(T == Float64 ? 1e-9 : 1e-4)
                return grads
            end
            ps0 = deepcopy(_rt_host(e.ts.parameters))
            check([3, 1, 4, 7, 9], ps0)                 # full batch
            f1 = e.ts.cache.extras.compiled_gradient_function
            check([11, 2, 19], ps0)                     # partial (padded) batch, changed indices
            @test e.ts.cache.extras.compiled_gradient_function === f1   # same compiled function
            # changed weights through the same compile: take real updates first
            SimpleFlows._engine_step!(e, [5, 6, 7, 8, 9], 1, 1)
            SimpleFlows._engine_step!(e, [1, 2], 1, 2)
            ps1 = SimpleFlows._engine_snapshot(e)
            @test _rt_vec(ps1) != _rt_vec(ps0)
            check([10, 12, 13, 14, 15], ps1)
            @test e.ts.cache.extras.compiled_gradient_function === f1
        end
    end
end

@testset "Reactant training: matched updates and full train_flow! parity" begin
    Reactant.set_default_backend("cpu")
    combos = [(:RealNVP, Float64, :whiten), (:NSF, Float64, :minmax), (:MAF, Float64, :whiten),
              (:RealNVP, Float32, :minmax), (:NSF, Float32, :whiten), (:MAF, Float32, :minmax)]
    for (arch, T, nrm) in combos
        @testset "$arch $T $nrm" begin
            data = _rt_data(T, 23); val = _rt_data(T, 11; seed=4)
            fz = _rt_flow(arch, T; identity_init=(T == Float64)); fr = deepcopy(fz)   # Float64 half: default identity init
            hz, hr = Any[], Any[]
            kw = (; n_epochs=3, batch_size=7, verbose=false, normalization=nrm, validation_data=val,
                  eval_every=1, lr=1e-2)
            train_flow!(fz, data; kw..., history=hz, rng=Random.MersenneTwister(5))
            ret = train_flow!(fr, data; kw..., history=hr, rng=Random.MersenneTwister(5), backend=:reactant)
            @test ret === fr
            @test typeof(fr.normalizer) == typeof(fz.normalizer)
            @test _rt_is_cpu(fr.ps)              # CPU weights returned
            rtol = T == Float64 ? 1e-7 : 5e-3
            @test [h.epoch for h in hr] == [h.epoch for h in hz]
            @test [h.best_epoch for h in hr] == [h.best_epoch for h in hz]
            @test [h.train_nll for h in hr] ≈ [h.train_nll for h in hz] rtol=rtol
            @test [h.validation_nll for h in hr] ≈ [h.validation_nll for h in hz] rtol=rtol
            @test _rt_vec(fr.ps) ≈ _rt_vec(fz.ps) rtol=(T == Float64 ? 1e-6 : 2e-2) atol=(T == Float64 ? 1e-8 : 1e-3)
            # history is the physical NLL of the returned weights when the best epoch is the last
            if hr[end].best_epoch == 3
                @test hr[end].validation_nll ≈ -mean(Float64.(logpdf(fr, val))) rtol=(T == Float64 ? 1e-9 : 1e-4)
            end
            # inference artifact round trip without Reactant state
            dir = mktempdir()
            fr.parameter_names = ["a", "b", "c"]
            save_trained_flow(dir, fr)
            f2 = load_trained_flow(dir)
            @test f2.parameter_names == ["a", "b", "c"]
            @test typeof(f2.normalizer) == typeof(fr.normalizer)
            @test logpdf(f2, val) == logpdf(fr, val)
        end
    end

    @testset "reuse of fitted normalizer, validation best restore" begin
        T = Float64
        data = _rt_data(T, 20); val = _rt_data(T, 9; seed=6)
        f = _rt_flow(:RealNVP, T)
        train_flow!(f, data; n_epochs=0, verbose=false)                      # CPU fit
        n1 = f.normalizer
        ps0 = _rt_vec(f.ps)
        hist = Any[]
        # gradient ascent: the initial weights must be restored as the best checkpoint
        train_flow!(f, data; n_epochs=2, batch_size=8, verbose=false, opt=Optimisers.Descent(-0.01),
                    validation_data=val, history=hist, backend=:reactant, rng=Random.MersenneTwister(1))
        @test f.normalizer === n1
        @test all(h -> h.best_epoch == 0, hist)
        @test _rt_vec(f.ps) == ps0
    end
end

@testset "Reactant training: invalid/non-finite handling and options" begin
    Reactant.set_default_backend("cpu")
    T = Float32
    data = _rt_data(T, 8)
    @test_throws ArgumentError train_flow!(_rt_flow(:RealNVP, T), data; verbose=false, backend=:tpu)
    @test_throws ArgumentError train_flow!(_rt_flow(:RealNVP, T), data[1:2, :]; verbose=false, backend=:reactant)

    flow = _rt_flow(:RealNVP, T); train_flow!(flow, data; n_epochs=0, verbose=false)
    ps0 = _rt_vec(flow.ps); nrm0 = flow.normalizer
    # an update that makes the parameters non-finite is caught even when it is the very last one
    @test_throws ErrorException train_flow!(flow, data; n_epochs=1, batch_size=8, verbose=false,
                                            opt=Optimisers.Descent(Inf32), backend=:reactant)
    @test _rt_vec(flow.ps) == ps0 && flow.normalizer === nrm0
    # non-finite starting weights: the gradient/loss flag stops before any update
    bad = Lux.fmap(x -> fill!(copy(x), NaN32), flow.ps)
    flow.ps = bad
    @test_throws ErrorException train_flow!(flow, data; n_epochs=2, verbose=false, backend=:reactant)
    @test all(isnan, _rt_vec(flow.ps))
end

@testset "Reactant training: 8D compile smoke" begin
    Reactant.set_default_backend("cpu")
    dims = 8
    # correlated 8-D data with a wide spread of scales (condition number ~ 1e5)
    rng = Random.MersenneTwister(12)
    M = randn(rng, dims, dims) .* (10.0 .^ range(-1.5, 1.0, length=dims))'
    for arch in (:RealNVP, :NSF, :MAF)
        @testset "$arch" begin
            T = Float32
            data = T.(M * randn(rng, dims, 96) .+ 3)
            val = T.(M * randn(rng, dims, 32) .+ 3)
            fz = _rt_flow(arch, T; dims, hidden=[16, 16], n_transforms=3, identity_init=true); fr = deepcopy(fz)
            hz, hr = Any[], Any[]
            kw = (; n_epochs=2, batch_size=32, verbose=false, normalization=:whiten, validation_data=val)
            train_flow!(fz, data; kw..., history=hz, rng=Random.MersenneTwister(3))
            train_flow!(fr, data; kw..., history=hr, rng=Random.MersenneTwister(3), backend=:reactant)
            @test all(h -> isfinite(h.train_nll) && isfinite(h.validation_nll), hr)
            @test hr[1].train_nll ≈ hz[1].train_nll rtol=1e-3          # epoch 0, same weights
            @test hr[1].validation_nll ≈ hz[1].validation_nll rtol=1e-3
            @test all(isfinite, logpdf(fr, val))
        end
    end
end
