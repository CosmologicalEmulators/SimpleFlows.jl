using SimpleFlows
using Test
using Random
using Optimisers, Lux, Statistics

@testset "Custom Optimizer & Training" begin
    rng = Random.default_rng()
    Random.seed!(rng, 10)
    
    dims = 2
    N = 100
    data = randn(Float32, dims, N)
    
    flow = FlowDistribution(Float32; identity_init=false, architecture=:RealNVP, n_transforms=2, dist_dims=dims,
                            hidden_layer_sizes=[16, 16], rng)
                            
    # Initial NLL
    lp_init = -mean(logpdf(flow, data))
    
    # Custom optimizer
    custom_opt = Optimisers.Adam(1f-4)
    train_flow!(flow, data; n_epochs=5, batch_size=50, opt=custom_opt, verbose=false)
    
    # Check that training occurred without error and actually evaluated
    lp_trained = -mean(logpdf(flow, data))
    @test isfinite(lp_trained)
    @test !isnothing(flow.normalizer)
end

@testset "Automatic Type Conversion" begin
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    
    dims = 2
    N = 100
    # User passes Float64 data accidentally
    data_f64 = randn(Float64, dims, N)
    
    # Model is strictly Float32
    flow = FlowDistribution(Float32; identity_init=false, architecture=:RealNVP, n_transforms=2, dist_dims=dims,
                            hidden_layer_sizes=[16, 16], rng)
                            
    # Should not throw MethodError
    train_flow!(flow, data_f64; n_epochs=2, batch_size=50, verbose=false)
    
    # Normalizer should now be correctly typed as MinMaxNormalizer{Float32}
    @test flow.normalizer isa MinMaxNormalizer{Float32}
    
    # logpdf on Float64 data should evaluate correctly without errors.
    # The output type may promote to Float64 depending on input type.
    lp = logpdf(flow, data_f64)
    @test all(isfinite, lp)
end

@testset "Reparameterization Trick Zygote Gradients" begin
    rng = Random.default_rng()
    Random.seed!(rng, 100)
    
    dims = 2
    
    # We test that we can differentiate a loss function evaluated on the output of draw_samples.
    # This proves that our flows support the reparameterization trick natively.
    for arch in [:RealNVP, :NSF, :MAF]
        flow = FlowDistribution(Float32; identity_init=false, architecture=arch, n_transforms=1, dist_dims=dims,
                                hidden_layer_sizes=[16], K=4, tail_bound=2.0, rng)
        
        # We need a stable test_rng since draw_samples is stochastic.
        # Zygote can sometimes struggle with capturing global RNGs during AD, so we pass one explicitly
        # and ignore the derivative with respect to it (which is 0).
        test_rng = Random.default_rng()
        Random.seed!(test_rng, 42)
        
        function loss_fn(ps)
            x_samples = SimpleFlows.draw_samples(test_rng, Float32, flow.model, ps, flow.st, 10)
            return sum(x_samples.^2)
        end
        
        loss, (dps,) = Zygote.withgradient(loss_fn, flow.ps)
        
        @test isfinite(loss)
        @test !isnothing(dps)
    end
end



# ── Validation-aware training, reproducibility, normalizer continuity ────────

function _train_flow(arch=:RealNVP, T=Float32; dims=2, seed=1)
    kw = (; architecture=arch, n_transforms=2, dist_dims=dims, hidden_layer_sizes=[8],
          rng=Random.MersenneTwister(seed))
    return arch == :NSF ? FlowDistribution(T; identity_init=false, kw..., K=4, tail_bound=3.0) : FlowDistribution(T; identity_init=false, kw...)
end

_train_data(T, n; dims=2, seed=2) = T.(randn(Random.MersenneTwister(seed), dims, n) .* [1.0, 3.0][1:dims] .+ [2.0, -1.0][1:dims])

function _flat(ps)
    f = SimpleFlows._flatten_params(ps)
    return reduce(vcat, [vec(f[k]) for k in sort!(collect(keys(f)))])
end

@testset "train_flow!: history, validation, best weights" begin
    for arch in (:RealNVP, :NSF, :MAF), T in (Float32, Float64)
        data = _train_data(T, 60); val = _train_data(T, 25; seed=3)
        flow = _train_flow(arch, T)
        hist = Any[]
        ret = train_flow!(flow, data; n_epochs=6, batch_size=20, verbose=false,
                          validation_data=val, eval_every=2, history=hist,
                          rng=Random.MersenneTwister(5))
        @test ret === flow
        @test [h.epoch for h in hist] == [0, 2, 4, 6]
        @test all(h -> isfinite(h.train_nll) && isfinite(h.validation_nll), hist)
        best = argmin([h.validation_nll for h in hist])
        @test hist[end].best_epoch == hist[best].epoch
        @test all(i -> hist[i].best_epoch == hist[argmin([h.validation_nll for h in hist[1:i]])].epoch, eachindex(hist))
        # the restored weights give exactly the recorded best validation NLL (direct logpdf)
        @test -mean(Float64.(logpdf(flow, val))) ≈ hist[best].validation_nll rtol=1e-4
        if hist[end].best_epoch == 6
            @test -mean(Float64.(logpdf(flow, data))) ≈ hist[end].train_nll rtol=1e-4
        end
    end

    @testset "full losses equal direct logpdf (no restoring needed)" begin
        for nrm in (:minmax, :whiten)
            T = Float64
            data = _train_data(T, 50); val = _train_data(T, 17; seed=4)
            flow = _train_flow(:RealNVP, T)
            hist = Any[]
            train_flow!(flow, data; n_epochs=0, verbose=false, normalization=nrm,
                        validation_data=val, history=hist, batch_size=7)   # epoch 0 only
            @test length(hist) == 1 && hist[1].epoch == 0 && hist[1].best_epoch == 0
            @test hist[1].train_nll ≈ -mean(logpdf(flow, data)) rtol=1e-12
            @test hist[1].validation_nll ≈ -mean(logpdf(flow, val)) rtol=1e-12
        end
    end

    @testset "initial model is a candidate; restored copy is independent" begin
        data = _train_data(Float32, 40); val = _train_data(Float32, 20; seed=3)
        flow = _train_flow()
        train_flow!(flow, data; n_epochs=0, verbose=false)   # fit the normalizer only
        ps0 = _flat(flow.ps)
        hist = Any[]
        # gradient *ascent* makes the validation NLL worse, so the initial weights must win
        train_flow!(flow, data; n_epochs=3, opt=Optimisers.Descent(-0.01f0), batch_size=20, verbose=false,
                    validation_data=val, history=hist, rng=Random.MersenneTwister(1))
        @test hist[end].validation_nll > hist[1].validation_nll
        @test all(h -> h.best_epoch == 0, hist)
        @test _flat(flow.ps) == ps0
        # the restored copy does not alias what training/optimizer arrays touch afterwards
        snapshot = _flat(flow.ps)
        flow_b = deepcopy(flow)
        train_flow!(flow_b, data; n_epochs=2, batch_size=20, verbose=false, rng=Random.MersenneTwister(2))
        @test _flat(flow.ps) == snapshot
        @test _flat(flow_b.ps) != snapshot
    end

    @testset "no validation keeps final weights" begin
        data = _train_data(Float32, 40)
        f1 = _train_flow(); f2 = _train_flow()
        hist = Any[]
        train_flow!(f1, data; n_epochs=3, batch_size=20, verbose=false, history=hist, rng=Random.MersenneTwister(1))
        train_flow!(f2, data; n_epochs=3, batch_size=20, verbose=false, rng=Random.MersenneTwister(1))
        @test _flat(f1.ps) == _flat(f2.ps)               # history changes nothing
        @test all(h -> h.validation_nll === nothing && h.best_epoch === nothing, hist)
        @test [h.epoch for h in hist] == [0, 1, 2, 3]
        @test hist[end].train_nll ≈ -mean(Float64.(logpdf(f1, data))) rtol=1e-4
    end
end

@testset "train_flow!: reproducibility" begin
    data = _train_data(Float64, 50); val = _train_data(Float64, 20; seed=3)
    run() = begin
        f = _train_flow(:NSF, Float64); h = Any[]
        train_flow!(f, data; n_epochs=4, batch_size=16, verbose=false, validation_data=val,
                    history=h, rng=Random.MersenneTwister(77))
        (f, h)
    end
    f1, h1 = run(); f2, h2 = run()
    @test _flat(f1.ps) == _flat(f2.ps)
    @test h1 == h2
    f3 = _train_flow(:NSF, Float64)
    train_flow!(f3, data; n_epochs=4, batch_size=16, verbose=false, rng=Random.MersenneTwister(78))
    @test _flat(f3.ps) != _flat(f1.ps)                   # a different shuffle seed changes the result
end

@testset "train_flow!: normalizer continuity" begin
    data = _train_data(Float64, 50); other = _train_data(Float64, 50; seed=9) .* 5 .+ 7
    flow = _train_flow(:RealNVP, Float64)
    train_flow!(flow, data; n_epochs=1, verbose=false)
    @test flow.normalizer isa MinMaxNormalizer
    n1 = flow.normalizer
    train_flow!(flow, other; n_epochs=1, verbose=false)                          # default: reuse
    @test flow.normalizer === n1
    train_flow!(flow, other; n_epochs=1, verbose=false, normalization=:minmax)   # matching kind: reuse
    @test flow.normalizer === n1
    ps = deepcopy(flow.ps)
    @test_throws ArgumentError train_flow!(flow, other; n_epochs=1, verbose=false, normalization=:whiten)
    @test flow.normalizer === n1 && _flat(flow.ps) == _flat(ps)
    train_flow!(flow, other; n_epochs=1, verbose=false, refit_normalizer=true)   # same kind, deliberate refit
    @test flow.normalizer isa MinMaxNormalizer && flow.normalizer !== n1
    @test flow.normalizer.x_max == vec(maximum(other; dims=2))
    train_flow!(flow, other; n_epochs=1, verbose=false, normalization=:whiten, refit_normalizer=true)
    @test flow.normalizer isa WhiteningNormalizer
    nw = flow.normalizer
    train_flow!(flow, data; n_epochs=1, verbose=false)
    @test flow.normalizer === nw
    # validation is normalised with the training normalizer (reused one here)
    hist = Any[]
    val = _train_data(Float64, 11; seed=6)
    train_flow!(flow, data; n_epochs=0, verbose=false, validation_data=val, history=hist)
    @test hist[1].validation_nll ≈ -mean(logpdf(flow, val)) rtol=1e-12
end

@testset "train_flow!: invalid inputs fail before mutation; failures restore state" begin
    data = _train_data(Float32, 30)
    flow = _train_flow()
    bad(; d=data, kw...) = () -> train_flow!(flow, d; verbose=false, n_epochs=1, kw...)
    snap() = (_flat(flow.ps), flow.normalizer)
    s0 = snap()
    nan = copy(data); nan[1, 3] = NaN
    for f in (bad(; d=data[1:1, :]), bad(; d=zeros(Float32, 2, 0)), bad(; d=nan),
              bad(; d=Complex.(data)),
              bad(validation_data=zeros(Float32, 3, 4)), bad(validation_data=zeros(Float32, 2, 0)),
              bad(validation_data=nan), bad(eval_every=0), bad(batch_size=0),
              bad(n_epochs=-1), bad(normalization=:zca), bad(lr=-1.0), bad(history=Int[]),
              bad(lr=NaN), bad(lr=1e-60), bad(lr=1e60))
        @test_throws ArgumentError f()
    end
    @test snap() == s0 && flow.normalizer === nothing        # nothing was mutated, nothing fitted

    # singular data for whitening fails in the fit, before any mutation
    sing = copy(data); sing[2, :] .= 2 .* sing[1, :]
    @test_throws ArgumentError train_flow!(flow, sing; verbose=false, n_epochs=1, normalization=:whiten)
    @test snap() == s0

    # non-finite loss/gradient: stops before the update and restores the entry state
    train_flow!(flow, data; n_epochs=1, verbose=false)
    ps_before = _flat(flow.ps); nrm_before = flow.normalizer
    bad_ps = Lux.fmap(x -> x isa AbstractArray ? fill!(copy(x), NaN32) : x, flow.ps)
    flow.ps = bad_ps
    @test_throws ErrorException train_flow!(flow, data; n_epochs=2, verbose=false)                  # loss check
    @test_throws ErrorException train_flow!(flow, data; n_epochs=2, verbose=false, history=Any[])   # epoch-0 evaluation
    @test all(isnan, _flat(flow.ps)) && flow.normalizer === nrm_before   # not "repaired", not refitted

    # failure after some successful updates restores the weights at entry
    flow2 = _train_flow(); train_flow!(flow2, data; n_epochs=1, verbose=false)
    ps2 = _flat(flow2.ps); nrm2 = flow2.normalizer
    err = try
        train_flow!(flow2, data; n_epochs=3, batch_size=10, verbose=false, opt=Optimisers.Descent(1f30),
                    refit_normalizer=true)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test _flat(flow2.ps) == ps2
    @test flow2.normalizer === nrm2
end

@testset "train_flow!: last-update check, verbose metrics, uneven batches" begin
    # An update that makes parameters non-finite is caught even when it is the last one
    # and nothing else evaluates the loss afterwards.
    data = _train_data(Float32, 8)
    flow = _train_flow(); train_flow!(flow, data; n_epochs=0, verbose=false)
    ps0 = _flat(flow.ps); nrm0 = flow.normalizer
    @test_throws ErrorException train_flow!(flow, data; n_epochs=1, batch_size=8, verbose=false,
                                            opt=Optimisers.Descent(Inf32))
    @test _flat(flow.ps) == ps0 && flow.normalizer === nrm0 && all(isfinite, _flat(flow.ps))
    # custom lr not representable after conversion fails before anything is touched
    f3 = _train_flow()
    @test_throws ArgumentError train_flow!(f3, data; n_epochs=1, verbose=false, lr=1e-60)
    @test f3.normalizer === nothing
    # unsupported leaves are an error, never silently "finite"
    @test_throws ErrorException SimpleFlows._all_finite((a=[1.0f0], b="text"))
    @test SimpleFlows._all_finite((a=[1.0f0], b=(c=nothing, d=2.0)))
    @test !SimpleFlows._all_finite((a=[1.0f0, NaN32],))

    # verbose-only run: logs the fixed-weight full-dataset physical NLL (uneven final batches)
    for nrm in (:minmax, :whiten)
        d = _train_data(Float64, 23); val = _train_data(Float64, 10; seed=4)
        f = _train_flow(:RealNVP, Float64)
        logger = Test.TestLogger()
        Base.CoreLogging.with_logger(logger) do
            train_flow!(f, d; n_epochs=4, batch_size=7, verbose=true, eval_every=2, normalization=nrm,
                        validation_data=val, rng=Random.MersenneTwister(3))
        end
        recs = [r for r in logger.logs if r.level == Base.CoreLogging.Info && haskey(Dict(r.kwargs), :train_nll)]
        @test [Dict(r.kwargs)[:epoch] for r in recs] == [0, 2, 4]
        last = Dict(recs[end].kwargs)
        be = last[:best_epoch]
        # the log is the fixed-weight full-data NLL at the (possibly restored) best weights if best==final
        if be == 4
            @test last[:train_nll] ≈ -mean(logpdf(f, d)) rtol=1e-12
            @test last[:validation_nll] ≈ -mean(logpdf(f, val)) rtol=1e-12
        end
    end
    # no validation, verbose only: final weights kept; logged NLL equals direct logpdf at those weights
    d = _train_data(Float64, 23)
    f = _train_flow(:RealNVP, Float64); g = _train_flow(:RealNVP, Float64)
    logger = Test.TestLogger()
    Base.CoreLogging.with_logger(logger) do
        train_flow!(f, d; n_epochs=2, batch_size=7, verbose=true, rng=Random.MersenneTwister(5))
    end
    train_flow!(g, d; n_epochs=2, batch_size=7, verbose=false, rng=Random.MersenneTwister(5))
    @test _flat(f.ps) == _flat(g.ps)
    recs = [Dict(r.kwargs) for r in logger.logs if haskey(Dict(r.kwargs), :train_nll)]
    @test [r[:epoch] for r in recs] == [0, 1, 2]
    @test recs[end][:train_nll] ≈ -mean(logpdf(f, d)) rtol=1e-12
    @test all(r -> r[:validation_nll] === nothing, recs)
    # history on a trained flow with uneven final batch equals direct logpdf at final weights
    h = Any[]
    f2 = _train_flow(:NSF, Float64)
    d2 = _train_data(Float64, 23)
    train_flow!(f2, d2; n_epochs=2, batch_size=7, verbose=false, history=h)
    @test h[end].train_nll ≈ -mean(logpdf(f2, d2)) rtol=1e-12
end
