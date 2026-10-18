using SimpleFlows, Test, Random, Distributions, Statistics, Zygote, Optimisers, Lux, JSON

function _init_flow(arch, T, init; dims=3, seed=1, hidden=[8, 8], n_transforms=3, activation=tanh, K=4)
    kw = (; architecture=arch, n_transforms, dist_dims=dims, hidden_layer_sizes=hidden, activation,
          identity_init=init, rng=Random.Xoshiro(seed))
    return arch == :NSF ? FlowDistribution(T; kw..., K, tail_bound=3.0) : FlowDistribution(T; kw...)
end

_std_normal_lp(X) = vec(sum(-0.5 .* (X .^ 2 .+ log(2 * Float64(π))); dims=1))
_flat_ps(x::AbstractArray) = vec(Array(x))
_flat_ps(x::NamedTuple) = reduce(vcat, map(_flat_ps, values(x)))
_heads(ps, arch) = arch == :MAF ? [last(values(m.layers)) for m in values(ps.mades)] : [last(values(c)) for c in values(ps.conditioners)]

@testset "identity_init: legacy mode reproduces the explicit Lux.setup pipeline exactly" begin
    for arch in (:RealNVP, :NSF, :MAF), T in (Float32, Float64)
        flow = _init_flow(arch, T, false)
        model = arch == :RealNVP ? RealNVP(; n_transforms=3, dist_dims=3, hidden_layer_sizes=[8, 8], activation=tanh) :
                arch == :NSF ? NeuralSplineFlow(; n_transforms=3, dist_dims=3, hidden_layer_sizes=[8, 8], K=4, tail_bound=3.0, activation=tanh) :
                MaskedAutoregressiveFlow(; n_transforms=3, dist_dims=3, hidden_layer_sizes=[8, 8], activation=tanh)
        ps, _ = Lux.setup(Random.Xoshiro(1), model)
        ps = Lux.fmap(x -> x isa AbstractArray ? T.(x) : x, ps)
        @test _flat_ps(flow.ps) == _flat_ps(ps)

        fid = _init_flow(arch, T, true)
        # same random stream: hidden layers identical, only the output heads differ
        for (h_id, h_leg) in zip(_heads(fid.ps, arch), _heads(flow.ps, arch))
            @test all(iszero, h_id.weight)
            @test any(!iszero, h_leg.weight)
        end
        hid(ps) = arch == :MAF ? [layer for m in values(ps.mades) for layer in values(m.layers)][1:2:end] : nothing
        if arch != :MAF
            c_id = values(fid.ps.conditioners); c_leg = values(flow.ps.conditioners)
            @test all(_flat_ps(a.layer_1) == _flat_ps(b.layer_1) && _flat_ps(a.layer_2) == _flat_ps(b.layer_2) for (a, b) in zip(c_id, c_leg))
        else
            @test all(_flat_ps(a.layers.layer_1) == _flat_ps(b.layers.layer_1) && _flat_ps(a.layers.layer_2) == _flat_ps(b.layers.layer_2)
                      for (a, b) in zip(values(fid.ps.mades), values(flow.ps.mades)))
        end
        # determinism
        @test _flat_ps(_init_flow(arch, T, true).ps) == _flat_ps(fid.ps)
    end
end

@testset "identity_init: default density is the standard Gaussian (all architectures, precisions, odd/even dims)" begin
    for arch in (:RealNVP, :NSF, :MAF), T in (Float32, Float64), dims in (3, 4)
        flow = _init_flow(arch, T, true; dims)
        rng = Random.Xoshiro(3)
        X = randn(rng, T, dims, 200); X[1, 1] = 11; X[end, 2] = -11     # whitened-scale outliers at |x| = 11
        lp = logpdf(flow, X)
        @test all(isfinite, lp)
        @test Float64.(lp) ≈ _std_normal_lp(Float64.(X)) rtol=(T == Float64 ? 1e-12 : 1e-5) atol=(T == Float64 ? 1e-10 : 2e-5)
        # single-vector method and a fitted normalizer: log p(x) = log N(L^-1 (x - mu)) + log|det L^-1|
        @test logpdf(flow, X[:, 3]) ≈ lp[3]
        data = T.(randn(rng, dims, 100) .* [1.0, 3.0, 0.5, 2.0][1:dims] .+ [2.0, -1.0, 0.0, 5.0][1:dims])
        flow.normalizer = WhiteningNormalizer(data)
        z = SimpleFlows.normalize(flow.normalizer, X)
        @test Float64.(logpdf(flow, X)) ≈ _std_normal_lp(Float64.(z)) .+ Float64(flow.normalizer.log_jac) rtol=(T == Float64 ? 1e-12 : 1e-4) atol=(T == Float64 ? 1e-10 : 1e-4)
        # samples are consistent with the density (invertible, finite)
        s = rand(Random.Xoshiro(4), flow, 50)
        @test all(isfinite, s)
        @test Float64.(logpdf(flow, s)) ≈ _std_normal_lp(Float64.(SimpleFlows.normalize(flow.normalizer, s))) .+ Float64(flow.normalizer.log_jac) rtol=(T == Float64 ? 1e-10 : 1e-3) atol=(T == Float64 ? 1e-9 : 1e-3)
    end
end

@testset "identity_init: the legacy random RealNVP overflows on whitened-scale tails, the default does not" begin
    rng = Random.Xoshiro(1)
    X = randn(rng, Float32, 8, 4000)
    X[rand(rng, 1:8, 40) .+ 8 .* (rand(rng, 1:4000, 40) .- 1)] .= 11f0
    legacy = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=3, dist_dims=8, hidden_layer_sizes=[32, 32],
                              identity_init=false, rng=Random.Xoshiro(1))
    default = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=3, dist_dims=8, hidden_layer_sizes=[32, 32],
                               rng=Random.Xoshiro(1))
    @test count(!isfinite, logpdf(legacy, X)) > 0               # recorded: 10 non-finite of 4000 for this seed
    @test all(isfinite, logpdf(default, X))
    @test Float64.(logpdf(default, X)) ≈ _std_normal_lp(Float64.(X)) rtol=1e-5
end

@testset "identity_init: head gradients are non-zero, training learns (not a vacuous identity model)" begin
    for arch in (:RealNVP, :NSF, :MAF)
        T = Float64
        flow = _init_flow(arch, T, true; dims=3)
        rng = Random.Xoshiro(7)
        X = randn(rng, T, 3, 64) .* [1.0, 2.0, 0.5] .+ [1.0, -2.0, 3.0]      # clearly not N(0, I)
        loss(ps) = -mean(SimpleFlows.log_prob(flow.model, ps, flow.st, X))
        g = Zygote.gradient(loss, flow.ps)[1]
        heads = _heads(g, arch)
        @test all(any(!iszero, h.weight) && any(!iszero, h.bias) for h in heads)      # the first update moves every head
        # hidden layers see exactly zero gradient while the head is zero (expected), then learn
        hidden_of(gr) = arch == :MAF ? first(values(gr.mades)).layers.layer_1 : first(values(gr.conditioners)).layer_1
        @test all(iszero, hidden_of(g).weight)
        hist = Any[]
        train_flow!(flow, X; n_epochs=40, batch_size=32, lr=1e-2, verbose=false, history=hist, normalization=:minmax)
        @test hist[end].train_nll < hist[1].train_nll - 0.05                          # density improved
        g2 = Zygote.gradient(ps -> -mean(SimpleFlows.log_prob(flow.model, ps, flow.st,
                                         SimpleFlows.normalize(flow.normalizer, X))), flow.ps)[1]
        @test any(!iszero, hidden_of(g2).weight)                                       # hidden layers now learn too
    end
end

@testset "identity_init: save/load with custom activations in both initialization modes" begin
    for arch in (:RealNVP, :NSF, :MAF), (aname, act) in (("tanh", tanh), ("relu", relu)), init in (true, false)
        flow = _init_flow(arch, Float32, init; activation=act)
        X = Float32.(randn(Random.Xoshiro(2), 3, 40) .* 1.5 .+ 0.5)
        train_flow!(flow, X; n_epochs=2, batch_size=20, verbose=false, rng=Random.Xoshiro(1))
        dir = mktempdir()
        save_trained_flow(dir, flow)
        f2 = load_trained_flow(dir)
        @test SimpleFlows._first_activation(f2.model) === act
        @test logpdf(f2, X) == logpdf(flow, X)
        @test rand(Random.Xoshiro(5), f2, 5) == rand(Random.Xoshiro(5), flow, 5)
    end
end
