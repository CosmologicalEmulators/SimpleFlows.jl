function _whiten_flow(arch, T; dims=3, seed=21)
    rng = Random.MersenneTwister(seed)
    kw = (; architecture=arch, n_transforms=2, dist_dims=dims,
          hidden_layer_sizes=[8, 8], activation=tanh, rng)
    return arch == :NSF ? FlowDistribution(T; identity_init=false, kw..., K=4, tail_bound=2.0) :
                          FlowDistribution(T; identity_init=false, kw...)
end

@testset "WhiteningNormalizer" begin
    for T in (Float32, Float64), dims in (2, 3)
        rng = Random.MersenneTwister(1)
        A = T.([1.5 0.2 0.0; -0.4 0.7 0.3; 0.1 0.0 2.0])[1:dims, 1:dims]
        μ0 = T.([10, -3, 0.5])[1:dims]
        x = A * randn(rng, T, dims, 5000) .+ μ0
        n = WhiteningNormalizer(x)
        @test n isa WhiteningNormalizer{T}
        tol = T == Float32 ? 1f-4 : 1e-10

        # fitted moments, Σ = L Lᵀ
        xs = Float64.(x)
        Σ = cov(xs; dims=2)
        @test Float64.(n.mean) ≈ vec(mean(xs; dims=2)) rtol=tol
        @test Float64.(n.chol * n.chol') ≈ Σ rtol=(T == Float32 ? 1e-5 : 1e-12)
        @test istril(n.chol)

        # whitened data has zero mean, identity covariance
        z = SimpleFlows.normalize(n, x)
        @test eltype(z) == T
        @test maximum(abs, mean(Float64.(z); dims=2)) < (T == Float32 ? 1e-3 : 1e-10)
        @test cov(Float64.(z); dims=2) ≈ I atol=(T == Float32 ? 1e-3 : 1e-10)

        # inverse, vector forms
        @test SimpleFlows.denormalize(n, z) ≈ x rtol=tol atol=tol * 10
        @test SimpleFlows.normalize(n, x[:, 1]) ≈ z[:, 1]
        @test SimpleFlows.denormalize(n, z[:, 1]) ≈ x[:, 1] rtol=tol atol=tol * 10

        # log_jac = -sum(log(diag L)) = -½ logdet Σ
        @test Float64(n.log_jac) ≈ -0.5 * logdet(Σ) rtol=(T == Float32 ? 1e-4 : 1e-10)

        # analytic density: if the base density is N(0, I), the whitened flow of
        # the normalizer alone gives log N(x; μ, Σ). Check the Gaussian identity
        # log N(x;μ,Σ) = log N(z;0,I) + log_jac.
        xt = x[:, 1:5]
        lp_gauss = [logpdf(MvNormal(Float64.(n.mean), Symmetric(Float64.(n.chol * n.chol'))), Float64.(xt[:, j])) for j in 1:5]
        zt = SimpleFlows.normalize(n, xt)
        lp_via = vec(sum(SimpleFlows.gaussian_logpdf.(zt); dims=1)) .+ n.log_jac
        @test Float64.(lp_via) ≈ lp_gauss rtol=(T == Float32 ? 1e-4 : 1e-10)
    end

    @testset "rejection" begin
        rng = Random.MersenneTwister(2)
        good = randn(rng, 3, 20)
        bad = copy(good); bad[2, 3] = NaN
        @test_throws ArgumentError WhiteningNormalizer(bad)
        bad[2, 3] = Inf
        @test_throws ArgumentError WhiteningNormalizer(bad)
        @test_throws ArgumentError WhiteningNormalizer(good[:, 1:3])   # n <= d
        @test_throws ArgumentError WhiteningNormalizer(good[:, 1:1])
        sing = copy(good); sing[3, :] .= 2 .* sing[1, :] .+ 1          # exact collinearity
        @test_throws ArgumentError WhiteningNormalizer(sing)
        const_dim = copy(good); const_dim[2, :] .= 4.0
        @test_throws ArgumentError WhiteningNormalizer(const_dim)
        nearly = copy(good); nearly[3, :] .= nearly[1, :] .+ 1e-9 .* good[3, :]
        @test_throws ArgumentError WhiteningNormalizer(Float32.(nearly))
        @test_throws ArgumentError WhiteningNormalizer(zeros(Float64, 0, 5))      # d = 0
        @test_throws ArgumentError WhiteningNormalizer(Int.(round.(10 .* good)))  # no silent coercion
        @test_throws ArgumentError WhiteningNormalizer(BigFloat.(good))
        @test_throws ArgumentError WhiteningNormalizer(Float16.(good))
        # exact collinearity stays rejected even at wildly different scales
        sc = [1e-6, 1.0, 1e6] .* good; sc[3, :] .= 1e6 .* sc[1, :] ./ 1e-6 .+ 3e6
        @test_throws ArgumentError WhiteningNormalizer(sc)
        @test_throws ArgumentError WhiteningNormalizer(Float32.(sc))
        # ill-conditioned but valid (cond ~ 2.6e5 as in real posteriors) is accepted in both precisions
        s = [1.0, 1.0, 1.0 / 512]
        ill = s .* good
        @test WhiteningNormalizer(ill) isa WhiteningNormalizer{Float64}
        @test WhiteningNormalizer(Float32.(ill)) isa WhiteningNormalizer{Float32}
    end

    @testset "unit invariance: independent coordinates, scales 1e-6, 1, 1e6" begin
        scales = [1e-6, 1.0, 1e6]
        for T in (Float32, Float64)
            rng = Random.MersenneTwister(8)
            x = T.(scales .* randn(rng, 3, 4000) .+ [0.0, 5.0, -1e6])
            n = WhiteningNormalizer(x)
            @test n isa WhiteningNormalizer{T}
            # independent coordinates: sample correlations are O(1/sqrt(n)), so the factor
            # is nearly diagonal with the sample std, and L Lᵀ reproduces the sample covariance
            @test Float64.(diag(n.chol)) ≈ vec(std(Float64.(x); dims=2)) rtol=2e-3
            Lf = Float64.(n.chol)
            @test (Lf * Lf') ./ (vec(std(Float64.(x); dims=2)) * vec(std(Float64.(x); dims=2))') ≈ cor(Float64.(x); dims=2) rtol=(T == Float32 ? 1e-4 : 1e-10) atol=1e-5
            @test all(isfinite, n.chol)
            z = SimpleFlows.normalize(n, x)
            @test eltype(z) == T
            # whitened covariance is the identity (checked in Float64)
            @test cov(Float64.(z); dims=2) ≈ I atol=(T == Float32 ? 5e-3 : 1e-9)
            @test maximum(abs, mean(Float64.(z); dims=2)) < (T == Float32 ? 5e-3 : 1e-9)
            # round trip, relative to each coordinate's scale
            xb = SimpleFlows.denormalize(n, z)
            @test all(abs.(Float64.(xb .- x)) .<= (T == Float32 ? 1e-4 : 1e-10) .* scales .+ (T == Float32 ? 1e-6 : 0) .* abs.(Float64.(x)))
            @test Float64(n.log_jac) ≈ -sum(log, Float64.(diag(n.chol))) rtol=1e-6
            # exact (unit-free) Gaussian identity for the physical-coordinate density
            lp = vec(sum(SimpleFlows.gaussian_logpdf.(z[:, 1:3]); dims=1)) .+ n.log_jac
            Σ = Symmetric(Float64.(n.chol) * Float64.(n.chol)')
            ref = [logpdf(MvNormal(Float64.(n.mean), Σ), Float64.(x[:, j])) for j in 1:3]
            @test Float64.(lp) ≈ ref rtol=(T == Float32 ? 1e-3 : 1e-9)
        end
    end

    @testset "train_flow! selector" begin
        data = Float32.(randn(Random.MersenneTwister(3), 3, 80) .* [1.0, 2.0, 0.5] .+ [3.0, 0.0, -1.0])
        flow = _whiten_flow(:RealNVP, Float32)
        train_flow!(flow, data; n_epochs=2, batch_size=40, verbose=false)
        @test flow.normalizer isa MinMaxNormalizer{Float32}          # default
        @test_throws ArgumentError train_flow!(flow, data; n_epochs=2, batch_size=40, verbose=false, normalization=:whiten)  # conflicts with fitted minmax
        @test flow.normalizer isa MinMaxNormalizer{Float32}
        train_flow!(flow, data; n_epochs=2, batch_size=40, verbose=false, normalization=:whiten, refit_normalizer=true)
        @test flow.normalizer isa WhiteningNormalizer{Float32}
        @test_throws ArgumentError train_flow!(flow, data; n_epochs=1, verbose=false, normalization=:bogus)
        @test_throws ArgumentError train_flow!(flow, data[:, 1:2]; n_epochs=1, verbose=false, normalization=:whiten, refit_normalizer=true)
        # logpdf includes the whitening Jacobian
        lp_z = SimpleFlows.log_prob(flow.model, flow.ps, flow.st, SimpleFlows.normalize(flow.normalizer, data[:, 1:4]))
        @test logpdf(flow, data[:, 1:4]) ≈ lp_z .+ flow.normalizer.log_jac
    end
end

@testset "Whitening persistence: all architectures" begin
    names = ["omega_m", "h", "b1"]
    for T in (Float32, Float64), arch in (:RealNVP, :NSF, :MAF)
        @testset "$arch $T" begin
            flow = _whiten_flow(arch, T)
            data = T.([2.0 0.3 0.0; 0.1 1.0 0.2; 0.0 0.4 0.5] * randn(Random.MersenneTwister(5), 3, 80) .+ [1.0, -2.0, 0.0])
            train_flow!(flow, data; n_epochs=2, batch_size=40, verbose=false, normalization=:whiten)
            dir = mktempdir()
            flow.parameter_names = names
            save_trained_flow(dir, flow)
            setup = JSON.parsefile(joinpath(dir, "flow_setup.json"))
            @test setup["format_version"] == 3
            @test setup["normalizer"] == "whiten"
            @test load_parameter_names(dir) == names

            flow2 = load_trained_flow(dir)
            @test flow2.parameter_names == names
            # names survive load -> save -> load without being passed again
            dir2 = mktempdir(); save_trained_flow(dir2, flow2)
            @test load_parameter_names(dir2) == names
            @test load_trained_flow(dir2).parameter_names == names
            @test flow2.normalizer isa WhiteningNormalizer{T}
            @test flow2.normalizer.mean == flow.normalizer.mean
            @test flow2.normalizer.chol == flow.normalizer.chol
            @test flow2.normalizer.log_jac == flow.normalizer.log_jac
            x = T.(randn(Random.MersenneTwister(6), 3, 8))
            @test logpdf(flow2, x) == logpdf(flow, x)
            @test Zygote.gradient(v -> logpdf(flow, v), x[:, 1])[1] == Zygote.gradient(v -> logpdf(flow2, v), x[:, 1])[1]
            gfd = ForwardDiff.gradient(v -> logpdf(flow2, v), x[:, 1])
            @test gfd ≈ Zygote.gradient(v -> logpdf(flow2, v), x[:, 1])[1] rtol=(T == Float32 ? 2e-3 : 1e-8) atol=(T == Float32 ? 1e-3 : 1e-9)
            @test rand(Random.MersenneTwister(9), flow, 6) == rand(Random.MersenneTwister(9), flow2, 6)
            # samples live in physical coordinates: sample moments near the data moments are not required,
            # but denormalize∘normalize must be consistent
            @test SimpleFlows.denormalize(flow2.normalizer, SimpleFlows.normalize(flow2.normalizer, x)) ≈ x rtol=(T == Float32 ? 1e-4 : 1e-10) atol=(T == Float32 ? 1e-4 : 1e-10)
        end
    end
end

@testset "Artifact metadata edge cases" begin
    flow = _whiten_flow(:RealNVP, Float32)
    train_flow!(flow, Float32.(randn(Random.MersenneTwister(1), 3, 30)); n_epochs=1, verbose=false, normalization=:whiten)
    dir = mktempdir()
    save_trained_flow(dir, flow)
    @test load_parameter_names(dir) === nothing
    @test load_trained_flow(dir).parameter_names === nothing
    @test_throws ArgumentError FlowDistribution(Float32; identity_init=false, n_transforms=1, dist_dims=3, hidden_layer_sizes=[4], parameter_names=["a", "b"])
    @test_throws ArgumentError FlowDistribution(Float32; identity_init=false, n_transforms=1, dist_dims=3, hidden_layer_sizes=[4], parameter_names=["a", "b", "a"])
    @test FlowDistribution(Float32; identity_init=false, n_transforms=1, dist_dims=2, hidden_layer_sizes=[4], parameter_names=["a", "b"]).parameter_names == ["a", "b"]
    # parameter_names validation happens before anything is written
    bad = joinpath(mktempdir(), "o")
    @test_throws ArgumentError save_trained_flow(bad, flow; parameter_names=["a", "b"])
    @test_throws ArgumentError save_trained_flow(bad, flow; parameter_names=["a", "a", "b"])
    @test !ispath(bad)

    file = joinpath(dir, "flow_setup.json")
    good = JSON.parsefile(file)
    function with_setup(f)
        d = deepcopy(good); f(d)
        open(file, "w") do io; JSON.print(io, d, 4); end
        return dir
    end
    @test_throws ArgumentError load_trained_flow(with_setup(d -> d["normalizer"] = "zca"))
    @test_throws ArgumentError load_trained_flow(with_setup(d -> d["normalizer"] = "minmax"))   # arrays mismatch
    with_setup(d -> d["parameter_names"] = ["a"])
    @test_throws ArgumentError load_parameter_names(dir)
    with_setup(d -> d["parameter_names"] = ["a", "b", "c"])
    @test load_parameter_names(dir) == ["a", "b", "c"]

    # Corrupt whitening arrays are rejected
    flat = NPZ.npzread(joinpath(dir, "weights.npz"))
    flat["normalizer_chol"] = flat["normalizer_chol"] .+ Float32.(triu(ones(3, 3), 1))
    NPZ.npzwrite(joinpath(dir, "weights.npz"), flat)
    @test_throws ArgumentError load_trained_flow(with_setup(identity))

    # Explicit format_version = 1 has legacy semantics (no precision/activation/normalizer keys needed)
    flow = _whiten_flow(:RealNVP, Float64)
    gflow = FlowDistribution(Float64; identity_init=false, architecture=:RealNVP, n_transforms=2, dist_dims=3,
                             hidden_layer_sizes=[8, 8], activation=gelu, rng=Random.MersenneTwister(4))
    gflow.normalizer = MinMaxNormalizer(randn(Random.MersenneTwister(5), 3, 20))
    d1 = mktempdir()
    save_trained_flow(d1, gflow)
    f1 = joinpath(d1, "flow_setup.json")
    s1 = JSON.parsefile(f1)
    for k in ("precision", "activation", "normalizer"); delete!(s1, k); end
    s1["format_version"] = 1
    open(f1, "w") do io; JSON.print(io, s1, 4); end
    l1 = load_trained_flow(d1)
    @test l1 isa FlowDistribution{Float64}
    x = randn(Random.MersenneTwister(6), 3, 5)
    @test logpdf(l1, x) == logpdf(gflow, x)
    # version 0 / non-integer versions are rejected
    s1["format_version"] = 0
    open(f1, "w") do io; JSON.print(io, s1, 4); end
    @test_throws ArgumentError load_trained_flow(d1)
end
