@testset "Save / Load round-trip" begin
    rng  = Random.MersenneTwister(4)
    dims = 3

    for arch in [:RealNVP, :NSF, :MAF]
        flow = if arch == :NSF
            FlowDistribution(Float32; identity_init=false, architecture=arch, n_transforms=3, dist_dims=dims,
                             hidden_layer_sizes=[16, 16], K=4, tail_bound=2.0, rng=rng)
        else
            FlowDistribution(Float32; identity_init=false, architecture=arch, n_transforms=3, dist_dims=dims,
                             hidden_layer_sizes=[16, 16], rng=rng)
        end

        # Save
        tmpdir = mktempdir()
        save_trained_flow(tmpdir, flow)

        @test isfile(joinpath(tmpdir, "flow_setup.json"))
        @test isfile(joinpath(tmpdir, "weights.npz"))

        # Load
        flow2 = load_trained_flow(tmpdir; rng)

        # Dimensions must match
        @test Distributions.length(flow2) == dims
        @test typeof(flow2.model) == typeof(flow.model)

        # logpdf must be identical at test points
        x_test = randn(rng, Float32, dims, 20)
        lp1 = logpdf(flow,  x_test)
        lp2 = logpdf(flow2, x_test)
        @test lp1 ≈ lp2  atol=1f-4
    end

    # Test backward-compat scalar API and legacy JSON load
    flow3 = FlowDistribution(; identity_init=false, architecture=:RealNVP, n_transforms=2, dist_dims=dims,
                               hidden_dims=16, n_layers=2, rng=rng)
    tmpdir2 = mktempdir()
    save_trained_flow(tmpdir2, flow3)
    
    # Manually modify JSON to simulate old format
    setup_file = joinpath(tmpdir2, "flow_setup.json")
    setup_dict = JSON.parsefile(setup_file)
    delete!(setup_dict, "hidden_layer_sizes")
    setup_dict["hidden_dims"] = 16
    setup_dict["n_layers"] = 2
    open(setup_file, "w") do io
        JSON.print(io, setup_dict, 4)
    end
    
    flow4 = load_trained_flow(tmpdir2; rng)
    @test flow4.hidden_layer_sizes == [16, 16]
    x_test = randn(rng, Float32, dims, 20)
    lp3 = logpdf(flow3, x_test)
    lp4 = logpdf(flow4, x_test)
    @test lp3 ≈ lp4  atol=1f-4
end

# ── Stage 2: versioned artifacts, precision, activations ─────────────────────

function _io_flow(arch, T, activation; dims=3, seed=11)
    rng = Random.MersenneTwister(seed)
    kw = (; architecture=arch, n_transforms=2, dist_dims=dims,
          hidden_layer_sizes=[8, 8], activation, rng)
    return arch == :NSF ? FlowDistribution(T; identity_init=false, kw..., K=4, tail_bound=2.0) :
                          FlowDistribution(T; identity_init=false, kw...)
end

@testset "Artifact persistence: fitted flows, precision, activations" begin
    dims = 3
    for T in (Float32, Float64), arch in (:RealNVP, :NSF, :MAF),
        (aname, act) in (("gelu", gelu), ("relu", relu), ("tanh", tanh), ("swish", swish))

        @testset "$arch $T $aname" begin
            flow = _io_flow(arch, T, act)
            data = T.(2 .* randn(Random.MersenneTwister(5), dims, 64) .+ 1)
            train_flow!(flow, data; n_epochs=2, batch_size=32, verbose=false)
            @test flow.normalizer isa MinMaxNormalizer{T}

            dir = mktempdir()
            save_trained_flow(dir, flow)
            setup = JSON.parsefile(joinpath(dir, "flow_setup.json"))
            @test setup["format_version"] == SimpleFlows.ARTIFACT_FORMAT_VERSION
            @test setup["precision"] == string(T)
            @test setup["activation"] in ("gelu", "gelu_tanh", "relu", "tanh", "swish")

            flow2 = load_trained_flow(dir)
            @test flow2 isa FlowDistribution{T}
            @test typeof(flow2.model) == typeof(flow.model)
            @test flow2.normalizer isa MinMaxNormalizer{T}
            @test flow2.normalizer.x_min == flow.normalizer.x_min
            @test flow2.normalizer.log_jac == flow.normalizer.log_jac
            @test SimpleFlows._first_activation(flow2.model) === act

            # Compare against a fresh network built with the *default* activation
            # to make sure the loaded activation is not silently replaced.
            x = T.(randn(Random.MersenneTwister(6), dims, 10))
            @test logpdf(flow2, x) == logpdf(flow, x)
            if act !== gelu
                other = _io_flow(arch, T, gelu)
                other.ps = flow.ps; other.normalizer = flow.normalizer
                @test logpdf(other, x) != logpdf(flow, x)
            end

            x1 = x[:, 1]
            g1 = Zygote.gradient(v -> logpdf(flow, v), x1)[1]
            g2 = Zygote.gradient(v -> logpdf(flow2, v), x1)[1]
            @test g1 == g2

            s1 = rand(Random.MersenneTwister(99), flow, 7)
            s2 = rand(Random.MersenneTwister(99), flow2, 7)
            @test eltype(s2) == T
            @test s1 == s2
        end
    end
end

@testset "Artifact persistence: rejection and legacy" begin
    # Unsupported callables are rejected before anything is written.
    flow = _io_flow(:RealNVP, Float32, x -> x^2)
    dir = joinpath(mktempdir(), "out")
    @test_throws ArgumentError save_trained_flow(dir, flow)
    @test !ispath(dir)

    # Unknown activation names / versions / precisions in artifacts are rejected,
    # and nothing is evaluated.
    flow = _io_flow(:RealNVP, Float32, relu)
    dir = mktempdir()
    save_trained_flow(dir, flow)
    file = joinpath(dir, "flow_setup.json")
    good = JSON.parsefile(file)
    function with_setup(f)
        d = deepcopy(good); f(d)
        open(file, "w") do io; JSON.print(io, d, 4); end
        return dir
    end
    @test_throws ArgumentError load_trained_flow(with_setup(d -> d["activation"] = "sin"))
    @test_throws ArgumentError load_trained_flow(with_setup(d -> d["activation"] = "error(\"pwned\")"))
    @test_throws ArgumentError load_trained_flow(with_setup(d -> d["format_version"] = 99))
    @test_throws ArgumentError load_trained_flow(with_setup(d -> d["precision"] = "Float16"))

    # Legacy artifact (no version/precision): GELU assumed, precision inferred.
    for T in (Float32, Float64), arch in (:RealNVP, :NSF, :MAF)
        flow = _io_flow(arch, T, gelu)
        train_flow!(flow, T.(randn(Random.MersenneTwister(3), 3, 40)); n_epochs=1, verbose=false)
        dir = mktempdir()
        save_trained_flow(dir, flow)
        file = joinpath(dir, "flow_setup.json")
        d = JSON.parsefile(file)
        delete!(d, "format_version"); delete!(d, "precision"); d["activation"] = "gelu"
        open(file, "w") do io; JSON.print(io, d, 4); end
        flow2 = load_trained_flow(dir)
        @test flow2 isa FlowDistribution{T}
        x = T.(randn(Random.MersenneTwister(4), 3, 6))
        @test logpdf(flow2, x) == logpdf(flow, x)
    end
end
