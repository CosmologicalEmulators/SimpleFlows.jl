@testset "Save / Load round-trip" begin
    rng  = Random.MersenneTwister(4)
    dims = 3

    for arch in [:RealNVP, :NSF, :MAF]
        flow = if arch == :NSF
            FlowDistribution(Float32; architecture=arch, n_transforms=3, dist_dims=dims,
                             hidden_layer_sizes=[16, 16], K=4, tail_bound=2.0, rng=rng)
        else
            FlowDistribution(Float32; architecture=arch, n_transforms=3, dist_dims=dims,
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
    flow3 = FlowDistribution(; architecture=:RealNVP, n_transforms=2, dist_dims=dims,
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
