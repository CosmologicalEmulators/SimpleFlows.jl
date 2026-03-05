@testset "Save / Load round-trip" begin
    rng  = Random.MersenneTwister(4)
    dims = 3

    # Test backward-compat scalar API
    flow = FlowDistribution(; n_transforms=3, dist_dims=dims,
                               hidden_dims=16, n_layers=2, rng)

    # We don't train here. Training stability is tested in test_realnvp.jl.
    # The IO test only verifies that model serialization survives round-trip.

    # Save
    tmpdir = mktempdir()
    save_trained_flow(tmpdir, flow)

    @test isfile(joinpath(tmpdir, "flow_setup.json"))
    @test isfile(joinpath(tmpdir, "weights.npz"))

    # Load
    flow2 = load_trained_flow(tmpdir; rng)

    # Dimensions must match
    @test Distributions.length(flow2) == dims

    # logpdf must be identical at test points
    x_test = randn(rng, Float32, dims, 20)
    lp1 = logpdf(flow,  x_test)
    lp2 = logpdf(flow2, x_test)
    @test lp1 ≈ lp2  atol=1f-4

    # Also test the new per-layer vector API round-trips correctly
    flow3 = FlowDistribution(; n_transforms=2, dist_dims=dims,
                               hidden_layer_sizes=[32, 64, 32], rng)
    tmpdir2 = mktempdir()
    save_trained_flow(tmpdir2, flow3)
    flow4 = load_trained_flow(tmpdir2; rng)
    @test flow4.hidden_layer_sizes == [32, 64, 32]
    lp3 = logpdf(flow3, x_test)
    lp4 = logpdf(flow4, x_test)
    @test lp3 ≈ lp4  atol=1f-4
end
