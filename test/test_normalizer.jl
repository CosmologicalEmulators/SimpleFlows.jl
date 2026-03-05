@testset "MinMaxNormalizer" begin
    rng  = Random.MersenneTwister(7)
    dims = 4

    # Build some data with known range per dimension
    data = Float32.(randn(rng, dims, 200)) .* 3f0 .+ Float32.(1:dims)
    n    = MinMaxNormalizer(data)

    # x_min and x_max should bracket the data
    @test all(n.x_min .≤ minimum(data; dims=2)[:] .+ 1f-5)
    @test all(n.x_max .≥ maximum(data; dims=2)[:] .- 1f-5)

    # Forward transform: all values in [0, 1]
    z = SimpleFlows.normalize(n, data)
    @test all(z .≥ -1f-5)
    @test all(z .≤ 1 + 1f-5)

    # Round-trip: denormalize ∘ normalize ≈ identity
    data_back = SimpleFlows.denormalize(n, z)
    @test data_back ≈ data  atol=1f-5

    # Jacobian correction: log_jac = sum(-log(x_max - x_min))
    expected_log_jac = Float32(sum(-log.(n.x_max .- n.x_min)))
    @test n.log_jac ≈ expected_log_jac  atol=1f-5

    # Verify logpdf correction is applied
    flow = FlowDistribution(; n_transforms=2, dist_dims=dims,
                               hidden_dims=16, n_layers=2, rng)
    train_flow!(flow, data; n_epochs=5, lr=1f-3, verbose=false)

    # Check that normalizer was fitted and attached
    @test !isnothing(flow.normalizer)

    # Manually compute: log p(x) = log p_z(normalize(x)) + log_jac
    x_test    = data[:, 1:5]
    z_test    = SimpleFlows.normalize(flow.normalizer, Float32.(x_test))
    lp_z      = SimpleFlows.log_prob(flow.model, flow.ps, flow.st, z_test)
    lp_manual = lp_z .+ flow.normalizer.log_jac
    lp_api    = logpdf(flow, x_test)
    @test lp_api ≈ lp_manual  atol=1f-4
end
