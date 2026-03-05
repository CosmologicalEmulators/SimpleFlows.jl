@testset "FlowDistribution" begin
    using LinearAlgebra
    rng  = Random.MersenneTwister(3)
    dims = 4

    flow = FlowDistribution(; n_transforms=4, dist_dims=dims,
                               hidden_layer_sizes=[32, 32], rng)

    @testset "Interface" begin
        @test Distributions.length(flow) == dims

        x_vec = randn(rng, Float64, dims)
        lp = logpdf(flow, x_vec)
        @test isa(lp, Real)
        @test isfinite(lp)

        x_mat = randn(rng, Float32, dims, 16)
        lps = logpdf(flow, x_mat)
        @test length(lps) == 16
        @test all(isfinite, lps)
    end

    @testset "Sampling" begin
        s = rand(rng, flow)
        @test length(s) == dims

        S = Distributions.rand(rng, flow, 50)
        @test size(S) == (dims, 50)
    end

    @testset "NLL decreases after training" begin
        # Sample from a simple 4D Gaussian, train briefly
        target   = MvNormal(zeros(dims), I)
        data     = Float32.(rand(rng, target, 2000))

        lp_before = mean(logpdf(flow, data))
        train_flow!(flow, data; n_epochs=50, lr=1f-3, batch_size=256, verbose=false)
        lp_after  = mean(logpdf(flow, data))

        @test lp_after > lp_before
    end
end
