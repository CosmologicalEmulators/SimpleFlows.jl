using SimpleFlows
using Test
using Random
using Statistics
using Distributions

@testset "NSF Integrated Model" begin
    rng = Random.default_rng()
    Random.seed!(rng, 123)
    
    dist_dims = 2
    n_transforms = 2
    K = 4
    
    # 1. Initialization
    dist = FlowDistribution(Float32; 
        architecture=:NSF,
        n_transforms=n_transforms, 
        dist_dims=dist_dims,
        hidden_dims=16,
        n_layers=2,
        K=K
    )
    
    @test dist.model isa NeuralSplineFlow
    @test dist.model.K == K
    
    # 2. Forward pass (logpdf)
    x = randn(Float32, dist_dims, 10)
    lp = logpdf(dist, x)
    @test length(lp) == 10
    @test all(isfinite, lp)
    
    # 3. Sampling
    samples = rand(rng, dist, 100)
    @test size(samples) == (dist_dims, 100)
    @test all(isfinite, samples)
    
    # 4. Training (Smoke test)
    # Simple data: Gaussian
    target_data = randn(Float32, dist_dims, 200)
    
    # Initialize with normalizer
    train_flow!(dist, target_data; n_epochs=1, batch_size=100)
    
    @test !isnothing(dist.normalizer)
    
    # Verify logpdf still works after training
    lp_after = logpdf(dist, target_data[:, 1:10])
    @test length(lp_after) == 10
    @test all(isfinite, lp_after)
end
