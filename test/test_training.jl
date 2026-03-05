using SimpleFlows
using Test
using Random
using Optimisers

@testset "Custom Optimizer & Training" begin
    rng = Random.default_rng()
    Random.seed!(rng, 10)
    
    dims = 2
    N = 100
    data = randn(Float32, dims, N)
    
    flow = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=2, dist_dims=dims,
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
    flow = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=2, dist_dims=dims,
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
        flow = FlowDistribution(Float32; architecture=arch, n_transforms=1, dist_dims=dims,
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


