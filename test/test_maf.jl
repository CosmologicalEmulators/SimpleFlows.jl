using SimpleFlows
using Test
using Random
using Lux
using ForwardDiff
using Zygote
using Bijectors
using Statistics
using LinearAlgebra

@testset "MAF Components" begin
    rng = Random.default_rng()
    Random.seed!(rng, 123)
    
    D = 4
    hidden = [16, 16]
    
    @testset "MADE Autoregressive Property" begin
        # Create a MADE network
        made = SimpleFlows.MADE(D, hidden, 2*D)
        ps, st = Lux.setup(rng, made)
        
        # Explicit Mask Multiplication (nflows inspired)
        # Extract masks from MaskedDense layers to ensure their product is strictly lower triangular.
        m1 = made.layers.layer_1.mask
        m2 = made.layers.layer_2.mask
        m3 = made.layers.layer_3.mask
        total_mask = m3 * m2 * m1
        
        # Block 1: m (1:D, 1:D)
        M_m = total_mask[1:D, :]
        @test isapprox(tril(M_m, -1), M_m, atol=1e-7)
        @test count(>(0), M_m) > 0 # make sure it's not identically zero
        
        # Block 2: log_alpha (D+1:2D, 1:D)
        M_la = total_mask[D+1:end, :]
        @test isapprox(tril(M_la, -1), M_la, atol=1e-7)
        @test count(>(0), M_la) > 0
        
        # We check the Jacobian of the outputs with respect to the inputs.
        
        function f(x)
            out, _ = Lux.apply(made, x, ps, st)
            return out
        end
        
        x0 = randn(Float32, D)
        J = ForwardDiff.jacobian(f, x0) # (2*D, D)
        
        # Block 1: m (1:D, 1:D)
        J_m = J[1:D, :]
        @test isapprox(tril(J_m, -1), J_m, atol=1e-7)
        
        # Block 2: log_alpha (D+1:2D, 1:D)
        J_la = J[D+1:end, :]
        @test isapprox(tril(J_la, -1), J_la, atol=1e-7)
    end
    
    @testset "Differentiability" begin
        made = SimpleFlows.MADE(D, hidden, 2*D)
        ps, st = Lux.setup(rng, made)
        x = randn(Float32, D, 5)
        
        # Zygote test
        gs = Zygote.gradient(ps) do p
            out, _ = Lux.apply(made, x, p, st)
            sum(out)
        end
        @test gs[1] !== nothing
        
        # ForwardDiff test
        function g(p_vec)
            # Reconstruct NamedTuple structure if possible, but easier to test w.r.t x
            return nothing
        end
        
        gx = ForwardDiff.gradient(x -> sum(Lux.apply(made, x, ps, st)[1]), x[:, 1])
        @test all(isfinite, gx)
    end
    
    @testset "MAFBijector Invertibility" begin
        made = SimpleFlows.MADE(D, hidden, 2*D)
        ps, st = Lux.setup(rng, made)
        bj = SimpleFlows.MAFBijector(made, ps, st)
        
        x = randn(Float32, D, 4)
        
        # Inverse (Density pass)
        u, lad_inv = Bijectors.with_logabsdet_jacobian(bj, x)
        
        # Forward (Sampling pass)
        x_rec, lad_fwd = SimpleFlows.forward_and_log_det(bj, u)
        
        @test x ≈ x_rec atol=1e-4
        @test lad_inv ≈ -lad_fwd atol=1e-4
    end
    
    @testset "Integrated MAF Model" begin
        dist = FlowDistribution(Float32; 
            architecture=:MAF,
            n_transforms=2, 
            dist_dims=D,
            hidden_dims=16,
            n_layers=2
        )
        
        x = randn(Float32, D, 10)
        lp = logpdf(dist, x)
        @test length(lp) == 10
        @test all(isfinite, lp)
        
        samples = rand(rng, dist, 10)
        @test size(samples) == (D, 10)
        @test all(isfinite, samples)
        
        # Training smoke test
        data = randn(Float32, D, 100)
        train_flow!(dist, data; n_epochs=1, batch_size=50)
        @test !isnothing(dist.normalizer)
    end
end
