using SimpleFlows
using Test
using NPZ
using ForwardDiff
using Zygote
using LinearAlgebra

@testset "Rational Quadratic Splines" begin
    # 1. Load reference data (F64)
    data = npzread("/tmp/nsf_test_data_f64.npz")
    
    for T in [Float32, Float64]
        @testset "Precision: $T" begin
            inputs = T.(data["inputs"])
            unnormalized_widths = T.(data["unnormalized_widths"])
            unnormalized_heights = T.(data["unnormalized_heights"])
            unnormalized_derivatives = T.(data["unnormalized_derivatives"])
            
            ref_outputs = T.(data["outputs"])
            ref_logabsdet = T.(data["logabsdet"])
            
            D, N = size(inputs)
            
            @testset "Numerical Parity with PyTorch" begin
                for d in 1:D
                    out, lad = unconstrained_rational_quadratic_spline(
                        inputs[d, :],
                        unnormalized_widths[d, :, :],
                        unnormalized_heights[d, :, :],
                        unnormalized_derivatives[d, :, :],
                        T(3.0) # tail_bound
                    )
                    
                    @test out ≈ ref_outputs[d, :] atol=(T == Float32 ? 1e-4 : 1e-6)
                    @test lad ≈ ref_logabsdet[d, :] atol=(T == Float32 ? 1e-4 : 1e-6)
                end
            end
            
            @testset "Invertibility" begin
                for d in 1:D
                    out, _ = unconstrained_rational_quadratic_spline(
                        inputs[d, :],
                        unnormalized_widths[d, :, :],
                        unnormalized_heights[d, :, :],
                        unnormalized_derivatives[d, :, :],
                        T(3.0)
                    )
                    
                    back, _ = unconstrained_rational_quadratic_spline(
                        out,
                        unnormalized_widths[d, :, :],
                        unnormalized_heights[d, :, :],
                        unnormalized_derivatives[d, :, :],
                        T(3.0);
                        inverse=true
                    )
                    
                    @test back ≈ inputs[d, :] atol=(T == Float32 ? 1e-4 : 1e-5)
                end
            end
            
            @testset "ForwardDiff Compatibility" begin
                d = 1
                x = inputs[d, 1:5]
                w = unnormalized_widths[d, 1:5, :]
                h = unnormalized_heights[d, 1:5, :]
                dv = unnormalized_derivatives[d, 1:5, :]
                
                f(x_in) = sum(unconstrained_rational_quadratic_spline(x_in, w, h, dv, T(3.0))[1])
                
                g = ForwardDiff.gradient(f, x)
                @test all(isfinite, g)
                @test length(g) == 5
            end
            
            @testset "Zygote Compatibility" begin
                d = 1
                x = inputs[d, 1:5]
                w = unnormalized_widths[d, 1:5, :]
                h = unnormalized_heights[d, 1:5, :]
                dv = unnormalized_derivatives[d, 1:5, :]
                
                f(x_in) = sum(unconstrained_rational_quadratic_spline(x_in, w, h, dv, T(3.0))[1])
                
                gz = Zygote.gradient(f, x)[1]
                @test all(isfinite, gz)
                
                gf = ForwardDiff.gradient(f, x)
                @test gz ≈ gf atol=T == Float32 ? 1e-4 : 1e-10
            end
        end
    end
end
