using Test
using SimpleFlows
using Distributions
using Random
using DifferentiationInterface
using ForwardDiff
using Zygote
using Mooncake

println("Starting AD tests...")

@testset "Automatic Differentiation" begin
    rng = Random.MersenneTwister(42)
    n_dims = 2
    architectures = [:RealNVP, :NSF, :MAF]

    for arch in architectures
        println("Testing architecture: ", arch)
        @testset "$arch" begin
            d = FlowDistribution(Float64; architecture=arch, n_transforms=2, dist_dims=n_dims, hidden_layer_sizes=[16, 16], n_layers=2)
            x = rand(rng, n_dims)
            
            f(x) = logpdf(d, x)
            val = f(x)
            @test isfinite(val)
            
            for (name, backend) in [
                ("ForwardDiff", AutoForwardDiff()),
                ("Zygote", AutoZygote()),
                ("Mooncake", AutoMooncake(config=nothing))
            ]
                println("  Testing backend: ", name)
                @testset "$name" begin
                    g = try
                        DifferentiationInterface.gradient(f, backend, x)
                    catch e
                        @error "Failed to differentiate $arch with $name" exception=(e, catch_backtrace())
                        nothing
                    end
                    @test g !== nothing
                    @test length(g) == n_dims
                    @test all(isfinite, g)
                end
            end
        end
    end
end
