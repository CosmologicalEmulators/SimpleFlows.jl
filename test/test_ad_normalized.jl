using DifferentiationInterface, ForwardDiff, Zygote, Mooncake

function check_normalized_input_gradients(T, input_type, architecture, normalization, identity_init)
    data = randn(Random.Xoshiro(301), T, 2, 48)
    flow = FlowDistribution(T; architecture, identity_init, n_transforms=2,
        dist_dims=2, hidden_layer_sizes=[8, 8], rng=Random.Xoshiro(302))
    train_flow!(flow, data; normalization, n_epochs=identity_init ? 2 : 0,
        verbose=false, rng=Random.Xoshiro(305))
    f = x -> logpdf(flow, x)
    points = [randn(Random.Xoshiro(seed), input_type, 2) for seed in (303, 304)]
    fd = AutoForwardDiff()
    fd_prep = DifferentiationInterface.prepare_gradient(f, fd, first(points))
    references = [DifferentiationInterface.gradient(f, fd_prep, fd, x) for x in points]
    @test Bijectors.bijector(flow) === identity
    for (backend_name, backend) in (("Zygote", AutoZygote()),
                                    ("Mooncake", AutoMooncake(config=nothing)))
        @testset "$backend_name" begin
            prep = DifferentiationInterface.prepare_gradient(f, backend, first(points))
            for (x, reference) in zip(points, references)
                @test all(isfinite, reference)
                actual = DifferentiationInterface.gradient(f, prep, backend, x)
                @test all(isfinite, actual)
                tol = T === Float32 ? 3e-4 : 2e-8
                @test isapprox(actual, reference; rtol=tol, atol=tol)
                @test Bijectors.logpdf_with_trans(flow, x, true) ≈ logpdf(flow, x)
            end
        end
    end
end

@testset "Normalized input gradients and Bijectors interface" begin
    for T in (Float32, Float64), architecture in (:RealNVP, :NSF, :MAF),
        normalization in (:minmax, :whiten), identity_init in (false, true),
        input_type in (T === Float32 ? (Float32, Float64) : (Float64,))
        @testset "$T/$input_type/$architecture/$normalization/identity=$identity_init" begin
            check_normalized_input_gradients(T, input_type, architecture, normalization, identity_init)
        end
    end
end
