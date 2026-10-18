using JET

@testset "JET static analysis" begin
    # Report errors originating in this package, not speculative dispatch into
    # every external array/AD backend loaded by the integration test suite.
    JET.test_package(SimpleFlows; target_modules=(SimpleFlows,))
    # Concrete numerical calls complement package-level analysis of generic
    # definitions. These test errors, not a blanket promise of type stability.
    data = randn(Random.Xoshiro(81), Float64, 2, 32)
    for normalizer in (MinMaxNormalizer(data), WhiteningNormalizer(data))
        JET.@test_call SimpleFlows.normalize(normalizer, data)
        JET.@test_call SimpleFlows.denormalize(normalizer, data)
        JET.@test_opt SimpleFlows.normalize(normalizer, data)
        JET.@test_opt SimpleFlows.denormalize(normalizer, data)
    end
    JET.@test_call SimpleFlows.compute_bin_idx([-1.0 0.0 1.0; -1.0 0.0 1.0], [0.2, -0.2], 2)
    for architecture in (:RealNVP, :NSF, :MAF)
        flow = FlowDistribution(Float64; architecture, n_transforms=2,
            dist_dims=2, hidden_layer_sizes=[8, 8], rng=Random.Xoshiro(82))
        JET.@test_call target_modules=(SimpleFlows,) logpdf(flow, data)
        JET.@test_call target_modules=(SimpleFlows,) logpdf(flow, data[:, 1])
        JET.@test_call target_modules=(SimpleFlows,) rand(Random.Xoshiro(83), flow, 3)
    end
end
