@testset "Explicit precision and inferable positional constructors" begin
    for T in (Float32, Float64), architecture in (:RealNVP, :NSF, :MAF)
        flow = FlowDistribution(T; architecture, n_transforms=2, dist_dims=2,
            hidden_layer_sizes=[8], rng=Random.Xoshiro(601))
        data = randn(Random.Xoshiro(602), T, 2, 32)
        explicit = FlowDistribution{T, typeof(flow.model)}(flow.model, deepcopy(flow.ps),
            flow.st, flow.n_dims, flow.hidden_layer_sizes, nothing, nothing)
        @test logpdf(explicit, data) == logpdf(flow, data)
        # No floating-point precision can be inferred from `nothing`.
        @test_throws MethodError FlowDistribution(flow.model, flow.ps, flow.st,
            flow.n_dims, flow.hidden_layer_sizes, nothing, nothing)
        for normalization in (:minmax, :whiten)
            fitted = deepcopy(flow)
            train_flow!(fitted, data; normalization, n_epochs=0, verbose=false)
            inferred = FlowDistribution(fitted.model, fitted.ps, fitted.st,
                fitted.n_dims, fitted.hidden_layer_sizes, fitted.normalizer, nothing)
            @test typeof(inferred) === typeof(fitted)
            @test logpdf(inferred, data) == logpdf(fitted, data)
        end
    end
end
