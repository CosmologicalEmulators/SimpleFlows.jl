using Reactant, Enzyme

_dynamic_normalizer_logpdf(x, flow) = logpdf(flow, x)
_dynamic_normalizer_loss(x, flow) = sum(logpdf(flow, x))
_dynamic_normalizer_score(x, flow) =
    Enzyme.gradient(Reverse, _dynamic_normalizer_loss, x, Const(flow))[1]

@testset "Compiled flow reuse updates normalizer density constants" begin
    Reactant.set_default_backend("cpu")
    for T in (Float32, Float64), arch in (:RealNVP, :NSF, :MAF), normalizer in (MinMaxNormalizer, WhiteningNormalizer)
        data = T[1 0.3; 0.2 2] * T[-1 -1 1 1; -1 1 -1 1]
        x = T[-0.2 0.4; -0.5 0.7]
        first_flow = FlowDistribution(T; architecture=arch, n_transforms=2, dist_dims=2,
                                     hidden_layer_sizes=[4], K=4, parameter_names=["x", "y"], rng=Xoshiro(83))
        first_flow.normalizer = normalizer(data)
        second_flow = deepcopy(first_flow)
        shift = T[0.4, -0.3]
        second_flow.normalizer = normalizer(2 .* data .+ shift)
        second_x = 2 .* x .+ shift
        first_device, second_device = to_reactant(first_flow), to_reactant(second_flow)
        xr, yr = Reactant.to_rarray(x), Reactant.to_rarray(second_x)
        evaluate = Reactant.@compile sync=true _dynamic_normalizer_logpdf(xr, first_device)
        score = Reactant.@compile sync=true _dynamic_normalizer_score(xr, first_device)
        tol = T == Float32 ? 5e-5 : 1e-10
        before = Array(evaluate(xr, first_device))
        @test before ≈ logpdf(first_flow, x) rtol=tol atol=tol
        after = Array(evaluate(yr, second_device))
        @test after ≈ logpdf(second_flow, second_x) rtol=tol atol=tol
        # Analytic change of variables: x -> 2x + shift in two dimensions.
        @test after ≈ before .- T(2log(2)) rtol=tol atol=tol
        @test Array(score(yr, second_device)) ≈ only(Zygote.gradient(a -> sum(logpdf(second_flow, a)), second_x)) rtol=tol atol=tol
        @test Array(evaluate(xr, first_device)) ≈ before rtol=tol atol=tol
        @test Array(xr) == x
        @test Array(yr) == second_x
        @test second_device.parameter_names == ["x", "y"]
        n = second_device.normalizer
        if normalizer === WhiteningNormalizer
            @test Array(n.mean) == second_flow.normalizer.mean
            @test Array(n.chol) == second_flow.normalizer.chol
        else
            @test Array(n.x_min) == second_flow.normalizer.x_min
            @test Array(n.x_max) == second_flow.normalizer.x_max
        end
    end
end
