using Reactant, Enzyme

function _tail_output_sum(x, p, inverse)
    y, _ = unconstrained_rational_quadratic_spline(x, p[:, 1:4], p[:, 5:8], p[:, 9:11], 3.0; inverse)
    return sum(y)
end
_tail_output_gradient(x, p, inverse) =
    Enzyme.gradient(Reverse, _tail_output_sum, x, Const(p), Const(inverse))[1]
_tail_score_loss(x, flow) = sum(logpdf(flow, x))
_tail_score_gradient(x, flow) =
    Enzyme.gradient(Reverse, _tail_score_loss, x, Const(flow))[1]

@testset "NSF exact tail endpoints retain the identity output slope" begin
    Reactant.set_default_backend("cpu")
    for T in (Float32, Float64), inverse in (false, true)
        x = T[-3, 3, -3, 3]
        p = randn(Xoshiro(81), T, 4, 11)
        y, ld = unconstrained_rational_quadratic_spline(x, p[:, 1:4], p[:, 5:8], p[:, 9:11], T(3); inverse)
        @test y == x
        @test all(iszero, ld)
        loss(v) = _tail_output_sum(v, p, inverse)
        @test only(Zygote.gradient(loss, x)) ≈ ones(T, 4)
        @test ForwardDiff.gradient(loss, x) ≈ ones(T, 4)
        xr, pr = Reactant.to_rarray(x), Reactant.to_rarray(p)
        compiled = Reactant.@compile sync=true _tail_output_gradient(xr, pr, inverse)
        @test Array(compiled(xr, pr, inverse)) ≈ ones(T, 4)
        # Changed parameters and reversed endpoints through the same compiled function.
        @test Array(compiled(Reactant.to_rarray(reverse(x)), Reactant.to_rarray(2 .* p), inverse)) ≈ ones(T, 4)
    end
end

@testset "Identity NSF public density scores at tail endpoints and neighbors" begin
    for T in (Float32, Float64), bound in (1, 3)
        b = T(bound)
        flow = FlowDistribution(T; architecture=:NSF, n_transforms=3, dist_dims=2,
                                hidden_layer_sizes=[4], K=4, tail_bound=b, rng=Xoshiro(82))
        points = T[-b, b, prevfloat(-b), nextfloat(-b), prevfloat(b), nextfloat(b), b-T(1e-4), b+T(1e-4)]
        x = vcat(reshape(points, 1, :), fill(T(0.2), 1, length(points)))
        loss(v) = sum(logpdf(flow, v))
        tol = T == Float32 ? 3e-5 : 1e-10
        @test only(Zygote.gradient(loss, x)) ≈ -x rtol=tol atol=tol
        @test ForwardDiff.gradient(loss, x) ≈ -x rtol=tol atol=tol
        for point in (-b, b)
            v = T[point, 0]
            @test only(Zygote.gradient(a -> logpdf(flow, a), v)) ≈ -v rtol=tol atol=tol
            @test ForwardDiff.gradient(a -> logpdf(flow, a), v) ≈ -v rtol=tol atol=tol
        end
        xr, fr = Reactant.to_rarray(x), to_reactant(flow)
        compiled = Reactant.@compile sync=true _tail_score_gradient(xr, fr)
        @test Array(compiled(xr, fr)) ≈ -x rtol=tol atol=tol
        changed = reverse(x; dims=2)
        @test Array(compiled(Reactant.to_rarray(changed), fr)) ≈ -changed rtol=tol atol=tol
    end
end
