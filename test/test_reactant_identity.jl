using Reactant, Enzyme

function _identity_spline_loss(p, x, inverse)
    y, ld = unconstrained_rational_quadratic_spline(x, p[:, 1:4], p[:, 5:8], p[:, 9:11], 3.0; inverse)
    return sum(y) + sum(ld)
end
_identity_spline_grad(p, x, inverse) =
    Enzyme.gradient(Reverse, _identity_spline_loss, p, Const(x), Const(inverse))[1]

@testset "Reactant spline identity and linear-limit adjoints" begin
    Reactant.set_default_backend("cpu")
    for T in (Float32, Float64), inverse in (false, true)
        # Off-knot points have unique parameter derivatives for the log-Jacobian.
        x = T[-2.8, -1.499, 0.001, 0.2, 1.501, 2.8]
        p = zeros(T, 6, 11)
        p[:, 9:11] .= SimpleFlows._nsf_identity_derivative_logit(T)
        pr, xr = Reactant.to_rarray(p), Reactant.to_rarray(x)
        compiled = Reactant.@compile sync=true _identity_spline_grad(pr, xr, inverse)
        # Same compiled function, exact identity then nonidentity weights.
        for shift in (zero(T), T(1e-5))
            q = copy(p)
            q[:, 9:11] .+= shift
            got = Array(compiled(Reactant.to_rarray(q), xr, inverse))
            loss(a) = _identity_spline_loss(a, x, inverse)
            zg = only(Zygote.gradient(loss, q))
            fd = ForwardDiff.gradient(loss, q)
            @test all(isfinite, got)
            tol = T == Float64 ? 1e-9 : 2e-5
            @test got ≈ zg rtol=tol atol=tol
            @test got ≈ fd rtol=tol atol=tol
            # Retain exact-knot coverage. After perturbation the log-Jacobian is
            # only piecewise differentiable there; ForwardDiff's Dual ordering
            # in bin search can choose a different one-sided derivative.
            knots = T[-2.8, -1.5, 0, 0.2, 1.5, 2.8]
            gk = Array(compiled(Reactant.to_rarray(q), Reactant.to_rarray(knots), inverse))
            zk = only(Zygote.gradient(a -> _identity_spline_loss(a, knots, inverse), q))
            @test all(isfinite, gk)
            @test gk ≈ zk rtol=tol atol=tol
        end
    end
end
