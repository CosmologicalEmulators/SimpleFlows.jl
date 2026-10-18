using SimpleFlows
using Test
using JSON
using ForwardDiff
using Zygote
using LinearAlgebra
using Random

@testset "Rational Quadratic Splines" begin
    # 1. Load reference data
    data = JSON.parsefile(joinpath(@__DIR__, "data", "nsf_test_data.json"))
    
    # Helper to reconstruct array from JSON dict
    function get_array(T, dict, key)
        d = dict[key]
        reshape(T.(d["data"]), d["shape"]...)
    end
    
    for T in [Float32, Float64]
        @testset "Precision: $T" begin
            inputs = get_array(T, data, "inputs")
            unnormalized_widths = get_array(T, data, "unnormalized_widths")
            unnormalized_heights = get_array(T, data, "unnormalized_heights")
            unnormalized_derivatives = get_array(T, data, "unnormalized_derivatives")
            
            ref_outputs = get_array(T, data, "outputs")
            ref_logabsdet = get_array(T, data, "logabsdet")
            
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
            
            @testset "Boundary & Tail Consistency" begin
                # x explicitly outside the tail_bound
                x_out = T.([-4.0, 4.0, -10.0, 10.0])
                N = length(x_out)
                K = 5
                w = randn(T, N, K)
                h = randn(T, N, K)
                dv = randn(T, N, K-1)
                
                # Forward should be strict linear identity outside bounds
                y_out, lad_out = unconstrained_rational_quadratic_spline(x_out, w, h, dv, T(3.0))
                
                @test y_out ≈ x_out atol=1e-6
                @test lad_out ≈ zeros(T, N) atol=1e-6
                
                # Inverse should also be strict identity
                x_rec, lad_inv = unconstrained_rational_quadratic_spline(x_out, w, h, dv, T(3.0); inverse=true)
                
                @test x_rec ≈ x_out atol=1e-6
                @test lad_inv ≈ zeros(T, N) atol=1e-6
            end
        end

    end
end

# ── Narrow-bin / near-edge numerics against a BigFloat reference ─────────────
# Independent reference: the original rational-quadratic equations evaluated in 256-bit
# arithmetic on the same (float) knots; the inverse by bisection of the forward map, not by
# the quadratic formula used in the implementation.

using NNlib
import SimpleFlows

function _rq_knots(::Type{T}, w, h, d, tb) where {T}
    M, K = size(w)
    mw = mh = md = T(1e-3)
    constant = T(log(exp(1 - md) - 1))
    ud = hcat(fill(constant, M), T.(d), fill(constant, M))
    widths = mw .+ (1 - mw * K) .* NNlib.softmax(T.(w); dims=2)
    cw = cumsum(widths; dims=2)
    cw = hcat(fill(-T(tb), M), (2 * T(tb)) .* cw[:, 1:end-1] .- T(tb), fill(T(tb), M))
    heights = mh .+ (1 - mh * K) .* NNlib.softmax(T.(h); dims=2)
    ch = cumsum(heights; dims=2)
    ch = hcat(fill(-T(tb), M), (2 * T(tb)) .* ch[:, 1:end-1] .- T(tb), fill(T(tb), M))
    return cw, ch, cw[:, 2:end] .- cw[:, 1:end-1], ch[:, 2:end] .- ch[:, 1:end-1], md .+ NNlib.softplus.(ud)
end

function _rq_ref_eval(xk, wk, yk, hk, d0, d1, x)
    δ = hk / wk; θ = (x - xk) / wk; t = θ * (1 - θ)
    den = δ + (d0 + d1 - 2δ) * t
    return yk + hk * (δ * θ^2 + d0 * t) / den, log(δ^2 * (d1 * θ^2 + 2δ * t + d0 * (1 - θ)^2) / den^2)
end

function _rq_ref(kn, i, v; inverse::Bool, tb)
    setprecision(BigFloat, 256) do
        cw, ch, wd, ht, der = kn
        abs(v) > tb && return BigFloat(v), BigFloat(0)
        edges = BigFloat.(inverse ? ch[i, :] : cw[i, :])
        k = clamp(searchsortedlast(edges, BigFloat(v)), 1, size(wd, 2))
        xk, wk, yk, hk, d0 = (BigFloat(A[i, k]) for A in (cw, wd, ch, ht, der))
        d1 = BigFloat(der[i, k + 1])
        if !inverse
            return _rq_ref_eval(xk, wk, yk, hk, d0, d1, BigFloat(v))
        end
        lo, hi = BigFloat(0), BigFloat(1)
        for _ in 1:300
            mid = (lo + hi) / 2
            _rq_ref_eval(xk, wk, yk, hk, d0, d1, xk + mid * wk)[1] < v ? (lo = mid) : (hi = mid)
        end
        x = xk + (lo + hi) / 2 * wk
        return x, -_rq_ref_eval(xk, wk, yk, hk, d0, d1, x)[2]
    end
end

# Largest x error attributable to rounding the input y by one ulp (conditioning floor of the inverse).
_ulp_floor(::Type{T}, y, ld_fwd) where {T} = Float64(eps(T)) * max(abs(Float64(y)), 1.0) / exp(Float64(ld_fwd))

# Inverse error must stay within an order of magnitude of the rounding-conditioning floor
# (1 ulp of y mapped back through dy/dx) plus a few ulp of the output itself. The previous
# implementation exceeded this floor by up to 1e8 on the same inputs.
_within_floor(r, ::Type{T}) where {T} = all(r.inv_x .<= 10 .* r.floor_x .+ 4 * eps(T) .* max.(abs.(Float64.(r.xr)), 1.0))

function _spline_vs_reference(T, w, h, d, x; tb=3.0)
    kn = _rq_knots(T, w, h, d, tb)
    y, lf = unconstrained_rational_quadratic_spline(x, w, h, d, T(tb))
    xr, li = unconstrained_rational_quadratic_spline(y, w, h, d, T(tb); inverse=true)
    M = length(x)
    fwd = [_rq_ref(kn, i, x[i]; inverse=false, tb) for i in 1:M]
    inv = [_rq_ref(kn, i, y[i]; inverse=true, tb) for i in 1:M]
    # Jacobian expression evaluated at the returned inverse point must be the exact reciprocal.
    at_xr = [_rq_ref(kn, i, xr[i]; inverse=false, tb) for i in 1:M]
    return (; y, lf, xr, li,
            fwd_y=[abs(Float64(y[i] - fwd[i][1])) for i in 1:M],
            slope=[exp(Float64(fwd[i][2])) for i in 1:M],
            fwd_ld=maximum(abs(Float64(lf[i] - fwd[i][2])) for i in 1:M),
            inv_x=[abs(Float64(xr[i] - inv[i][1])) for i in 1:M],
            floor_x=[_ulp_floor(T, y[i], fwd[i][2]) for i in 1:M],
            inv_ld_consistency=maximum(abs(Float64(li[i] + at_xr[i][2])) for i in 1:M),
            inv_ld=maximum(abs(Float64(li[i] - inv[i][2])) for i in 1:M),
            roundtrip_ld=maximum(abs.(Float64.(lf .+ li))))
end

@testset "Spline narrow bins and edges vs BigFloat" begin
    k = 8
    @testset "parent reproduction (valid parameters, narrow bins)" begin
        t = vcat(10.0 .^ range(-8, -1, length=80), 1 .- 10.0 .^ range(-8, -1, length=80))
        x = -3 .+ 0.006 .+ 0.006 .* t
        w = fill(-50.0, length(x), k); w[:, k] .= 50.0
        r = _spline_vs_reference(Float64, w, copy(w), fill(-50.0, length(x), k - 1), x)
        @test r.fwd_ld < 1e-12
        @test maximum(r.inv_x) < 1e-13                    # was 4.1e-8
        @test r.inv_ld < 1e-8                             # was 1.4e-2
        @test r.inv_ld_consistency < 1e-9
        @test r.roundtrip_ld < 1e-6                       # was 1.37e-2
    end

    rng = Random.MersenneTwister(1)
    for T in (Float64, Float32), scale in (3.0, 10.0, 50.0)
        @testset "random parameters scale $scale $T" begin
            M = 150
            w, h, d = (T.(scale .* randn(rng, M, k)), T.(scale .* randn(rng, M, k)), T.(scale .* randn(rng, M, k - 1)))
            x = T.(6 .* rand(rng, M) .- 3)
            r = _spline_vs_reference(T, w, h, d, x)
            @test r.fwd_ld < (T == Float64 ? 1e-13 : 1e-4)   # Float32: limited by rounding of theta in narrow bins
            @test _within_floor(r, T)     # below the 1-ulp(y) conditioning floor
            @test r.inv_ld_consistency < (T == Float64 ? 1e-8 : 5e-4)
        end
    end

    @testset "points next to bin edges $T" for T in (Float64, Float32)
        M = 140
        w, h, d = (T.(8 .* randn(rng, M, k)), T.(8 .* randn(rng, M, k)), T.(8 .* randn(rng, M, k - 1)))
        cw = _rq_knots(T, w, h, d, 3.0)[1]
        x = [T(clamp(cw[i, 2 + (i % (k - 1))] + (isodd(i) ? 1 : -1) * 1e-7, -3, 3)) for i in 1:M]
        r = _spline_vs_reference(T, w, h, d, x)
        @test all(r.fwd_y .<= 10 * eps(T) * 3 .* (1 .+ r.slope))   # rounding of x mapped through dy/dx
        @test _within_floor(r, T)
        @test all(isfinite, r.lf) && all(isfinite, r.li)
    end

    @testset "linear limit, steep derivatives, tails $T" for T in (Float64, Float32)
        M = 100
        x = T.(6 .* rand(rng, M) .- 3)
        dlin = T(log(exp(1 - 1e-3) - 1))      # derivative 1 at every knot, equal widths/heights: the identity
        y, lf = unconstrained_rational_quadratic_spline(x, zeros(T, M, k), zeros(T, M, k), fill(dlin, M, k - 1), T(3))
        @test y ≈ x atol=(T == Float64 ? 1e-14 : 2e-6)
        @test all(abs.(lf) .< (T == Float64 ? 1e-14 : 2e-6))
        r = _spline_vs_reference(T, zeros(T, M, k), zeros(T, M, k), fill(T(30), M, k - 1), x)
        @test r.fwd_ld < (T == Float64 ? 1e-13 : 1e-5)
        @test _within_floor(r, T)
        # identity tails and exact bounds
        xt = T[-5, -3.0001, -3, 3, 3.0001, 7]
        wt = randn(rng, T, length(xt), k); ht = randn(rng, T, length(xt), k); dt = randn(rng, T, length(xt), k - 1)
        yt, lt = unconstrained_rational_quadratic_spline(xt, wt, ht, dt, T(3))
        @test yt[[1, 2, 5, 6]] == xt[[1, 2, 5, 6]] && all(lt[[1, 2, 5, 6]] .== 0)
        @test yt[3] ≈ -3 atol=2eps(T) * 3
        @test yt[4] ≈ 3 atol=2eps(T) * 3
        xb, _ = unconstrained_rational_quadratic_spline(yt, wt, ht, dt, T(3); inverse=true)
        @test xb ≈ xt atol=(T == Float64 ? 1e-12 : 1e-5)
    end
end

@testset "Spline parameter and input gradients (narrow bins, edges)" begin
    k = 6
    rng = Random.MersenneTwister(5)
    M = 12
    w0, h0, d0 = (4 .* randn(rng, M, k), 4 .* randn(rng, M, k), 4 .* randn(rng, M, k - 1))
    kn = _rq_knots(Float64, w0, h0, d0, 3.0)
    # include a point 1e-9 beside a knot (a point exactly on an interior knot is a kink: the one-sided
    # derivatives differ and AD backends legitimately pick different sides), the tail bounds and outside
    x = vcat([kn[1][1, 3] + 1e-9, -3.0, 3.0, 3.5, -4.0], 6 .* rand(rng, M - 5) .- 3)
    nw, nh, nd = M * k, M * k, M * (k - 1)
    unpack(p) = (reshape(p[1:nw], M, k), reshape(p[nw+1:nw+nh], M, k), reshape(p[nw+nh+1:end], M, k - 1))
    p0 = vcat(vec(w0), vec(h0), vec(d0))

    for inverse in (false, true)
        src = inverse ? first(unconstrained_rational_quadratic_spline(x, w0, h0, d0, 3.0)) : x
        loss(p) = begin
            w, h, d = unpack(p)
            o, l = unconstrained_rational_quadratic_spline(src, w, h, d, 3.0; inverse)
            sum(o .* cos.(1:M)) + sum(l .* sin.(1:M))
        end
        gz = Zygote.gradient(loss, p0)[1]
        gf = ForwardDiff.gradient(loss, p0)
        @test all(isfinite, gz) && all(isfinite, gf)
        @test gz ≈ gf rtol=1e-8 atol=1e-10
    end

    # Inverse parameter gradient = implicit-function derivative of the forward map:
    # d x / d p |_(y fixed) = -(dF/dp) / (dF/dx), checked against the AD gradient of the inverse.
    xs = 6 .* rand(rng, M) .- 3
    y0 = first(unconstrained_rational_quadratic_spline(xs, w0, h0, d0, 3.0))
    inv_out(p) = first(unconstrained_rational_quadratic_spline(y0, unpack(p)..., 3.0; inverse=true))
    Jinv = ForwardDiff.jacobian(inv_out, p0)
    fwd_out(p, xv) = first(unconstrained_rational_quadratic_spline(xv, unpack(p)..., 3.0))
    dFdp = ForwardDiff.jacobian(p -> fwd_out(p, xs), p0)
    dFdx = ForwardDiff.gradient(xv -> sum(fwd_out(p0, xv)), xs)     # rows are independent, so this is the diagonal
    @test Jinv ≈ -(dFdp ./ dFdx) rtol=1e-6 atol=1e-9
    Jz = Zygote.jacobian(inv_out, p0)[1]
    @test Jz ≈ Jinv rtol=1e-8 atol=1e-10

    # input gradient: d y / d x = exp(logdet)
    dydx = ForwardDiff.gradient(xv -> sum(first(unconstrained_rational_quadratic_spline(xv, w0, h0, d0, 3.0))), x)
    _, lf = unconstrained_rational_quadratic_spline(x, w0, h0, d0, 3.0)
    @test dydx ≈ exp.(lf) rtol=1e-9
end
