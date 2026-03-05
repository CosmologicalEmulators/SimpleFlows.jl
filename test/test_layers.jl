@testset "AffineBijector" begin
    rng = Random.MersenneTwister(1)
    n, B = 4, 8

    # Build params: first half = shift, second half = log_scale
    params  = randn(rng, Float32, 2n, B)
    b       = SimpleFlows.AffineBijector(params)

    x = randn(rng, Float32, n, B)

    # Forward
    y, ld_fwd = SimpleFlows.forward_and_log_det(b, x)
    @test size(y)  == (n, B)
    @test size(ld_fwd) == (n, B)
    @test all(isfinite, y)

    # Inverse recovers x
    x_rec, ld_inv = SimpleFlows.inverse_and_log_det(b, y)
    @test x_rec ≈ x  atol=1f-5

    # Forward and inverse log-dets cancel
    @test ld_fwd .+ ld_inv ≈ zeros(Float32, n, B)  atol=1f-6
end

@testset "MaskedCoupling" begin
    rng  = Random.MersenneTwister(2)
    n, B = 6, 10
    mask = Bool.(collect(1:n) .% 2 .== 0)   # [false, true, false, true, false, true]

    # Trivial conditioner: returns zeros (identity bijector)
    conditioner = x_cond -> zeros(Float32, 2*sum(mask), B)
    bj = SimpleFlows.MaskedCoupling(mask, conditioner, SimpleFlows.AffineBijector)

    x = randn(rng, Float32, n, B)
    y, ld = SimpleFlows.forward_and_log_det(bj, x)

    # Unmasked dims must be unchanged
    @test y[.!mask, :] ≈ x[.!mask, :]   atol=1f-6
    # With zero params the bijector is identity, log-det = 0
    @test all(ld .≈ 0f0)

    # Inverse recovers x
    x_rec, _ = SimpleFlows.inverse_and_log_det(bj, y)
    @test x_rec ≈ x  atol=1f-5
end

@testset "Explicit Dense Jacobian Inverses" begin
    rng = Random.MersenneTwister(3)
    n = 4
    
    # 1. AffineBijector
    params = randn(rng, Float32, 2n, 1)
    b_aff = SimpleFlows.AffineBijector(params)
    
    x_single = randn(rng, Float32, n)
    f_aff(x) = SimpleFlows.forward_and_log_det(b_aff, reshape(x, n, 1))[1][:]
    finv_aff(y) = SimpleFlows.inverse_and_log_det(b_aff, reshape(y, n, 1))[1][:]
    
    y_aff = f_aff(x_single)
    J_aff = ForwardDiff.jacobian(f_aff, x_single)
    Jinv_aff = ForwardDiff.jacobian(finv_aff, y_aff)
    
    # Check J * J_inv ≈ I
    @test J_aff * Jinv_aff ≈ I(n) atol=1f-4
    # Check determinant matches (Affine returns per-dimension ld)
    ld_fwd = SimpleFlows.forward_and_log_det(b_aff, reshape(x_single, n, 1))[2][:]
    @test log(abs(det(J_aff))) ≈ sum(ld_fwd) atol=1f-4
    
    # 2. MaskedCoupling
    mask = [false, true, false, true]
    # Trivial deterministic conditioner for testing structural zeros. MUST be deterministic for ForwardDiff!
    fixed_params = randn(rng, Float32, 2*sum(mask), 1)
    conditioner = x_cond -> fixed_params .+ sum(x_cond) * 0.0f0 # Ensure dual numbers propagate if needed
    bj_m = SimpleFlows.MaskedCoupling(mask, conditioner, SimpleFlows.AffineBijector)
    
    f_m(x) = SimpleFlows.forward_and_log_det(bj_m, reshape(x, n, 1))[1][:]
    finv_m(y) = SimpleFlows.inverse_and_log_det(bj_m, reshape(y, n, 1))[1][:]
    
    x_m = randn(rng, Float32, n)
    y_m = f_m(x_m)
    
    J_m = ForwardDiff.jacobian(f_m, x_m)
    Jinv_m = ForwardDiff.jacobian(finv_m, y_m)
    
    @test J_m * Jinv_m ≈ I(n) atol=1f-4
end

