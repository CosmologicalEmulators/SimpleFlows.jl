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
    conditioner = x_cond -> zeros(Float32, 2n, B)
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
