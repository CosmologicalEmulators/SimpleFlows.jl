# ── Min-Max Normalizer ────────────────────────────────────────────────────────

"""
    MinMaxNormalizer(x_min, x_max)
    MinMaxNormalizer(data)

Per-dimension min-max scaler that maps each feature to [0, 1].

The change-of-variables term added to logpdf is:
    log |det J| = sum(-log(x_max - x_min))
which is a constant precomputed at construction time.
"""
struct MinMaxNormalizer{T<:Real, A}
    x_min   :: A
    x_max   :: A
    log_jac :: T          # precomputed = sum(-log(x_max - x_min))
end

"""
    MinMaxNormalizer(data::AbstractMatrix)

Fit a `MinMaxNormalizer` from training data (shape `n_dims × n_samples`).
"""
function MinMaxNormalizer(x::AbstractMatrix{T}) where {T}
    x_min = vec(minimum(x, dims=2))
    x_max = vec(maximum(x, dims=2))
    
    if any(x_min .≈ x_max)
        throw(ArgumentError("Data has zero variance along one or more dimensions. Cannot initialize MinMaxNormalizer."))
    end
    
    # Base volume change: sum(-log(x_max - x_min))
    logabsdet = sum(-log.(x_max .- x_min))
    return MinMaxNormalizer(x_min, x_max, logabsdet)
end

"""
    normalize(n::MinMaxNormalizer, x) -> z ∈ [0, 1]^d

Apply the forward transform: `z = (x - x_min) / (x_max - x_min)`.
"""
function normalize(n::MinMaxNormalizer, x::AbstractMatrix)
    return (x .- n.x_min) ./ (n.x_max .- n.x_min)
end

function normalize(n::MinMaxNormalizer, x::AbstractVector)
    return (x .- n.x_min) ./ (n.x_max .- n.x_min)
end

"""
    denormalize(n::MinMaxNormalizer, z) -> x

Apply the inverse transform: `x = z * (x_max - x_min) + x_min`.
"""
function denormalize(n::MinMaxNormalizer, z::AbstractMatrix)
    return z .* (n.x_max .- n.x_min) .+ n.x_min
end

function denormalize(n::MinMaxNormalizer, z::AbstractVector)
    return z .* (n.x_max .- n.x_min) .+ n.x_min
end

# ── Whitening Normalizer ─────────────────────────────────────────────────────

"""
    WhiteningNormalizer(mean, chol, log_jac)
    WhiteningNormalizer(data)

Affine whitening fitted from the sample mean `μ` and covariance `Σ = L Lᵀ`
(`L` lower-triangular Cholesky factor, stored as a dense matrix with a zero
upper triangle).

- forward: `z = L \\ (x - μ)` (triangular solve, no explicit inverse)
- inverse: `x = L z + μ`
- change-of-variables term added to logpdf: `-sum(log, diag(L))`, precomputed.

Fitting from `data::AbstractMatrix{<:Union{Float32,Float64}}` (shape
`n_dims × n_samples`; integer, `BigFloat`, dual-number or empty inputs are
rejected, nothing is coerced) accumulates the mean and covariance in `Float64`
and casts `μ`, `L` to the data's element type.

The singularity check is unit invariant. With `D = diag(sqrt.(diag(Σ)))` the
correlation matrix `R = D⁻¹ Σ D⁻¹` is factorized, `R = Lc Lcᵀ`, and the physical
factor is `L = D Lc` (row scaling), so coordinates differing by many orders of
magnitude are accepted. The fit throws `ArgumentError` for non-finite data, fewer
than `n_dims + 1` samples, a zero-variance coordinate, a failed Cholesky
factorization, or a correlation-matrix condition number above `1/eps(T)`
(numerically collinear coordinates), or if the cast factor is not finite with
positive diagonal in precision `T`. No jitter or dimension dropping is applied.
"""
struct WhiteningNormalizer{T<:Real, V, M}
    mean    :: V
    chol    :: M
    log_jac :: T          # precomputed = -sum(log, diag(L))
end

# Computed from the stored (precision-T) factor so that a save/load round trip,
# which recomputes it from the stored factor, reproduces it bit for bit.
_whitening_log_jac(L::AbstractMatrix{T}) where {T} = T(-sum(log, Float64.(diag(L))))

function WhiteningNormalizer(x::AbstractMatrix)
    T = eltype(x)
    T in (Float32, Float64) || throw(ArgumentError(
        "WhiteningNormalizer requires Float32 or Float64 data, got element type $T."))
    d, n = size(x)
    d >= 1 || throw(ArgumentError("Cannot fit a WhiteningNormalizer with zero dimensions."))
    all(isfinite, x) || throw(ArgumentError("Data contains non-finite values; cannot fit WhiteningNormalizer."))
    n >= d + 1 || throw(ArgumentError("Need at least n_dims + 1 = $(d + 1) samples to fit a WhiteningNormalizer, got $n."))

    x64 = Float64.(x)
    μ = vec(sum(x64; dims=2)) ./ n
    xc = x64 .- μ
    Σ = (xc * xc') ./ (n - 1)

    s = sqrt.(diag(Σ))
    all(s -> isfinite(s) && s > 0, s) ||
        throw(ArgumentError("Data has zero (or non-finite) variance along one or more dimensions; cannot fit WhiteningNormalizer."))
    R = Symmetric(Σ ./ (s * s'))
    R = Symmetric(R + I - Diagonal(diag(R)))   # exact unit diagonal

    F = cholesky(R; check=false)
    issuccess(F) || throw(ArgumentError("Sample covariance is not positive definite (collinear coordinates); cannot fit WhiteningNormalizer."))
    λ = eigvals(R)
    if !(first(λ) > 0) || last(λ) / first(λ) > 1 / eps(T)
        throw(ArgumentError("Sample correlation matrix is numerically singular (condition number $(last(λ) / first(λ))) for $T; cannot fit WhiteningNormalizer."))
    end

    L = T.(Diagonal(s) * Matrix(F.L))
    (all(isfinite, L) && all(>(0), diag(L))) ||
        throw(ArgumentError("Whitening factor is not representable in $T (scale range too large); cannot fit WhiteningNormalizer."))
    return WhiteningNormalizer(T.(μ), L, _whitening_log_jac(L))
end

# Solve `L z = y` for lower-triangular `L` without forming an inverse. A separate
# function so the Reactant extension can substitute a differentiable
# device implementation (see ext/ReactantExt.jl).
_lower_solve(L::AbstractMatrix, y::AbstractMatrix) = LowerTriangular(L) \ y

"""Whiten: `z = L \\ (x - μ)`."""
normalize(n::WhiteningNormalizer, x::AbstractMatrix) = _lower_solve(n.chol, x .- n.mean)

normalize(n::WhiteningNormalizer, x::AbstractVector) =
    vec(_lower_solve(n.chol, reshape(x .- n.mean, :, 1)))

"""Un-whiten: `x = L z + μ`."""
denormalize(n::WhiteningNormalizer, z::AbstractMatrix) = n.chol * z .+ n.mean

denormalize(n::WhiteningNormalizer, z::AbstractVector) = n.chol * z .+ n.mean

# Host evaluation uses the fitted constant. Device methods derive it from traced
# arrays so reusing a compiled flow with a different normalizer updates its density.
_normalizer_log_jac(n::MinMaxNormalizer) = n.log_jac
_normalizer_log_jac(n::WhiteningNormalizer) = n.log_jac
