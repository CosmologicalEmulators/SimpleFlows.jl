# ── Min-Max Normalizer ────────────────────────────────────────────────────────

"""
    MinMaxNormalizer(x_min, x_max)
    MinMaxNormalizer(data)

Per-dimension min-max scaler that maps each feature to [0, 1].

The change-of-variables term added to logpdf is:
    log |det J| = sum(-log(x_max - x_min))
which is a constant precomputed at construction time.
"""
struct MinMaxNormalizer{T<:Real}
    x_min   :: Vector{T}
    x_max   :: Vector{T}
    log_jac :: T          # precomputed = sum(-log(x_max - x_min))
end

"""
    MinMaxNormalizer(data::AbstractMatrix)

Fit a `MinMaxNormalizer` from training data (shape `n_dims × n_samples`).
"""
function MinMaxNormalizer(data::AbstractMatrix{T}) where T
    xmin = vec(minimum(data; dims=2))
    xmax = vec(maximum(data; dims=2))
    log_jac = sum(-log.(xmax .- xmin))
    return MinMaxNormalizer{T}(xmin, xmax, log_jac)
end

"""
    normalize(n::MinMaxNormalizer, x) -> z ∈ [0, 1]^d

Apply the forward transform: `z = (x - x_min) / (x_max - x_min)`.
"""
normalize(n::MinMaxNormalizer, x::AbstractMatrix) =
    (x .- n.x_min) ./ (n.x_max .- n.x_min)

normalize(n::MinMaxNormalizer, x::AbstractVector) =
    (x .- n.x_min) ./ (n.x_max .- n.x_min)

"""
    denormalize(n::MinMaxNormalizer, z) -> x

Apply the inverse transform: `x = z * (x_max - x_min) + x_min`.
"""
denormalize(n::MinMaxNormalizer, z::AbstractMatrix) =
    z .* (n.x_max .- n.x_min) .+ n.x_min

denormalize(n::MinMaxNormalizer, z::AbstractVector) =
    z .* (n.x_max .- n.x_min) .+ n.x_min
