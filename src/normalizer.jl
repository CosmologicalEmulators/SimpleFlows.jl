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
function MinMaxNormalizer(x::AbstractMatrix{T}) where {T}
    x_min = vec(minimum(x, dims=2))
    x_max = vec(maximum(x, dims=2))
    
    if any(x_min .≈ x_max)
        throw(ArgumentError("Data has zero variance along one or more dimensions. Cannot initialize MinMaxNormalizer."))
    end
    
    # Base volume change: sum(-log(x_max - x_min))
    logabsdet = sum(-log.(x_max .- x_min))
    return MinMaxNormalizer{T}(x_min, x_max, logabsdet)
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
