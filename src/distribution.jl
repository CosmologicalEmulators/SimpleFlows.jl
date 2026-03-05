"""
    FlowDistribution(model, ps, st, n_dims, hidden_dims, n_layers, normalizer)

A trained normalizing flow wrapped as a `Distributions.jl`
`ContinuousMultivariateDistribution`, usable directly in `Turing.jl` models.

# Fields
- `model`: the `RealNVP` Lux model
- `ps`: trained parameters (NamedTuple)
- `st`: Lux state (NamedTuple)
- `n_dims`, `hidden_dims`, `n_layers`: architecture metadata for serialization
- `normalizer`: fitted `MinMaxNormalizer` (always present after training)
"""
mutable struct FlowDistribution{T<:Real, M<:AbstractLuxLayer} <: ContinuousMultivariateDistribution
    model              :: M
    ps
    st
    n_dims             :: Int
    hidden_layer_sizes :: Vector{Int}
    normalizer         :: Union{Nothing, MinMaxNormalizer{T}}
end

"""
    FlowDistribution([Type=Float32]; architecture=:RealNVP, n_transforms, dist_dims, 
                       hidden_layer_sizes, hidden_dims=64, n_layers=3,
                       activation=gelu, rng=Random.default_rng(), K=8, tail_bound=3.0)

Construct and randomly initialise a `FlowDistribution`.
`architecture` can be `:RealNVP` or `:NSF`.
"""
function FlowDistribution(::Type{T}=Float32;
                            architecture=:RealNVP,
                            n_transforms::Int, dist_dims::Int,
                            hidden_layer_sizes::Vector{Int}=Int[],
                            hidden_dims::Int=64, n_layers::Int=3,
                            activation=gelu,
                            K=8, tail_bound=3.0,
                            rng::AbstractRNG=Random.default_rng()) where {T<:Real}
    # If no vector given, fall back to the scalar convenience args
    if isempty(hidden_layer_sizes)
        hidden_layer_sizes = fill(hidden_dims, n_layers)
    end
    
    model = if architecture == :RealNVP
        RealNVP(; n_transforms, dist_dims, hidden_layer_sizes, activation)
    elseif architecture == :NSF
        NeuralSplineFlow(; n_transforms, dist_dims, hidden_layer_sizes, K, tail_bound, activation)
    elseif architecture == :MAF
        MaskedAutoregressiveFlow(; n_transforms, dist_dims, hidden_layer_sizes, activation)
    else
        error("Unknown architecture: $architecture. Supported: :RealNVP, :NSF, :MAF")
    end
    
    ps, st = Lux.setup(rng, model)
    ps = Lux.fmap(x -> x isa AbstractArray ? T.(x) : x, ps)
    return FlowDistribution{T, typeof(model)}(model, ps, st, dist_dims, hidden_layer_sizes, nothing)
end

# ── Distributions.jl interface ────────────────────────────────────────────────

Distributions.length(d::FlowDistribution) = d.n_dims

function _apply_normalizer(d::FlowDistribution{T}, x::AbstractMatrix{<:Real}) where {T}
    isnothing(d.normalizer) && return x, zero(T)
    return normalize(d.normalizer, x), d.normalizer.log_jac
end

function _apply_normalizer(d::FlowDistribution{T}, x::AbstractVector{<:Real}) where {T}
    isnothing(d.normalizer) && return x, zero(T)
    return normalize(d.normalizer, x), d.normalizer.log_jac
end

function Distributions.logpdf(d::FlowDistribution, x::AbstractVector{<:Real})
    x_norm, log_jac = _apply_normalizer(d, x)
    x_mat = reshape(x_norm, :, 1)
    lp = log_prob(d.model, d.ps, d.st, x_mat)
    return first(lp) + log_jac
end

function Distributions.logpdf(d::FlowDistribution, x::AbstractMatrix{<:Real})
    x_norm, log_jac = _apply_normalizer(d, x)
    lp = log_prob(d.model, d.ps, d.st, x_norm)
    return lp .+ log_jac
end

function Base.rand(rng::AbstractRNG, d::FlowDistribution{T}) where {T}
    z = draw_samples(rng, T, d.model, d.ps, d.st, 1)
    x = isnothing(d.normalizer) ? z : denormalize(d.normalizer, z)
    return T.(vec(x))
end

function Distributions.rand(rng::AbstractRNG, d::FlowDistribution{T}, n::Int) where {T}
    z = draw_samples(rng, T, d.model, d.ps, d.st, n)
    isnothing(d.normalizer) && return z
    return denormalize(d.normalizer, z)
end

# ── Bijectors.jl interface ───────────────────────────────────────────────────

# Normalizing flows are already defined on unconstrained space (ℝⁿ),
# therefore no parameter transformation is required for HMC/NUTS.
Bijectors.bijector(::FlowDistribution) = identity

# Explicitly implement the VectorBijectors interface
Bijectors.VectorBijectors.vec_length(d::FlowDistribution) = d.n_dims
Bijectors.VectorBijectors.linked_vec_length(d::FlowDistribution) = d.n_dims
Bijectors.VectorBijectors.to_vec(d::FlowDistribution) = Base.identity
Bijectors.VectorBijectors.from_vec(d::FlowDistribution) = Base.identity
Bijectors.VectorBijectors.to_linked_vec(d::FlowDistribution) = Base.identity
Bijectors.VectorBijectors.from_linked_vec(d::FlowDistribution) = Base.identity
