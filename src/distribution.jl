"""
    FlowDistribution([Type=Float32]; architecture=:RealNVP, n_transforms,
                     dist_dims, hidden_layer_sizes, kwargs...)

A trained normalizing flow wrapped as a `Distributions.jl`
`ContinuousMultivariateDistribution`, usable directly in `Turing.jl` models.

# Fields
- `model`: the RealNVP, NeuralSplineFlow or MaskedAutoregressiveFlow Lux model
- `ps`: trained parameters (NamedTuple)
- `st`: Lux state (NamedTuple)
- `n_dims`, `hidden_layer_sizes`: architecture metadata for serialization
- `parameter_names`: optional unique names, one per dimension in coordinate order;
  saved and restored by `save_trained_flow`/`load_trained_flow`
- `normalizer`: fitted `MinMaxNormalizer` or `WhiteningNormalizer` (always present after training)
"""
mutable struct FlowDistribution{T<:Real, M<:AbstractLuxLayer} <: ContinuousMultivariateDistribution
    model              :: M
    ps
    st
    n_dims             :: Int
    hidden_layer_sizes :: Vector{Int}
    normalizer         :: Union{Nothing, MinMaxNormalizer{T}, WhiteningNormalizer{T}}
    parameter_names    :: Union{Nothing, Vector{String}}

    # Precision cannot be inferred from a missing normalizer. An explicit inner
    # constructor avoids generating an outer method with an unbound T.
    function FlowDistribution{T, M}(model, ps, st, n_dims, hidden_layer_sizes,
                                    normalizer, parameter_names) where {T<:Real, M<:AbstractLuxLayer}
        return new{T, M}(model, ps, st, n_dims, hidden_layer_sizes, normalizer, parameter_names)
    end
end

# Preserve the inferable fitted-normalizer positional constructor. Without a
# normalizer, callers must select the precision through the keyword constructor.
FlowDistribution(model::M, ps, st, n_dims::Int, hidden_layer_sizes::Vector{Int},
                 normalizer::Union{MinMaxNormalizer{T}, WhiteningNormalizer{T}},
                 parameter_names::Union{Nothing, Vector{String}}) where {T<:Real, M<:AbstractLuxLayer} =
    FlowDistribution{T, M}(model, ps, st, n_dims, hidden_layer_sizes, normalizer, parameter_names)

function _check_parameter_names(names, n_dims::Int)
    names isa AbstractVector && all(n -> n isa AbstractString, names) || throw(ArgumentError("parameter_names must be a vector of strings."))
    length(names) == n_dims || throw(ArgumentError("parameter_names has length $(length(names)) but the flow has $n_dims dimensions."))
    length(unique(names)) == n_dims || throw(ArgumentError("parameter_names must be unique."))
    return String.(names)
end

# Convenience constructor without parameter names.
FlowDistribution{T, M}(model, ps, st, n_dims, hidden_layer_sizes, normalizer) where {T<:Real, M<:AbstractLuxLayer} =
    FlowDistribution{T, M}(model, ps, st, n_dims, hidden_layer_sizes, normalizer, nothing)

"""
    FlowDistribution([Type=Float32]; architecture=:RealNVP, n_transforms, dist_dims, 
                       hidden_layer_sizes, hidden_dims=64, n_layers=3,
                       activation=gelu, rng=Random.default_rng(), K=8, tail_bound=3.0,
                       parameter_names=nothing, identity_init=true)

Construct and initialise a `FlowDistribution`. `architecture` can be `:RealNVP`, `:NSF` or `:MAF`.

# Initialization
- `identity_init=true` (default): the hidden layers get the usual random initialization, but the
  final (output-head) layer of every conditioner is set so that each transform starts as the
  identity: RealNVP shift = log-scale = 0, MAF mean = log-scale = 0, NSF zero head weights, uniform
  width/height logits (equal bins) and derivative biases giving derivative 1 (an exact identity
  spline). The initial density is therefore the standard-normal base density (times the
  normalizer's Jacobian once one is attached), instead of random affine layers that can amplify
  tails exponentially (e.g. a 3-layer RealNVP on whitened data overflowed the Float32 Gaussian
  base before any update). Gradients of the head layers are non-zero from the first step, so
  training can start immediately; hidden layers receive gradient after the first update.
- `identity_init=false`: the legacy initialization (all layers random), reproduced exactly from
  the same `rng`. The random stream is consumed identically in both modes, so hidden layers are
  identical between them. Saved artifacts store the weights, so loading never depends on this flag.
"""
function FlowDistribution(::Type{T}=Float32;
                            architecture=:RealNVP,
                            n_transforms::Int, dist_dims::Int,
                            hidden_layer_sizes::Vector{Int}=Int[],
                            hidden_dims::Int=64, n_layers::Int=3,
                            activation=gelu,
                            K=8, tail_bound=3.0,
                            parameter_names::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
                            identity_init::Bool=true,
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
    identity_init && _identity_init!(model, ps, T)
    names = isnothing(parameter_names) ? nothing : _check_parameter_names(parameter_names, dist_dims)
    return FlowDistribution{T, typeof(model)}(model, ps, st, dist_dims, hidden_layer_sizes, nothing, names)
end

# ── Identity initialization of the conditioner output heads ──────────────────

# Same constant as the spline's derivative padding: derivative = min_derivative + softplus(c) = 1.
# (`min_derivative` is the spline default; NSF does not override it.)
_nsf_identity_derivative_logit(::Type{T}, min_derivative=1e-3) where {T} = T(log(exp(1 - T(min_derivative)) - 1))

_head(chain_ps::NamedTuple) = last(values(chain_ps))        # parameters of the last (output) Dense layer

function _identity_init!(model::RealNVP, ps, ::Type{T}) where {T}
    for conditioner in values(ps.conditioners)
        head = _head(conditioner)
        fill!(head.weight, zero(T)); fill!(head.bias, zero(T))      # shift = 0, log_scale = 0
    end
    return ps
end

function _identity_init!(model::MaskedAutoregressiveFlow, ps, ::Type{T}) where {T}
    for made in values(ps.mades)
        head = _head(made.layers)
        fill!(head.weight, zero(T)); fill!(head.bias, zero(T))      # mean = 0, log_scale = 0
    end
    return ps
end

function _identity_init!(model::NeuralSplineFlow, ps, ::Type{T}) where {T}
    K = model.K
    for (i, conditioner) in enumerate(values(ps.conditioners))
        head = _head(conditioner)
        D_tr = sum(model.mask_list[i])
        fill!(head.weight, zero(T))
        fill!(head.bias, zero(T))
        # The output of one sample is reshaped to (D_tr, 3K-1): columns 1:K width logits,
        # K+1:2K height logits (zero = uniform bins), 2K+1:3K-1 derivative logits.
        b = reshape(head.bias, D_tr, 3K - 1)
        b[:, 2K+1:end] .= _nsf_identity_derivative_logit(T)
    end
    return ps
end

# ── Distributions.jl interface ────────────────────────────────────────────────

Distributions.length(d::FlowDistribution) = d.n_dims

function _apply_normalizer(d::FlowDistribution{T}, x::AbstractMatrix) where {T}
    normalizer = d.normalizer
    normalizer === nothing && return x, zero(T)
    return SimpleFlows.normalize(normalizer, x), _normalizer_log_jac(normalizer)
end

function _apply_normalizer(d::FlowDistribution{T}, x::AbstractVector) where {T}
    normalizer = d.normalizer
    normalizer === nothing && return x, zero(T)
    return SimpleFlows.normalize(normalizer, x), _normalizer_log_jac(normalizer)
end

# Widened to AbstractVector (instead of AbstractVector{<:Real}) to allow Reactant traced arrays
# whose eltypes might not subtype Real.
function Distributions.logpdf(d::FlowDistribution, x::AbstractVector)
    x_norm, log_jac = _apply_normalizer(d, x)
    x_mat = reshape(x_norm, :, 1)
    lp = log_prob(d.model, d.ps, d.st, x_mat)
    return first(lp) + log_jac
end

# Widened to AbstractMatrix (instead of AbstractMatrix{<:Real}) to allow Reactant traced arrays
# whose eltypes might not subtype Real.
function Distributions.logpdf(d::FlowDistribution, x::AbstractMatrix)
    x_norm, log_jac = _apply_normalizer(d, x)
    lp = log_prob(d.model, d.ps, d.st, x_norm)
    return lp .+ log_jac
end

function Base.rand(rng::AbstractRNG, d::FlowDistribution{T}) where {T}
    z = draw_samples(rng, T, d.model, d.ps, d.st, 1)
    normalizer = d.normalizer
    x = normalizer === nothing ? z : SimpleFlows.denormalize(normalizer, z)
    return T.(vec(x))
end

function Distributions.rand(rng::AbstractRNG, d::FlowDistribution{T}, n::Int) where {T}
    z = draw_samples(rng, T, d.model, d.ps, d.st, n)
    normalizer = d.normalizer
    normalizer === nothing && return z
    return SimpleFlows.denormalize(normalizer, z)
end

# ── Bijectors.jl interface ───────────────────────────────────────────────────

# Normalizing flows are already defined on unconstrained space (ℝⁿ),
# therefore no parameter transformation is required for HMC/NUTS.
Bijectors.bijector(::FlowDistribution) = identity

if isdefined(Bijectors, :VectorBijectors)
    # Explicitly implement the VectorBijectors interface
    Bijectors.VectorBijectors.vec_length(d::FlowDistribution) = d.n_dims
    Bijectors.VectorBijectors.linked_vec_length(d::FlowDistribution) = d.n_dims
    Bijectors.VectorBijectors.to_vec(d::FlowDistribution) = Base.identity
    Bijectors.VectorBijectors.from_vec(d::FlowDistribution) = Base.identity
    Bijectors.VectorBijectors.to_linked_vec(d::FlowDistribution) = Base.identity
    Bijectors.VectorBijectors.from_linked_vec(d::FlowDistribution) = Base.identity
end
