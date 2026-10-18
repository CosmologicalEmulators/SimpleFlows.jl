# ── Parameter flatten / unflatten (for NPZ serialization) ───────────────────

"""Recursively flatten a NamedTuple of arrays into a flat Dict{String,Array}."""
function _flatten_params(nt::NamedTuple, prefix::String="")
    flat = Dict{String, Array}()
    for (k, v) in pairs(nt)
        key = isempty(prefix) ? string(k) : prefix * "__" * string(k)
        if v isa AbstractArray
            flat[key] = Array(v)          # ensure plain CPU Array
        elseif v isa NamedTuple
            merge!(flat, _flatten_params(v, key))
        end
    end
    return flat
end

"""Reconstruct a NamedTuple from a flat Dict, using `template` for structure."""
function _unflatten_params(flat::AbstractDict{String}, template::NamedTuple, prefix::String="")
    vals = map(keys(template)) do k
        key = isempty(prefix) ? string(k) : prefix * "__" * string(k)
        v = template[k]
        if v isa AbstractArray
            raw = flat[key]
            return eltype(v).(reshape(raw, size(v)))
        elseif v isa NamedTuple
            return _unflatten_params(flat, v, key)
        else
            return v
        end
    end
    return NamedTuple{keys(template)}(Tuple(vals))
end

# ── Activation whitelist ──────────────────────────────────────────────────────

# Artifact format version written by `save_trained_flow`. Artifacts without a
# `format_version` key are legacy (version 1): float precision is inferred from
# the stored arrays and the activation is assumed to be GELU.
const ARTIFACT_FORMAT_VERSION = 3

# Activation names are resolved only through this table; artifact strings are
# never evaluated. Note that in the current NNlib `gelu === gelu_tanh`, so both
# names resolve to the same function.
const _ACTIVATIONS = (
    "identity"     => identity,
    "relu"         => relu,
    "relu6"        => relu6,
    "gelu"         => gelu,
    "gelu_tanh"    => gelu_tanh,
    "gelu_erf"     => gelu_erf,
    "tanh"         => tanh,
    "tanh_fast"    => tanh_fast,
    "sigmoid"      => sigmoid,
    "sigmoid_fast" => sigmoid_fast,
    "swish"        => swish,
    "hardswish"    => hardswish,
    "softplus"     => softplus,
    "softsign"     => softsign,
    "elu"          => elu,
    "selu"         => selu,
    "celu"         => celu,
    "leakyrelu"    => leakyrelu,
    "mish"         => mish,
    "logsigmoid"   => logsigmoid,
)

function _activation_name(f)
    matches = [name for (name, g) in _ACTIVATIONS if g === f]
    isempty(matches) && throw(ArgumentError(
        "Cannot serialize activation $(repr(f)): only the built-in activations " *
        "$(join(first.(_ACTIVATIONS), ", ")) are supported."))
    # Prefer the function's own name when several names are aliases of it.
    own = string(nameof(f))
    return own in matches ? own : first(matches)
end

function _activation_from_name(name)
    name isa AbstractString || throw(ArgumentError("Invalid activation entry in artifact: $(repr(name))"))
    for (n, f) in _ACTIVATIONS
        n == name && return f
    end
    throw(ArgumentError("Unsupported activation \"$name\" in artifact."))
end

# Recover the activation actually used by the first hidden layer of the model.
_first_activation(l::Lux.Chain) = first(values(l.layers)).activation
_first_activation(m::Union{RealNVP, NeuralSplineFlow}) = _first_activation(first(values(m.conditioners)))
_first_activation(m::MADE) = _first_activation(m.layers)
_first_activation(m::MaskedAutoregressiveFlow) = _first_activation(first(values(m.mades)))

# ── Architecture dict ─────────────────────────────────────────────────────────

_precision_name(::Type{Float32}) = "Float32"
_precision_name(::Type{Float64}) = "Float64"
_precision_name(T::Type) = throw(ArgumentError("Unsupported flow precision $T; only Float32 and Float64 can be saved."))

function _precision_from_name(name)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    throw(ArgumentError("Unsupported precision $(repr(name)) in artifact."))
end

_normalizer_kind(::Nothing) = "none"
_normalizer_kind(::MinMaxNormalizer) = "minmax"
_normalizer_kind(::WhiteningNormalizer) = "whiten"

function _flow_to_dict(flow::FlowDistribution{T}) where {T}
    arch = if flow.model isa RealNVP
        "RealNVP"
    elseif flow.model isa NeuralSplineFlow
        "NSF"
    elseif flow.model isa MaskedAutoregressiveFlow
        "MAF"
    else
        error("Unknown architecture type: $(typeof(flow.model))")
    end

    d = Dict(
        "format_version"    => ARTIFACT_FORMAT_VERSION,
        "precision"         => _precision_name(T),
        "architecture"      => arch,
        "n_transforms"      => flow.model.n_transforms,
        "dist_dims"         => flow.n_dims,
        "hidden_layer_sizes" => flow.hidden_layer_sizes,
        "activation"        => _activation_name(_first_activation(flow.model)),
        "normalizer"        => _normalizer_kind(flow.normalizer),
    )
    if flow.model isa NeuralSplineFlow
        d["K"] = flow.model.K
        d["tail_bound"] = flow.model.tail_bound
    end
    return d
end

function _build_flow_from_dict(d::AbstractDict, ::Type{T}=Float32, rng::AbstractRNG=Random.default_rng()) where {T<:Real}
    arch = d["architecture"]
    arch_sym = if arch == "RealNVP"
        :RealNVP
    elseif arch == "NSF"
        :NSF
    elseif arch == "MAF"
        :MAF
    else
        error("Unknown architecture: $arch")
    end
    
    # Support both new (hidden_layer_sizes) and legacy (hidden_dims + n_layers) formats
    hidden_layer_sizes = if haskey(d, "hidden_layer_sizes")
        Int.(d["hidden_layer_sizes"])
    else
        fill(Int(d["hidden_dims"]), Int(d["n_layers"]))
    end
    
    # Legacy artifacts always wrote "gelu" regardless of the real activation.
    activation = get(d, "format_version", 1) >= 2 ? _activation_from_name(get(d, "activation", nothing)) : gelu

    kwargs = Dict{Symbol, Any}(
        :architecture       => arch_sym,
        :n_transforms       => Int(d["n_transforms"]),
        :dist_dims          => Int(d["dist_dims"]),
        :hidden_layer_sizes => hidden_layer_sizes,
        :activation         => activation,
        :rng                => rng,
    )
    
    if arch == "NSF"
        kwargs[:K] = Int(d["K"])
        kwargs[:tail_bound] = Float64(d["tail_bound"])
    end
    
    return FlowDistribution(T; kwargs...)
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    save_trained_flow(path, flow; parameter_names=flow.parameter_names)

Save a trained `FlowDistribution` to `path` (created if absent).

Files written:
- `flow_setup.json` — architecture hyperparameters, format version, precision,
  activation, normalizer kind (`none`/`minmax`/`whiten`) and, if given,
  `parameter_names`
- `weights.npz`    — flattened parameters and normalizer arrays
  (`normalizer_xmin/xmax` or `normalizer_mean/chol`), readable via `numpy.load`

`parameter_names` (optional, one unique string per dimension, in the flow's
coordinate order) defaults to `flow.parameter_names`, so names survive a
load/save cycle. They are stored in the JSON metadata, restored into
`flow.parameter_names` by `load_trained_flow`, and can also be read with
[`load_parameter_names`](@ref).
"""
function save_trained_flow(path::String, flow::FlowDistribution;
                           parameter_names::Union{Nothing, AbstractVector{<:AbstractString}}=flow.parameter_names)
    # Build (and thereby validate) the metadata before touching the filesystem.
    setup = _flow_to_dict(flow)
    if !isnothing(parameter_names)
        setup["parameter_names"] = _check_parameter_names(parameter_names, flow.n_dims)
    end
    mkpath(path)

    open(joinpath(path, "flow_setup.json"), "w") do io
        JSON.print(io, setup, 4)
    end

    flat = _flatten_params(flow.ps)
    nrm = flow.normalizer
    if nrm isa MinMaxNormalizer
        flat["normalizer_xmin"] = Array(nrm.x_min)
        flat["normalizer_xmax"] = Array(nrm.x_max)
    elseif nrm isa WhiteningNormalizer
        flat["normalizer_mean"] = Array(nrm.mean)
        flat["normalizer_chol"] = Array(nrm.chol)
    end
    NPZ.npzwrite(joinpath(path, "weights.npz"), flat)

    @info "Flow saved to $path ($(length(flat)) weight arrays)"
    return path
end

"""
    load_parameter_names(path) -> Union{Nothing, Vector{String}}

Return the `parameter_names` stored by `save_trained_flow`, or `nothing` if none
were saved (including all legacy artifacts). The length is validated against
`dist_dims`.
"""
function load_parameter_names(path::String)
    setup = JSON.parsefile(joinpath(path, "flow_setup.json"))
    return load_parameter_names(setup, Int(setup["dist_dims"]))
end

function load_parameter_names(setup::AbstractDict, n_dims::Int)
    haskey(setup, "parameter_names") || return nothing
    return _check_parameter_names(setup["parameter_names"], n_dims)
end

function _load_normalizer!(flat, setup, version::Int, ::Type{T}, d::Int) where {T}
    kind = if version >= 3
        k = get(setup, "normalizer", nothing)
        k in ("none", "minmax", "whiten") || throw(ArgumentError("Unknown normalizer kind $(repr(k)) in artifact."))
        k
    else
        haskey(flat, "normalizer_xmin") ? "minmax" : "none"
    end
    present = [k for k in ("normalizer_xmin", "normalizer_xmax", "normalizer_mean", "normalizer_chol") if haskey(flat, k)]
    expected = kind == "minmax" ? ["normalizer_xmin", "normalizer_xmax"] :
               kind == "whiten" ? ["normalizer_mean", "normalizer_chol"] : String[]
    sort(present) == sort(expected) ||
        throw(ArgumentError("Normalizer arrays $(present) do not match normalizer kind \"$kind\"."))
    arrays = Dict(k => pop!(flat, k) for k in present)

    if kind == "minmax"
        xmin = T.(vec(arrays["normalizer_xmin"]))
        xmax = T.(vec(arrays["normalizer_xmax"]))
        (length(xmin) == d && length(xmax) == d && all(isfinite, xmin) && all(isfinite, xmax) && all(xmax .> xmin)) ||
            throw(ArgumentError("Invalid min-max normalizer arrays in artifact."))
        return MinMaxNormalizer(xmin, xmax, T(sum(-log.(xmax .- xmin))))
    elseif kind == "whiten"
        μ = T.(vec(arrays["normalizer_mean"]))
        L = T.(arrays["normalizer_chol"])
        (length(μ) == d && size(L) == (d, d) && all(isfinite, μ) && all(isfinite, L) &&
         istril(L) && all(>(0), diag(L))) ||
            throw(ArgumentError("Invalid whitening normalizer arrays in artifact (need finite mean and lower-triangular factor with positive diagonal)."))
        return WhiteningNormalizer(μ, Matrix(L), _whitening_log_jac(L))
    end
    return nothing
end

"""
    load_trained_flow(path; rng) -> FlowDistribution

Load a `FlowDistribution` previously saved with `save_trained_flow`.
Artifacts with no `format_version`, or `format_version == 1`, are legacy:
GELU activation is assumed and the precision is inferred from the weights.
Unknown versions, normalizer kinds, activations or precisions throw `ArgumentError`.
"""
function load_trained_flow(path::String; rng::AbstractRNG=Random.default_rng())
    setup = JSON.parsefile(joinpath(path, "flow_setup.json"))
    flat  = NPZ.npzread(joinpath(path, "weights.npz"))

    version = get(setup, "format_version", 1)
    version isa Integer && 1 <= version <= ARTIFACT_FORMAT_VERSION ||
        throw(ArgumentError("Unsupported artifact format_version $(repr(version)); this SimpleFlows supports up to $ARTIFACT_FORMAT_VERSION."))

    T = if version >= 2
        _precision_from_name(get(setup, "precision", nothing))
    else
        # Legacy: infer from the first model weight (not the normalizer).
        wkeys = sort!([k for k in keys(flat) if !startswith(k, "normalizer_")])
        isempty(wkeys) ? Float32 : eltype(flat[first(wkeys)])
    end

    # Build a freshly-initialised flow to get the correct parameter structure
    flow = _build_flow_from_dict(setup, T, rng)

    flow.parameter_names = load_parameter_names(setup, flow.n_dims)
    flow.normalizer = _load_normalizer!(flat, setup, Int(version), T, flow.n_dims)

    # Overwrite random params with the saved ones
    flow.ps = _unflatten_params(flat, flow.ps)

    @info "Flow loaded from $path"
    return flow
end
