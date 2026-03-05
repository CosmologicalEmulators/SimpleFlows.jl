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

# ── Architecture dict ─────────────────────────────────────────────────────────

function _flow_to_dict(flow::FlowDistribution)
    d = Dict(
        "architecture"      => (flow.model isa RealNVP ? "RealNVP" : "NSF"),
        "n_transforms"      => flow.model.n_transforms,
        "dist_dims"         => flow.n_dims,
        "hidden_layer_sizes" => flow.hidden_layer_sizes,
        "activation"        => "gelu",
    )
    if flow.model isa NeuralSplineFlow
        d["K"] = flow.model.K
        d["tail_bound"] = flow.model.tail_bound
    end
    return d
end

function _build_flow_from_dict(d::AbstractDict, ::Type{T}=Float32, rng::AbstractRNG=Random.default_rng()) where {T<:Real}
    arch = d["architecture"]
    arch_sym = (arch == "RealNVP" ? :RealNVP : :NSF)
    
    # Support both new (hidden_layer_sizes) and legacy (hidden_dims + n_layers) formats
    hidden_layer_sizes = if haskey(d, "hidden_layer_sizes")
        Int.(d["hidden_layer_sizes"])
    else
        fill(Int(d["hidden_dims"]), Int(d["n_layers"]))
    end
    
    kwargs = Dict{Symbol, Any}(
        :architecture       => arch_sym,
        :n_transforms       => Int(d["n_transforms"]),
        :dist_dims          => Int(d["dist_dims"]),
        :hidden_layer_sizes => hidden_layer_sizes,
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
    save_trained_flow(path, flow)

Save a trained `FlowDistribution` to `path` (created if absent).

Files written:
- `flow_setup.json` — architecture hyperparameters
- `weights.npz`    — flattened parameters (Python-readable via `numpy.load`)
"""
function save_trained_flow(path::String, flow::FlowDistribution)
    mkpath(path)

    # Architecture
    open(joinpath(path, "flow_setup.json"), "w") do io
        JSON.print(io, _flow_to_dict(flow), 4)
    end

    # Parameters
    flat = _flatten_params(flow.ps)
    # Also save normalizer if present
    if !isnothing(flow.normalizer)
        flat["normalizer_xmin"] = flow.normalizer.x_min
        flat["normalizer_xmax"] = flow.normalizer.x_max
    end
    NPZ.npzwrite(joinpath(path, "weights.npz"), flat)

    @info "Flow saved to $path ($(length(flat)) weight arrays)"
    return path
end

"""
    load_trained_flow(path; rng) -> FlowDistribution

Load a `FlowDistribution` previously saved with `save_trained_flow`.
"""
function load_trained_flow(path::String; rng::AbstractRNG=Random.default_rng())
    setup = JSON.parsefile(joinpath(path, "flow_setup.json"))
    flat  = NPZ.npzread(joinpath(path, "weights.npz"))

    # Infer type T from the flat arrays
    T = Float32
    for (k, v) in flat
        if v isa AbstractArray
            T = eltype(v)
            break
        end
    end

    # Build a freshly-initialised flow to get the correct parameter structure
    flow = _build_flow_from_dict(setup, T, rng)

    # Separate normalizer arrays from model weights
    if haskey(flat, "normalizer_xmin")
        xmin = flat["normalizer_xmin"]
        xmax = flat["normalizer_xmax"]
        T_norm = eltype(xmin)
        flow.normalizer = MinMaxNormalizer(xmin, xmax, sum(-log.(xmax .- xmin)))
        delete!(flat, "normalizer_xmin")
        delete!(flat, "normalizer_xmax")
    end

    # Overwrite random params with the saved ones
    flow.ps = _unflatten_params(flat, flow.ps)

    @info "Flow loaded from $path"
    return flow
end
