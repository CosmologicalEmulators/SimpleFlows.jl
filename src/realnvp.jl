# ── MLP helper ───────────────────────────────────────────────────────────────

"""Build an MLP conditioner; `hidden_layer_sizes` is a vector of per-layer widths."""
function MLP(in_dims::Int, hidden_layer_sizes::Vector{Int}, out_dims::Int;
             activation=gelu)
    widths = [in_dims; hidden_layer_sizes; out_dims]
    layers = [Dense(widths[i] => widths[i+1],
                    i < length(widths) - 1 ? activation : identity)
              for i in 1:(length(widths) - 1)]
    return Chain(layers...)
end

# ── RealNVP ───────────────────────────────────────────────────────────────────

"""
    RealNVP(; n_transforms, dist_dims, hidden_dims, n_layers, activation=gelu)

Real-valued Non-Volume Preserving (RealNVP) normalizing flow.
Each coupling layer is conditioned by an MLP producing affine parameters.
Masks alternate between even/odd dimensions.
"""
@concrete struct RealNVP <: AbstractLuxContainerLayer{(:conditioners,)}
    conditioners
    mask_list          :: Vector{BitVector}
    dist_dims          :: Int
    n_transforms       :: Int
    hidden_layer_sizes :: Vector{Int}
end

function RealNVP(; n_transforms::Int, dist_dims::Int,
                   hidden_layer_sizes::Vector{Int}, activation=gelu)
    D = dist_dims
    
    # Pre-generate masks
    mask_list = [BitVector(collect(1:D) .% 2 .== i % 2)
                 for i in 1:n_transforms]
                 
    mlps = []
    for i in 1:n_transforms
        m = mask_list[i]
        D_tr = sum(m)
        push!(mlps, MLP(D, hidden_layer_sizes, 2 * D_tr; activation))
    end
    
    keys_ = ntuple(i -> Symbol(:conditioners_, i), n_transforms)
    conditioners = NamedTuple{keys_}(Tuple(mlps))
    return RealNVP(conditioners, mask_list, D, n_transforms, hidden_layer_sizes)
end

function Lux.initialstates(rng::AbstractRNG, m::RealNVP)
    return (; mask_list=m.mask_list, conditioners=Lux.initialstates(rng, m.conditioners))
end
