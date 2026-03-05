# src/made.jl
using Lux
using Random
using Statistics

"""
    create_degrees(input_dims::Int, hidden_dims::Vector{Int}, out_dims::Int)

Generate degrees for each unit in a MADE network to ensure autoregressive property.
Returns a list of degree vectors (one for input, each hidden layer, and output).
"""
function create_degrees(input_dims::Int, hidden_dims::Vector{Int}, out_dims::Int; sequential::Bool=true)
    # Degrees for input: 1 to D
    degrees = [collect(1:input_dims)]
    
    # Degrees for hidden layers
    for h_dim in hidden_dims
        if sequential
            # Use a deterministic pattern: repeat 1 to D-1
            d_hidden = [((i - 1) % (input_dims - 1)) + 1 for i in 1:h_dim]
            push!(degrees, d_hidden)
        else
            # Sample hidden degrees in [1, D-1]
            push!(degrees, rand(1:(input_dims-1), h_dim))
        end
    end
    
    # Degrees for output: same as input (usually out_dims = input_dims or 2*input_dims)
    # If out_dims is a multiple of input_dims, we repeat the degrees
    if out_dims % input_dims == 0
        n_repeats = out_dims ÷ input_dims
        push!(degrees, repeat(collect(1:input_dims), n_repeats))
    else
        # Fallback for arbitrary output sizes (though usually it's D or 2D)
        push!(degrees, collect(1:out_dims))
    end
    
    return degrees
end

"""
    create_masks(degrees::Vector{Vector{Int}})

Create binary masks from a list of degree vectors.
"""
function create_masks(degrees::Vector{Vector{Int}})
    masks = Matrix{Float32}[]
    # Layer l goes from degrees[l] to degrees[l+1]
    for i in 1:(length(degrees) - 1)
        m_curr = degrees[i]
        m_next = degrees[i+1]
        
        # Matrix shape is (length(m_next), length(m_curr))
        mask = zeros(Float32, length(m_next), length(m_curr))
        
        for r in 1:length(m_next)
            for c in 1:length(m_curr)
                if i < length(degrees)-1
                    # Hidden layers: weight exists if degree(prev) <= degree(curr)
                    if m_next[r] >= m_curr[c]
                        mask[r, c] = 1.0f0
                    end
                else
                    # Output layer: weight exists if degree(prev) < degree(curr)
                    # This ensures y_i depends only on x_{<i}
                    if m_next[r] > m_curr[c]
                        mask[r, c] = 1.0f0
                    end
                end
            end
        end
        push!(masks, mask)
    end
    return masks
end

# ── MaskedDense Layer ────────────────────────────────────────────────────────

struct MaskedDense{F} <: Lux.AbstractLuxLayer
    activation::F
    in_dims::Int
    out_dims::Int
    mask::Matrix{Float32}
end

function MaskedDense(in_dims::Int, out_dims::Int, mask::Matrix{Float32}, activation=identity)
    return MaskedDense(activation, in_dims, out_dims, mask)
end

function Lux.initialparameters(rng::AbstractRNG, l::MaskedDense)
    return (
        weight = Lux.glorot_uniform(rng, l.out_dims, l.in_dims),
        bias   = zeros(Float32, l.out_dims)
    )
end

function Lux.initialstates(rng::AbstractRNG, l::MaskedDense)
    return NamedTuple()
end

function (l::MaskedDense)(x::AbstractArray, ps, st)
    # y = (W .* mask) * x + b
    # We use ps.weight .* l.mask for differentiability
    y = (ps.weight .* l.mask) * x .+ ps.bias
    return l.activation.(y), st
end

# ── MADE Network ────────────────────────────────────────────────────────────

struct MADE <: Lux.AbstractLuxContainerLayer{(:layers,)}
    layers::Lux.Chain
    input_dims::Int
    hidden_dims::Vector{Int}
    out_dims::Int
end

function MADE(input_dims::Int, hidden_dims::Vector{Int}, out_dims::Int; 
              activation=relu, sequential::Bool=true)
    degrees = create_degrees(input_dims, hidden_dims, out_dims; sequential)
    masks = create_masks(degrees)
    
    layers_list = []
    for i in 1:length(masks)
        in_d = size(masks[i], 2)
        out_d = size(masks[i], 1)
        act = (i == length(masks)) ? identity : activation
        push!(layers_list, MaskedDense(in_d, out_d, masks[i], act))
    end
    
    return MADE(Lux.Chain(layers_list...), input_dims, hidden_dims, out_dims)
end

function (l::MADE)(x::AbstractArray, ps, st)
    return Lux.apply(l.layers, x, ps.layers, st.layers)
end
