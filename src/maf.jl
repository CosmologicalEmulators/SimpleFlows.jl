# src/maf.jl
using Lux
using Bijectors
using Random
using LinearAlgebra

# ── MAF Bijector ────────────────────────────────────────────────────────────

struct MAFBijector
    made
    ps
    st
end

"""
    inverse_and_log_det(b::MAFBijector, x::AbstractArray)

Density estimation pass (Inverse): u = (x - m) * exp(-alpha). Fast O(1).
"""
function Bijectors.with_logabsdet_jacobian(b::MAFBijector, x::AbstractMatrix)
    # MADE(x) returns (m, log_alpha) concatenated
    out, _ = Lux.apply(b.made, x, b.ps, b.st)
    D = size(x, 1)
    m = out[1:D, :]
    log_alpha = out[D+1:end, :]
    
    # Inverse transform
    u = (x .- m) .* exp.(-log_alpha)
    
    # Log-determinant: sum of -log_alpha across dimensions
    lad = -sum(log_alpha; dims=1)
    
    # Return as (output, logabsdet)
    return u, vec(lad)
end

"""
    forward_and_log_det(b::MAFBijector, u::AbstractArray)

Sampling pass (Forward): x_i = u_i * exp(alpha_i) + m_i. Sequential O(D).
"""
function forward_and_log_det(b::MAFBijector, u::AbstractMatrix)
    D, N = size(u)
    T = eltype(u)
    
    # Initialize x with zeros. We will fill it dimension by dimension.
    # To be Zygote friendly, we can use a loop and vcat/hcat or just use a copy if allowed.
    # Actually, sampling is usually not the target of AD during training, but for completeness:
    
    x = zeros(T, D, N)
    
    for i in 1:D
        # Compute parameters for the current state of x
        out, _ = Lux.apply(b.made, x, b.ps, b.st)
        m_i = out[i, :]
        log_alpha_i = out[D+i, :]
        
        # Update row i of x
        # x[i, :] = u[i, :] .* exp.(log_alpha_i) .+ m_i
        # Non-mutating version for Zygote:
        new_row = u[i:i, :] .* exp.(log_alpha_i') .+ m_i'
        x = vcat(x[1:i-1, :], new_row, x[i+1:end, :])
    end
    
    # Compute final log_alpha for the fully constructed x to get log_det
    out_final, _ = Lux.apply(b.made, x, b.ps, b.st)
    log_alpha_final = out_final[D+1:end, :]
    lad = sum(log_alpha_final; dims=1)
    
    return x, vec(lad)
end

# ── ReversePermute ──────────────────────────────────────────────────────────

struct ReversePermute <: Lux.AbstractLuxLayer end

function (l::ReversePermute)(x::AbstractArray, ps, st)
    return x[end:-1:1, :], st
end

# ── MaskedAutoregressiveFlow ───────────────────────────────────────────────

@concrete struct MaskedAutoregressiveFlow <: Lux.AbstractLuxContainerLayer{(:mades,)}
    mades
    dist_dims::Int
    n_transforms::Int
    hidden_layer_sizes::Vector{Int}
end

function MaskedAutoregressiveFlow(; n_transforms::Int, dist_dims::Int, hidden_layer_sizes::Vector{Int}, activation=relu)
    mades_list = [MADE(dist_dims, hidden_layer_sizes, 2 * dist_dims; activation) for _ in 1:n_transforms]
    keys_ = ntuple(i -> Symbol(:made_, i), n_transforms)
    mades = NamedTuple{keys_}(Tuple(mades_list))
    return MaskedAutoregressiveFlow(mades, dist_dims, n_transforms, hidden_layer_sizes)
end

function Lux.initialstates(rng::AbstractRNG, m::MaskedAutoregressiveFlow)
    return (mades = Lux.initialstates(rng, m.mades),)
end
