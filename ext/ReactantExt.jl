module ReactantExt

using SimpleFlows
using Reactant

# Reactant-specific array type aliases for vectors and matrices.
# Used for dispatch since Reactant does not currently export a stable public abstract supertype.
const DeviceVec{T} = Union{Reactant.TracedRArray{T, 1}, Reactant.ConcretePJRTArray{T, 1}}
const DeviceMat{T} = Union{Reactant.TracedRArray{T, 2}, Reactant.ConcretePJRTArray{T, 2}}

# ── 1. Vectorized Bin Search Overloads ────────────────────────────────────────

# XLA-compatible vectorized bin search. Uses a dense broadcast comparison and 
# reduction to avoid dynamic branch indexing which is unsupported/inefficient on XLA.
function SimpleFlows.compute_bin_idx(cum_arrays::DeviceMat, inputs::DeviceVec, K::Int)
    cmp = reshape(inputs, :, 1) .>= cum_arrays
    idx = vec(sum(cmp; dims=2))
    return clamp.(idx, 1, K)
end

function SimpleFlows.compute_bin_idx(cum_arrays::DeviceMat, inputs::AbstractVector, K::Int)
    cmp = reshape(inputs, :, 1) .>= cum_arrays
    idx = vec(sum(cmp; dims=2))
    return clamp.(idx, 1, K)
end

function SimpleFlows.compute_bin_idx(cum_arrays::AbstractMatrix, inputs::DeviceVec, K::Int)
    cmp = reshape(inputs, :, 1) .>= cum_arrays
    idx = vec(sum(cmp; dims=2))
    return clamp.(idx, 1, K)
end

# ── 2. Differentiable Gather Overloads ────────────────────────────────────────

# XLA-compatible differentiable gather. Uses one-hot matrix multiplication / broadcast
# to retrieve values without dynamic indexing paths.
function SimpleFlows.gather_from_matrix(A::DeviceMat, indices::DeviceVec)
    M, C = size(A)
    one_hot = eltype(A).(reshape(1:C, 1, C) .== reshape(indices, M, 1))
    return vec(sum(A .* one_hot; dims=2))
end

function SimpleFlows.gather_from_matrix(A::DeviceMat, indices::AbstractVector)
    M, C = size(A)
    one_hot = eltype(A).(reshape(1:C, 1, C) .== reshape(indices, M, 1))
    return vec(sum(A .* one_hot; dims=2))
end

function SimpleFlows.gather_from_matrix(A::AbstractMatrix, indices::DeviceVec)
    M, C = size(A)
    one_hot = eltype(A).(reshape(1:C, 1, C) .== reshape(indices, M, 1))
    return vec(sum(A .* one_hot; dims=2))
end

# ── 3. Static Coupling Slice & Reconstruction ─────────────────────────────────

function SimpleFlows._apply_mask(bj::SimpleFlows.MaskedCoupling, x::DeviceMat, transform_fn)
    D, N = size(x)
    m = bj.mask
    
    # 1. Conditioning
    x_cond = x .* reshape(.!m, D, 1)
    params = bj.conditioner(x_cond)
    
    # 2. Transform the active dims only
    masked_indices = findall(m)
    x_tr = x[masked_indices, :]
    
    bj_inner = bj.bijector_constructor(params)
    y_tr, ld_tr = transform_fn(bj_inner, x_tr)
    
    # 3. Reconstruct full y
    unmasked_indices = findall(.!m)
    x_unmasked = x[unmasked_indices, :]
    z = vcat(x_unmasked, y_tr)
    
    # Precompute static permutation vector
    p_inv = zeros(Int, D)
    D_unmasked = length(unmasked_indices)
    for j in 1:D_unmasked
        p_inv[unmasked_indices[j]] = j
    end
    for k in 1:length(masked_indices)
        p_inv[masked_indices[k]] = D_unmasked + k
    end
    
    y = z[p_inv, :]
    
    return y, SimpleFlows.dsum(ld_tr; dims=(1,))
end

# ── 4. Reactant Device Conversion (to_reactant) ──────────────────────────────

function SimpleFlows.to_reactant(norm::SimpleFlows.MinMaxNormalizer)
    return SimpleFlows.MinMaxNormalizer(
        Reactant.to_rarray(norm.x_min),
        Reactant.to_rarray(norm.x_max),
        Reactant.to_rarray(norm.log_jac)
    )
end

function SimpleFlows.to_reactant(flow::SimpleFlows.FlowDistribution{T, M}) where {T, M}
    reactant_ps = Reactant.to_rarray(flow.ps)
    reactant_norm = isnothing(flow.normalizer) ? nothing : SimpleFlows.to_reactant(flow.normalizer)
    
    return SimpleFlows.FlowDistribution{T, M}(
        flow.model,
        reactant_ps,
        flow.st,
        flow.n_dims,
        flow.hidden_layer_sizes,
        reactant_norm
    )
end

end
