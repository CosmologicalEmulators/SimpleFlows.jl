# MODIFIED: layers.jl — fused gaussian_logpdf_sum and @views for efficiency

# ── Utility ──────────────────────────────────────────────────────────────────

"""Sum and drop the given dimensions (batch-safe reduction)."""
dsum(x; dims) = dropdims(sum(x; dims=dims); dims=dims)

"""Single fused reduction that avoids intermediate matrix allocation."""
function gaussian_logpdf_sum(x::AbstractMatrix{T}) where T
    c = T(-0.5f0 * log(2π))
    return dropdims(sum(muladd.(x, T(-0.5) .* x, c); dims=1); dims=1)
end

# ── AffineBijector ────────────────────────────────────────────────────────────

"""
    AffineBijector(shift, log_scale)

Elementwise affine bijection: y = x ⊙ exp(log_scale) + shift.
Constructed from a concatenated [shift; log_scale] array produced by an MLP.
"""
@concrete struct AffineBijector
    shift       <: AbstractArray
    log_scale   <: AbstractArray
end

function AffineBijector(params::AbstractArray)
    n = size(params, 1) ÷ 2
    idx = ntuple(Returns(Colon()), ndims(params) - 1)
    return @views AffineBijector(params[1:n, idx...], params[(n + 1):end, idx...])
end

function forward_and_log_det(b::AffineBijector, x::AbstractArray)
    y = x .* exp.(b.log_scale) .+ b.shift
    return y, b.log_scale
end

function inverse_and_log_det(b::AffineBijector, y::AbstractArray)
    x = (y .- b.shift) ./ exp.(b.log_scale)
    return x, -b.log_scale
end

# ── MaskedCoupling ────────────────────────────────────────────────────────────

"""
    MaskedCoupling(mask, conditioner, bijector_constructor, perm=nothing, invperm=nothing, D_tr=nothing)

Coupling layer using a binary mask. Unmasked dimensions condition the bijector;
masked dimensions are transformed.
"""
struct MaskedCoupling{M, C, B, P, IP, DT}
    mask::M
    conditioner::C
    bijector_constructor::B
    perm::P
    invperm::IP
    D_tr::DT
end

function MaskedCoupling(mask, conditioner, bijector_constructor)
    return MaskedCoupling(mask, conditioner, bijector_constructor, nothing, nothing, nothing)
end

function _apply_mask(bj::MaskedCoupling, x::AbstractMatrix, transform_fn)
    m = bj.mask
    
    # 1. Conditioning
    # x_cond must be full D-dimensional for the MLP conditioner
    x_cond = x .* .!m
    params = bj.conditioner(x_cond)
    
    # 2. Transform the active dims only
    if bj.perm !== nothing
        p = bj.perm
        invp = bj.invperm
        D_tr = bj.D_tr
        
        # Static shape indexing via precomputed permutations
        x_perm = x[p, :]
        x_tr = x_perm[1:D_tr, :]
        
        bj_inner = bj.bijector_constructor(params)
        y_tr, ld_tr = transform_fn(bj_inner, x_tr)
        
        # 3. Reconstruct full y (non-mutating, Reactant-friendly)
        x_untr = x_perm[(D_tr + 1):end, :]
        y_perm = vcat(y_tr, x_untr)
        y = y_perm[invp, :]
    else
        # Fallback path for NSF (non-Reactant compatible due to boolean indexing)
        @views x_tr = x[m, :]
        bj_inner = bj.bijector_constructor(params)
        y_tr, ld_tr = transform_fn(bj_inner, x_tr)
        y = _reconstruct(m, x, y_tr)
    end
    
    # sum log-dets over the transformed dims
    return y, dsum(ld_tr; dims=(1,))
end

function _reconstruct(m::AbstractArray{Bool}, x::AbstractMatrix, y_tr::AbstractMatrix)
    y = similar(x)
    @views y[.!m, :] .= x[.!m, :]
    y[m, :] .= y_tr
    return y
end

function ChainRulesCore.rrule(::typeof(_reconstruct), m::AbstractArray{Bool}, x::AbstractMatrix, y_tr::AbstractMatrix)
    y = _reconstruct(m, x, y_tr)
    function _reconstruct_pullback(Δy)
        Δx = similar(x)
        @views Δx[.!m, :] .= Δy[.!m, :]
        Δx[m, :] .= 0
        @views Δy_tr = Δy[m, :]
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Δx, Δy_tr
    end
    return y, _reconstruct_pullback
end

function forward_and_log_det(bj::MaskedCoupling, x::AbstractArray)
    _apply_mask(bj, x, forward_and_log_det)
end

function inverse_and_log_det(bj::MaskedCoupling, y::AbstractArray)
    _apply_mask(bj, y, inverse_and_log_det)
end
