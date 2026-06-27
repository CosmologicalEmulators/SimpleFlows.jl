# ── Utility ──────────────────────────────────────────────────────────────────

"""Sum and drop the given dimensions (batch-safe reduction)."""
dsum(x; dims) = dropdims(sum(x; dims=dims); dims=dims)

"""Elementwise log-pdf of a standard normal."""
gaussian_logpdf(x) = -oftype(x, 0.5) * (log(oftype(x, 2π)) + x^2)

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
    MaskedCoupling(mask, conditioner, bijector_constructor)

Coupling layer using a binary mask. Unmasked dimensions condition the bijector;
masked dimensions are transformed.
"""
@concrete struct MaskedCoupling
    mask
    conditioner
    bijector_constructor
end

function _apply_mask(bj::MaskedCoupling, x::AbstractMatrix, transform_fn)
    D, N = size(x)
    m = bj.mask
    
    # 1. Conditioning
    x_cond = x .* .!m
    params = bj.conditioner(x_cond)
    
    # 2. Transform the active dims only
    x_tr = x[m, :]
    bj_inner = bj.bijector_constructor(params)
    y_tr, ld_tr = transform_fn(bj_inner, x_tr)
    
    # 3. Reconstruct full y
    y = _reconstruct(m, x, y_tr)
    
    # sum log-dets over the transformed dims
    return y, dsum(ld_tr; dims=(1,))
end

function _reconstruct(m::AbstractArray{Bool}, x::AbstractMatrix, y_tr::AbstractMatrix)
    y = similar(x)
    y[.!m, :] .= x[.!m, :]
    y[m, :] .= y_tr
    return y
end

function ChainRulesCore.rrule(::typeof(_reconstruct), m::AbstractArray{Bool}, x::AbstractMatrix, y_tr::AbstractMatrix)
    y = _reconstruct(m, x, y_tr)
    function _reconstruct_pullback(Δy)
        Δx = similar(x)
        Δx[.!m, :] .= Δy[.!m, :]
        Δx[m, :] .= 0
        Δy_tr = Δy[m, :]
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
