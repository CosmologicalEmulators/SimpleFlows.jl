# src/nsf.jl
using Lux
using Bijectors
using Random
using LinearAlgebra

"""
    NSFCouplingLayer(mask, conditioner; K=8, tail_bound=3.0)

A Neural Spline Flow (NSF) coupling layer using Rational Quadratic Splines.
`mask` is a binary vector (1 for variables to transform, 0 for variables that condition).
`conditioner` is a Lux network that takes variables with 0 and returns spline parameters for variables with 1.
"""
# Bijectors and Layers for NSF
struct NSFSplineBijector
    mask
    params
    K::Int
    tail_bound::Float64
end

function forward_and_log_det(b::NSFSplineBijector, x::AbstractArray)
    # x is (D, N)
    D, N = size(x)
    mask = b.mask
    K = b.K
    tail_bound = b.tail_bound
    params = b.params
    
    # x_tr: variables to be transformed
    x_tr = x[mask, :]
    D_tr = size(x_tr, 1)
    
    # Reshape params to (D_tr, 3K-1, N)
    params = reshape(params, D_tr, 3*K - 1, N)
    
    # Partition params
    w_unnorm = params[:, 1:K, :]
    h_unnorm = params[:, K+1:2*K, :]
    dv_unnorm = params[:, 2*K+1:end, :]
    
    # Flatten everything to call the spline function
    x_tr_flat = vec(x_tr)
    w_flat = reshape(permutedims(w_unnorm, (1, 3, 2)), D_tr * N, K)
    h_flat = reshape(permutedims(h_unnorm, (1, 3, 2)), D_tr * N, K)
    dv_flat = reshape(permutedims(dv_unnorm, (1, 3, 2)), D_tr * N, K-1)
    
    y_tr_flat, lad_flat = unconstrained_rational_quadratic_spline(
        x_tr_flat, w_flat, h_flat, dv_flat, eltype(x)(tail_bound)
    )
    
    y_tr = reshape(y_tr_flat, D_tr, N)
    lad_tr = reshape(lad_flat, D_tr, N)
    
    # We yield the full transformed y (only for masked dims)
    # The MaskedCoupling logic will handle the identity part.
    # But wait, MaskedCoupling expects the bijector to return an array of same size as x?
    # No, MaskedCoupling: y, log_det = transform_fn(params)
    # Then y = ifelse.(bj.mask, y, x)
    # So y MUST have same size as x.
    
    # Reconstruction using comprehension (Zygote friendly)
    # We need to find the index into y_tr for each masked dimension.
    # tr_indices[i] will be the row index in y_tr if mask[i] is true.
    tr_indices = cumsum(mask)
    
    y = vcat([mask[i] ? y_tr[tr_indices[i]:tr_indices[i], :] : x[i:i, :] for i in 1:D]...)
    log_det = vcat([mask[i] ? lad_tr[tr_indices[i]:tr_indices[i], :] : fill(zero(eltype(x)), 1, N) for i in 1:D]...)
    
    return y, log_det
    
    return y, log_det
end

function inverse_and_log_det(b::NSFSplineBijector, y::AbstractArray)
    # y is (D, N)
    D, N = size(y)
    mask = b.mask
    K = b.K
    tail_bound = b.tail_bound
    params = b.params
    
    y_tr = y[mask, :]
    D_tr = size(y_tr, 1)
    
    params = reshape(params, D_tr, 3*K - 1, N)
    w_unnorm = params[:, 1:K, :]
    h_unnorm = params[:, K+1:2*K, :]
    dv_unnorm = params[:, 2*K+1:end, :]
    
    y_tr_flat = vec(y_tr)
    w_flat = reshape(permutedims(w_unnorm, (1, 3, 2)), D_tr * N, K)
    h_flat = reshape(permutedims(h_unnorm, (1, 3, 2)), D_tr * N, K)
    dv_flat = reshape(permutedims(dv_unnorm, (1, 3, 2)), D_tr * N, K-1)
    
    x_tr_flat, lad_flat = unconstrained_rational_quadratic_spline(
        y_tr_flat, w_flat, h_flat, dv_flat, eltype(y)(tail_bound);
        inverse=true
    )
    
    x_tr = reshape(x_tr_flat, D_tr, N)
    lad_tr = reshape(lad_flat, D_tr, N)
    
    tr_indices = cumsum(mask)
    
    x = vcat([mask[i] ? x_tr[tr_indices[i]:tr_indices[i], :] : y[i:i, :] for i in 1:D]...)
    log_det = vcat([mask[i] ? lad_tr[tr_indices[i]:tr_indices[i], :] : fill(zero(eltype(y)), 1, N) for i in 1:D]...)
    
    return x, log_det
    
    return x, log_det
end

function NSFCouplingBijector_from_flat(params, mask, K, tail_bound)
    return NSFSplineBijector(mask, params, K, tail_bound)
end

"""
    NeuralSplineFlow(; n_transforms, dist_dims, hidden_dims, n_layers, K=8, tail_bound=3.0, activation=gelu)

Neural Spline Flow (NSF) with rational quadratic coupling layers.
"""
@concrete struct NeuralSplineFlow <: Lux.AbstractLuxContainerLayer{(:conditioners,)}
    conditioners
    dist_dims          :: Int
    n_transforms       :: Int
    hidden_layer_sizes :: Vector{Int}
    K                  :: Int
    tail_bound         :: Float64
end

function NeuralSplineFlow(; n_transforms::Int, dist_dims::Int,
                             hidden_layer_sizes::Vector{Int}, K=8, tail_bound=3.0, activation=gelu)
    # Number of transformed dimensions in each layer (mask alternate half)
    D = dist_dims
    D_tr = D - (D ÷ 2) # Approximately half
    # Conditioner output size: D_tr * (3K - 1)
    out_dims = D_tr * (3*K - 1)
    
    mlps = [MLP(D, hidden_layer_sizes, out_dims; activation)
            for _ in 1:n_transforms]
    keys_ = ntuple(i -> Symbol(:conditioners_, i), n_transforms)
    conditioners = NamedTuple{keys_}(Tuple(mlps))
    return NeuralSplineFlow(conditioners, D, n_transforms, hidden_layer_sizes, K, Float64(tail_bound))
end

function Lux.initialstates(rng::AbstractRNG, m::NeuralSplineFlow)
    mask_list = [Bool.(collect(1:(m.dist_dims)) .% 2 .== i % 2)
                 for i in 1:(m.n_transforms)]
    return (; mask_list, conditioners=Lux.initialstates(rng, m.conditioners))
end

# Generic log_prob and draw_samples
function log_prob(model::Union{RealNVP, NeuralSplineFlow}, ps, st, x::AbstractMatrix)
    lp = nothing
    for i in model.n_transforms:-1:1
        k = keys(model.conditioners)[i]
        mask = st.mask_list[i]
        cond_fn = let m = model.conditioners[k], p = ps.conditioners[k],
                      s = st.conditioners[k]
            x_cond -> Lux.apply(m, x_cond, p, s)[1]
        end
        
        bj = if model isa RealNVP
            MaskedCoupling(mask, cond_fn, AffineBijector)
        else
            MaskedCoupling(mask, cond_fn, p -> NSFCouplingBijector_from_flat(p, mask, model.K, model.tail_bound))
        end
        
        x, ld = inverse_and_log_det(bj, x)
        lp = isnothing(lp) ? ld : lp .+ ld
    end
    base_lp = dsum(gaussian_logpdf.(x); dims=(1,))
    return isnothing(lp) ? base_lp : lp .+ base_lp
end

function draw_samples(rng::AbstractRNG, ::Type{T}, model::Union{RealNVP, NeuralSplineFlow},
                      ps, st, n_samples::Int) where T
    x = randn(rng, T, model.dist_dims, n_samples)
    for i in 1:(model.n_transforms)
        k = keys(model.conditioners)[i]
        mask = st.mask_list[i]
        cond_fn = let m = model.conditioners[k], p = ps.conditioners[k],
                      s = st.conditioners[k]
            x_cond -> Lux.apply(m, x_cond, p, s)[1]
        end
        
        bj = if model isa RealNVP
            MaskedCoupling(mask, cond_fn, AffineBijector)
        else
            MaskedCoupling(mask, cond_fn, p -> NSFCouplingBijector_from_flat(p, mask, model.K, model.tail_bound))
        end
        
        x, _ = forward_and_log_det(bj, x)
    end
    return x
end

# Helper to build the bijector from flat parameters
struct NSFSplineConstructor
    mask
    K::Int
    tail_bound::Float64
end

function (c::NSFSplineConstructor)(params)
    return NSFCouplingLayer(c.mask, params, c.K, c.tail_bound)
end

# Wait, I need to fix MaskedCoupling to work with NSFCouplingLayer
# or just use the logic in NSFCouplingLayer.
