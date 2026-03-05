# src/nsf.jl
using Lux
using Bijectors
using Random
using LinearAlgebra

"""
    NSFSplineBijector(mask, params, K, tail_bound)

A Neural Spline Flow (NSF) coupling layer using Rational Quadratic Splines.
`mask` is a binary vector (1 for variables to transform, 0 for variables that condition).
`params` is the output of the conditioner for the transformed dimensions.
"""
# Bijectors and Layers for NSF
struct NSFSplineBijector
    params
    K::Int
    tail_bound::Float64
end

function forward_and_log_det(b::NSFSplineBijector, x::AbstractArray)
    # x is (D_tr, N)
    D_tr, N = size(x)
    K = b.K
    tail_bound = b.tail_bound
    params = b.params
    
    # Reshape params to (D_tr, 3K-1, N)
    params = reshape(params, D_tr, 3*K - 1, N)
    
    # Partition params
    w_unnorm = params[:, 1:K, :]
    h_unnorm = params[:, K+1:2*K, :]
    dv_unnorm = params[:, 2*K+1:end, :]
    
    # Flatten everything to call the spline function
    x_flat = vec(x)
    w_flat = reshape(permutedims(w_unnorm, (1, 3, 2)), D_tr * N, K)
    h_flat = reshape(permutedims(h_unnorm, (1, 3, 2)), D_tr * N, K)
    dv_flat = reshape(permutedims(dv_unnorm, (1, 3, 2)), D_tr * N, K-1)
    
    y_flat, lad_flat = unconstrained_rational_quadratic_spline(
        x_flat, w_flat, h_flat, dv_flat, eltype(x)(tail_bound)
    )
    
    y = reshape(y_flat, D_tr, N)
    lad = reshape(lad_flat, D_tr, N)
    
    return y, lad
end

function inverse_and_log_det(b::NSFSplineBijector, y::AbstractArray)
    # y is (D_tr, N)
    D_tr, N = size(y)
    K = b.K
    tail_bound = b.tail_bound
    params = b.params
    
    params = reshape(params, D_tr, 3*K - 1, N)
    w_unnorm = params[:, 1:K, :]
    h_unnorm = params[:, K+1:2*K, :]
    dv_unnorm = params[:, 2*K+1:end, :]
    
    y_flat = vec(y)
    w_flat = reshape(permutedims(w_unnorm, (1, 3, 2)), D_tr * N, K)
    h_flat = reshape(permutedims(h_unnorm, (1, 3, 2)), D_tr * N, K)
    dv_flat = reshape(permutedims(dv_unnorm, (1, 3, 2)), D_tr * N, K-1)
    
    x_flat, lad_flat = unconstrained_rational_quadratic_spline(
        y_flat, w_flat, h_flat, dv_flat, eltype(y)(tail_bound);
        inverse=true
    )
    
    x = reshape(x_flat, D_tr, N)
    lad = reshape(lad_flat, D_tr, N)
    
    return x, lad
end

function NSFCouplingBijector_from_flat(params, K, tail_bound)
    return NSFSplineBijector(params, K, tail_bound)
end

"""
    NeuralSplineFlow(; n_transforms, dist_dims, hidden_dims, n_layers, K=8, tail_bound=3.0, activation=gelu)

Neural Spline Flow (NSF) with rational quadratic coupling layers.
"""
@concrete struct NeuralSplineFlow <: Lux.AbstractLuxContainerLayer{(:conditioners,)}
    conditioners
    mask_list          :: Vector{BitVector}
    dist_dims          :: Int
    n_transforms       :: Int
    hidden_layer_sizes :: Vector{Int}
    K                  :: Int
    tail_bound         :: Float64
end

function NeuralSplineFlow(; n_transforms::Int, dist_dims::Int,
                             hidden_layer_sizes::Vector{Int}, K=8, tail_bound=3.0, activation=gelu)
    D = dist_dims
    
    # Pre-generate masks to know out_dims per layer
    mask_list = [BitVector(collect(1:D) .% 2 .== i % 2)
                 for i in 1:n_transforms]
    
    mlps = []
    for i in 1:n_transforms
        m = mask_list[i]
        D_tr = sum(m)
        out_dims = D_tr * (3*K - 1)
        push!(mlps, MLP(D, hidden_layer_sizes, out_dims; activation))
    end
    
    keys_ = ntuple(i -> Symbol(:conditioners_, i), n_transforms)
    conditioners = NamedTuple{keys_}(Tuple(mlps))
    return NeuralSplineFlow(conditioners, mask_list, D, n_transforms, hidden_layer_sizes, K, Float64(tail_bound))
end

function Lux.initialstates(rng::AbstractRNG, m::NeuralSplineFlow)
    return (; mask_list=m.mask_list, conditioners=Lux.initialstates(rng, m.conditioners))
end


# Helper to build the bijector from flat parameters
struct NSFSplineConstructor
    mask
    K::Int
    tail_bound::Float64
end

function (c::NSFSplineConstructor)(params)
    return NSFSplineBijector(c.mask, params, c.K, c.tail_bound)
end
