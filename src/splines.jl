# src/splines.jl
using NNlib
using ChainRulesCore
using ForwardDiff

# Helper to find bins. Zygote ignores gradients for indices.
function compute_bin_idx(cum_arrays::AbstractMatrix{T}, inputs::AbstractVector{T}, K::Int) where {T<:Real}
    M = length(inputs)
    bin_idx = zeros(Int, M)
    for i in 1:M
        # Handle ForwardDiff values properly if needed, but searchsortedlast supports Duals natively
        idx = searchsortedlast(@view(cum_arrays[i, :]), inputs[i])
        bin_idx[i] = clamp(idx, 1, K)
    end
    return bin_idx
end
ChainRulesCore.@non_differentiable compute_bin_idx(Any...)

"""
    unconstrained_rational_quadratic_spline(inputs, widths, heights, derivs; kwargs...)

Inputs:
- `inputs`: Vector of length M.
- `unnormalized_widths`: Matrix of shape (M, K).
- `unnormalized_heights`: Matrix of shape (M, K).
- `unnormalized_derivatives`: Matrix of shape (M, K-1).

All inputs are flattened. Returns `(outputs, logabsdet)`.
"""
function unconstrained_rational_quadratic_spline(
    inputs::AbstractVector{T_in},
    unnormalized_widths::AbstractMatrix{T_w},
    unnormalized_heights::AbstractMatrix{T_h},
    unnormalized_derivatives::AbstractMatrix{T_dv},
    tail_bound=3.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3;
    inverse::Bool=false
) where {T_in<:Real, T_w<:Real, T_h<:Real, T_dv<:Real}
    # Operations will be performed in the promoted type
    T = promote_type(T_in, T_w, T_h, T_dv)
    
    # Cast everything to the consistency type T
    inputs = T.(inputs)
    unnormalized_widths = T.(unnormalized_widths)
    unnormalized_heights = T.(unnormalized_heights)
    unnormalized_derivatives = T.(unnormalized_derivatives)
    
    # Convert keyword arguments to T
    tail_bound = T(tail_bound)
    min_bin_width = T(min_bin_width)
    min_bin_height = T(min_bin_height)
    min_derivative = T(min_derivative)
    
    M, K = size(unnormalized_widths)
    
    # 1. Pad derivatives with constant for linear tails
    constant = T(log(exp(1 - min_derivative) - 1))
    pad_c = fill(constant, M)
    unnorm_derivs = hcat(pad_c, unnormalized_derivatives, pad_c) # (M, K+1)
    
    # 2. Extract valid interior regions (using Mask to avoid mutating variables for Zygote)
    inside_mask = (inputs .>= -tail_bound) .& (inputs .<= tail_bound)
    
    # We will compute the spline for ALL points, but only conditionally select the output
    # This avoids Zygote mutating array issues, relying on `ifelse`.
    # To avoid NaNs or domain errors, we clamp the inputs outside the tail bound.
    clamped_inputs = clamp.(inputs, -tail_bound, tail_bound)
    
    # 3. Compute normalize bin parameters
    widths_raw = NNlib.softmax(unnormalized_widths; dims=2)
    widths = min_bin_width .+ (1 - min_bin_width * K) .* widths_raw
    
    cumwidths_raw = cumsum(widths; dims=2)
    # Scale to (0, 1) then to (-tail_bound, tail_bound)
    # We use hcat to ensure exact boundaries
    interior_cumwidths = (2 * tail_bound) .* (@view cumwidths_raw[:, 1:end-1]) .- tail_bound
    cumwidths = hcat(fill(-tail_bound, M), interior_cumwidths, fill(tail_bound, M))
    widths = cumwidths[:, 2:end] .- cumwidths[:, 1:end-1]
    
    derivatives = min_derivative .+ NNlib.softplus.(unnorm_derivs)
    
    heights_raw = NNlib.softmax(unnormalized_heights; dims=2)
    heights = min_bin_height .+ (1 - min_bin_height * K) .* heights_raw
    
    cumheights_raw = cumsum(heights; dims=2)
    interior_cumheights = (2 * tail_bound) .* (@view cumheights_raw[:, 1:end-1]) .- tail_bound
    cumheights = hcat(fill(-tail_bound, M), interior_cumheights, fill(tail_bound, M))
    heights = cumheights[:, 2:end] .- cumheights[:, 1:end-1]
    
    # 4. Find the appropriate bin
    if inverse
        bin_idx = compute_bin_idx(cumheights, clamped_inputs, K)
    else
        bin_idx = compute_bin_idx(cumwidths, clamped_inputs, K)
    end
    
    # 5. Gather bin specific parameters
    # Zygote friendly gather using linear indexing
    # cumwidths is (M, K+1)
    # widths is (M, K)
    linear_indices_k_plus_1 = (bin_idx .- 1) .* M .+ (1:M)
    linear_indices_k = (bin_idx .- 1) .* M .+ (1:M)
    
    input_cumwidths = cumwidths[linear_indices_k_plus_1]
    input_bin_widths = widths[linear_indices_k]
    
    input_cumheights = cumheights[linear_indices_k_plus_1]
    input_heights = heights[linear_indices_k]
    
    delta = heights ./ widths
    input_delta = delta[linear_indices_k]
    
    input_derivatives = derivatives[linear_indices_k_plus_1]
    input_derivatives_plus_one = derivatives[linear_indices_k_plus_1 .+ M]
    
    # 6. Evaluate spline equations
    if inverse
        a = (clamped_inputs .- input_cumheights) .* (input_derivatives .+ input_derivatives_plus_one .- 2 .* input_delta) .+ input_heights .* (input_delta .- input_derivatives)
        b = input_heights .* input_derivatives .- (clamped_inputs .- input_cumheights) .* (input_derivatives .+ input_derivatives_plus_one .- 2 .* input_delta)
        c = -input_delta .* (clamped_inputs .- input_cumheights)
        
        discriminant = b.^2 .- 4 .* a .* c
        # Numerical stability: splines are monotonic so discriminant >= 0.
        # We use abs and a tiny epsilon for square root gradients.
        root = (2 .* c) ./ (-b .- sqrt.(abs.(discriminant) .+ T(1e-12)))
        
        rq_outputs = root .* input_bin_widths .+ input_cumwidths
        
        theta_one_minus_theta = root .* (1 .- root)
        denominator = input_delta .+ ((input_derivatives .+ input_derivatives_plus_one .- 2 .* input_delta) .* theta_one_minus_theta)
        derivative_numerator = (input_delta.^2) .* (input_derivatives_plus_one .* root.^2 .+ 2 .* input_delta .* theta_one_minus_theta .+ input_derivatives .* (1 .- root).^2)
        
        rq_logabsdet = log.(abs.(derivative_numerator) .+ T(1e-12)) .- 2 .* log.(abs.(denominator) .+ T(1e-12))
        rq_logabsdet = -rq_logabsdet
    else
        theta = (clamped_inputs .- input_cumwidths) ./ input_bin_widths
        theta_one_minus_theta = theta .* (1 .- theta)
        
        numerator = input_heights .* (input_delta .* theta.^2 .+ input_derivatives .* theta_one_minus_theta)
        denominator = input_delta .+ ((input_derivatives .+ input_derivatives_plus_one .- 2 .* input_delta) .* theta_one_minus_theta)
        rq_outputs = input_cumheights .+ numerator ./ denominator
        
        derivative_numerator = (input_delta.^2) .* (input_derivatives_plus_one .* theta.^2 .+ 2 .* input_delta .* theta_one_minus_theta .+ input_derivatives .* (1 .- theta).^2)
        rq_logabsdet = log.(abs.(derivative_numerator) .+ T(1e-12)) .- 2 .* log.(abs.(denominator) .+ T(1e-12))
    end
    
    # 7. Apply identity outside bounds
    outputs = ifelse.(inside_mask, rq_outputs, inputs)
    logabsdet = ifelse.(inside_mask, rq_logabsdet, zero(T))
    
    return outputs, logabsdet
end
