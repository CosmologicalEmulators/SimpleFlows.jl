# src/generic_ops.jl

"""
    log_prob(model, ps, st, x) -> Vector

Compute per-sample log-probability of `x` (shape: dist_dims × batch) under the flow.
Pure functional — no mutations, safe for Zygote.
Supports RealNVP, NeuralSplineFlow, and MaskedAutoregressiveFlow.
"""
function log_prob(model::Union{RealNVP, NeuralSplineFlow, MaskedAutoregressiveFlow}, ps, st, x::AbstractMatrix)
    lp = nothing
    for i in model.n_transforms:-1:1
        ks = model isa MaskedAutoregressiveFlow ? keys(model.mades) : keys(model.conditioners)
        k = ks[i]
        
        if model isa MaskedAutoregressiveFlow
            bj = MAFBijector(model.mades[k], ps.mades[k], st.mades[k])
            # Inverse is x -> u, which is what we need for log_prob
            x, ld = Bijectors.with_logabsdet_jacobian(bj, x)
        else
            mask = st.mask_list[i]
            cond_fn = let m = model.conditioners[k], p = ps.conditioners[k],
                          s = st.conditioners[k]
                x_cond -> Lux.apply(m, x_cond, p, s)[1]
            end
            
            bj = if model isa RealNVP
                MaskedCoupling(mask, cond_fn, AffineBijector)
            else
                MaskedCoupling(mask, cond_fn, p -> NSFCouplingBijector_from_flat(p, model.K, model.tail_bound))
            end
            x, ld = inverse_and_log_det(bj, x)
        end
        
        lp = isnothing(lp) ? ld : lp .+ ld
        
        # Apply ReversePermute between MAF blocks
        if model isa MaskedAutoregressiveFlow && i > 1
             x = x[end:-1:1, :]
        end
    end
    base_lp = dsum(gaussian_logpdf.(x); dims=(1,))
    return isnothing(lp) ? base_lp : lp .+ base_lp
end

"""
    draw_samples(rng, T, model, ps, st, n_samples) -> Matrix

Sample from the flow by pushing Gaussian noise through the forward transforms.
Supports RealNVP, NeuralSplineFlow, and MaskedAutoregressiveFlow.
"""
function draw_samples(rng::AbstractRNG, ::Type{T}, model::Union{RealNVP, NeuralSplineFlow, MaskedAutoregressiveFlow},
                      ps, st, n_samples::Int) where T
    x = randn(rng, T, model.dist_dims, n_samples)
    for i in 1:(model.n_transforms)
        ks = model isa MaskedAutoregressiveFlow ? keys(model.mades) : keys(model.conditioners)
        k = ks[i]
        
        if model isa MaskedAutoregressiveFlow
            bj = MAFBijector(model.mades[k], ps.mades[k], st.mades[k])
            x, _ = forward_and_log_det(bj, x)
        else
            mask = st.mask_list[i]
            cond_fn = let m = model.conditioners[k], p = ps.conditioners[k],
                          s = st.conditioners[k]
                x_cond -> Lux.apply(m, x_cond, p, s)[1]
            end
            
            bj = if model isa RealNVP
                MaskedCoupling(mask, cond_fn, AffineBijector)
            else
                MaskedCoupling(mask, cond_fn, p -> NSFCouplingBijector_from_flat(p, model.K, model.tail_bound))
            end
            
            x, _ = forward_and_log_det(bj, x)
        end
        
        # Apply ReversePermute between MAF blocks
        if model isa MaskedAutoregressiveFlow && i < model.n_transforms
             x = x[end:-1:1, :]
        end
    end
    return x
end
