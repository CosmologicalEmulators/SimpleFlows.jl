# MODIFIED: generic_ops.jl — type-stable log_prob and fused gaussian_logpdf_sum

"""
    log_prob(model, ps, st, x) -> Vector

Compute per-sample log-probability of `x` (shape: dist_dims × batch) under the flow.
Pure functional — no mutations, safe for Zygote.
Supports RealNVP, NeuralSplineFlow, and MaskedAutoregressiveFlow.
"""
function log_prob(model::Union{RealNVP, NeuralSplineFlow, MaskedAutoregressiveFlow}, ps, st, x::AbstractMatrix)
    n = model.n_transforms
    
    # Base case for zero transforms (though unlikely in practice)
    if n == 0
        return gaussian_logpdf_sum(x)
    end

    # First iteration to infer correct type and size of the log-determinant (lp)
    x, lp = _single_inverse(model, ps, st, x, n)
    
    # Loop over remaining transforms
    for i in (n - 1):-1:1
        x, ld = _single_inverse(model, ps, st, x, i)
        lp = lp .+ ld
    end
    
    # Gaussian base log-probability (fused reduction)
    base_lp = gaussian_logpdf_sum(x)
    return lp .+ base_lp
end

"""
    _single_inverse(model, ps, st, x, i) -> (x_next, log_det)

Helper to apply the i-th inverse transform of the model.
"""
function _single_inverse(model::Union{RealNVP, NeuralSplineFlow, MaskedAutoregressiveFlow}, ps, st, x::AbstractMatrix, i::Int)
    ks = model isa MaskedAutoregressiveFlow ? keys(model.mades) : keys(model.conditioners)
    k = ks[i]
    
    if model isa MaskedAutoregressiveFlow
        bj = MAFBijector(model.mades[k], ps.mades[k], st.mades[k])
        # Density estimation: u = (x - m) * exp(-alpha). Fast O(1).
        x_next, ld = Bijectors.with_logabsdet_jacobian(bj, x)
    else
        mask = model isa RealNVP ? model.mask_list[i] : st.mask_list[i]

        cond_fn = let m_layer = model.conditioners[k], p = ps.conditioners[k],
                      s = st.conditioners[k]
            x_cond -> Lux.apply(m_layer, x_cond, p, s)[1]
        end
        
        bj = if model isa RealNVP
            p    = model.perm_list[i]
            invp = model.invperm_list[i]
            D_tr = model.D_tr_list[i]
            MaskedCoupling(mask, cond_fn, AffineBijector, p, invp, D_tr)
        else
            MaskedCoupling(mask, cond_fn, p -> NSFCouplingBijector_from_flat(p, model.K, model.tail_bound))
        end
        x_next, ld = inverse_and_log_det(bj, x)
    end
    
    # Apply ReversePermute between MAF blocks (standard practice for MAF)
    if model isa MaskedAutoregressiveFlow && i > 1
         x_next = x_next[end:-1:1, :]
    end
    
    return x_next, ld
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
            mask = model isa RealNVP ? model.mask_list[i] : st.mask_list[i]

            cond_fn = let m_layer = model.conditioners[k], p = ps.conditioners[k],
                          s = st.conditioners[k]
                x_cond -> Lux.apply(m_layer, x_cond, p, s)[1]
            end
            
            bj = if model isa RealNVP
                p    = model.perm_list[i]
                invp = model.invperm_list[i]
                D_tr = model.D_tr_list[i]
                MaskedCoupling(mask, cond_fn, AffineBijector, p, invp, D_tr)
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
