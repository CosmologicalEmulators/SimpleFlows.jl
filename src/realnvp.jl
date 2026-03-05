# ── MLP helper ───────────────────────────────────────────────────────────────

"""Build an MLP conditioner; `hidden_layer_sizes` is a vector of per-layer widths."""
function MLP(in_dims::Int, hidden_layer_sizes::Vector{Int}, out_dims::Int;
             activation=gelu)
    widths = [in_dims; hidden_layer_sizes; out_dims]
    layers = [Dense(widths[i] => widths[i+1],
                    i < length(widths) - 1 ? activation : identity)
              for i in 1:(length(widths) - 1)]
    return Chain(layers...)
end

# ── RealNVP ───────────────────────────────────────────────────────────────────

"""
    RealNVP(; n_transforms, dist_dims, hidden_dims, n_layers, activation=gelu)

Real-valued Non-Volume Preserving (RealNVP) normalizing flow.
Each coupling layer is conditioned by an MLP producing affine parameters.
Masks alternate between even/odd dimensions.
"""
@concrete struct RealNVP <: AbstractLuxContainerLayer{(:conditioners,)}
    conditioners
    dist_dims          :: Int
    n_transforms       :: Int
    hidden_layer_sizes :: Vector{Int}
end

function RealNVP(; n_transforms::Int, dist_dims::Int,
                   hidden_layer_sizes::Vector{Int}, activation=gelu)
    mlps = [MLP(dist_dims, hidden_layer_sizes, 2 * dist_dims; activation)
            for _ in 1:n_transforms]
    keys_ = ntuple(i -> Symbol(:conditioners_, i), n_transforms)
    conditioners = NamedTuple{keys_}(Tuple(mlps))
    return RealNVP(conditioners, dist_dims, n_transforms, hidden_layer_sizes)
end

function Lux.initialstates(rng::AbstractRNG, m::RealNVP)
    mask_list = [Bool.(collect(1:(m.dist_dims)) .% 2 .== i % 2)
                 for i in 1:(m.n_transforms)]
    return (; mask_list, conditioners=Lux.initialstates(rng, m.conditioners))
end

# ── Pure-functional log-prob and sample ──────────────────────────────────────

"""
    log_prob(model, ps, st, x) -> Vector

Compute per-sample log-probability of `x` (shape: dist_dims × batch) under the flow.
Pure functional — no mutations, safe for Zygote.
"""
function log_prob(model::RealNVP, ps, st, x::AbstractMatrix)
    lp = nothing
    for i in model.n_transforms:-1:1
        k = keys(model.conditioners)[i]
        mask = st.mask_list[i]
        cond_fn = let m = model.conditioners[k], p = ps.conditioners[k],
                      s = st.conditioners[k]
            x_cond -> Lux.apply(m, x_cond, p, s)[1]
        end
        bj = MaskedCoupling(mask, cond_fn, AffineBijector)
        x, ld = inverse_and_log_det(bj, x)
        lp = isnothing(lp) ? ld : lp .+ ld
    end
    base_lp = dsum(gaussian_logpdf.(x); dims=(1,))
    return isnothing(lp) ? base_lp : lp .+ base_lp
end

"""
    draw_samples(rng, T, model, ps, st, n_samples) -> Matrix

Sample from the flow by pushing Gaussian noise through the forward transforms.
"""
function draw_samples(rng::AbstractRNG, ::Type{T}, model::RealNVP,
                      ps, st, n_samples::Int) where T
    x = randn(rng, T, model.dist_dims, n_samples)
    for i in 1:(model.n_transforms)
        k = keys(model.conditioners)[i]
        cond_fn = let m = model.conditioners[k], p = ps.conditioners[k],
                      s = st.conditioners[k]
            x_cond -> Lux.apply(m, x_cond, p, s)[1]
        end
        bj = MaskedCoupling(st.mask_list[i], cond_fn, AffineBijector)
        x, _ = forward_and_log_det(bj, x)
    end
    return x
end
