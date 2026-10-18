module ReactantExt

using SimpleFlows
using Reactant
using Lux, Optimisers   # parent dependencies; Lux's own Reactant extension needs Enzyme loaded too

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

# ── 3b. Whitening forward substitution ────────────────────────────────────────

# `LowerTriangular(L) \ y` lowers to `stablehlo.triangular_solve`. With
# Reactant v0.2.295 / Enzyme v0.13.216 (Julia 1.12.6) the forward pass compiles but the Enzyme reverse
# pass fails ("could not compute the adjoint for this operation
# stablehlo.triangular_solve"); see test/reactant_trisolve_repro.jl and
# STAGE3_REPORT.md for the reproduction. Instead solve `L z = y` by forward substitution over
# the (static) dimension: still a triangular solve, no explicit inverse, built
# only from statically-indexed slices, matmuls and broadcasts, hence
# differentiable and free of scalar indexing.
function _forward_substitute(L::DeviceMat, y::AbstractMatrix)
    d = size(L, 1)
    z = y[1:1, :] ./ L[1:1, 1:1]
    for i in 2:d
        zi = (y[i:i, :] .- L[i:i, 1:(i - 1)] * z) ./ L[i:i, i:i]
        z = vcat(z, zi)
    end
    return z
end

SimpleFlows._lower_solve(L::DeviceMat, y::AbstractMatrix) = _forward_substitute(L, y)

# Plain scalar fields are static during tracing. Derive density corrections from
# the same dynamic arrays used by normalize, not the cached host log_jac field.
SimpleFlows._normalizer_log_jac(n::SimpleFlows.MinMaxNormalizer{T, A}) where {T<:Real, A<:DeviceVec} =
    -sum(log.(n.x_max .- n.x_min))

SimpleFlows._normalizer_log_jac(n::SimpleFlows.WhiteningNormalizer{T, V, M}) where {T<:Real, V, M<:DeviceMat} =
    SimpleFlows._whitening_log_jac(n.chol)

# ── 4. Reactant Device Conversion (to_reactant) ──────────────────────────────

function SimpleFlows.to_reactant(norm::SimpleFlows.MinMaxNormalizer)
    return SimpleFlows.MinMaxNormalizer(
        Reactant.to_rarray(norm.x_min),
        Reactant.to_rarray(norm.x_max),
        Reactant.to_rarray(norm.log_jac)
    )
end

function SimpleFlows.to_reactant(norm::SimpleFlows.WhiteningNormalizer)
    return SimpleFlows.WhiteningNormalizer(
        Reactant.to_rarray(norm.mean),
        Reactant.to_rarray(norm.chol),
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
        reactant_norm,
        flow.parameter_names
    )
end

# ── 5. Reactant training engine (backend=:reactant) ──────────────────────────
#
# Uses Lux's native `Training.TrainState` + `AutoEnzyme()` route: gradients come
# from `Lux.Training.compute_gradients` (compiled once, cached in the returned
# TrainState) and the update from `Lux.Training.apply_gradients!`. The model
# configuration and the coupling masks stay static: they live in the objective
# struct (`st`), while the TrainState carries an empty dynamic state. The whole
# (normalised) training set is device resident; every step passes only a padded
# index vector and 0/1 weights, and the batch is gathered *inside* the compiled
# objective. Every batch has the same padded size, so the exact final partial
# batch needs no second compilation; the padded duplicates carry weight 0, so the
# loss is exactly the mean over the real samples.
#
# Validity: after each gradient computation a compiled device-side flag
# (loss and every gradient leaf finite) is synchronised as one scalar *before*
# the update is applied, and the parameters are checked the same way after the
# last update of every epoch. This costs one scalar device->host sync per step
# and is deliberately conservative.

struct ReactantObjective{S}
    st::S                 # static layer state (coupling masks); never traced
end

function (o::ReactantObjective)(model, ps, st, data)
    lp = SimpleFlows.log_prob(model, ps, o.st, data.x[:, data.idx])
    loss = -sum(lp .* data.w) / sum(data.w)
    return loss, st, (;)
end

_eval_lp(o::ReactantObjective, model, ps, data) =
    SimpleFlows.log_prob(model, ps, o.st, data.x[:, data.idx])

_dev_finite(x::AbstractArray) = all(isfinite, x)
_dev_finite(x::Number) = isfinite(x)
_dev_finite(x::Reactant.TracedRNumber) = isfinite(x)
_dev_finite(x::Union{NamedTuple, Tuple}) = mapreduce(_dev_finite, &, values(x); init=true)
_dev_finite(::Nothing) = true

mutable struct ReactantEngine{T, O, X, J}
    obj        :: O
    ts                      # Lux TrainState; its type changes once the compile cache is attached
    xtrain     :: X
    xval       :: Union{Nothing, X}
    n_train    :: Int
    n_val      :: Int
    batch      :: Int
    log_jac    :: J
    compiled   :: Dict{Symbol, Any}
end

function SimpleFlows._make_reactant_engine(flow::SimpleFlows.FlowDistribution{T}, data_norm, val_norm, normalizer;
                                           lr, batch_size, opt) where {T}
    Lux.is_extension_loaded(Val(:Reactant)) || throw(ArgumentError(
        "backend=:reactant needs Lux's Reactant extension: run `using Reactant, Enzyme` before training."))
    actual_opt = SimpleFlows._default_optimizer(T, lr, opt)
    ps_dev = Reactant.to_rarray(deepcopy(flow.ps))
    ts = Lux.Training.TrainState(flow.model, ps_dev, (;), actual_opt)
    xtrain = Reactant.to_rarray(data_norm)
    xval = isnothing(val_norm) ? nothing : Reactant.to_rarray(val_norm)
    n_train = size(data_norm, 2)
    return ReactantEngine{T, typeof(ReactantObjective(flow.st)), typeof(xtrain), typeof(normalizer.log_jac)}(
        ReactantObjective(flow.st), ts, xtrain, xval, n_train,
        isnothing(val_norm) ? 0 : size(val_norm, 2), min(batch_size, n_train),
        normalizer.log_jac, Dict{Symbol, Any}())
end

# Fixed-size padded batch: padding repeats a real sample of the batch with weight 0.
function _padded_batch(e::ReactantEngine{T}, x, idx) where {T}
    n = length(idx)
    pidx = fill(first(idx), e.batch)
    pidx[1:n] .= idx
    w = zeros(T, e.batch); w[1:n] .= one(T)
    return (; x, idx=Reactant.to_rarray(pidx), w=Reactant.to_rarray(w))
end

function _device_ok!(e::ReactantEngine, key::Symbol, tree)
    f = get!(e.compiled, key) do
        Reactant.@compile sync=true _dev_finite(tree)
    end
    return Bool(f(tree))
end

function SimpleFlows._engine_step!(e::ReactantEngine, idx, epoch, batch_number)
    data = _padded_batch(e, e.xtrain, idx)
    grads, loss, _, ts = Lux.Training.compute_gradients(AutoEnzyme(), e.obj, data, e.ts)
    e.ts = ts
    _device_ok!(e, :grads, (loss, grads)) ||
        error("Non-finite training loss or gradient at epoch $epoch, batch $batch_number; stopping before the optimizer update.")
    e.ts = Lux.Training.apply_gradients!(e.ts, grads)
    return nothing
end

function SimpleFlows._engine_epoch_done!(e::ReactantEngine, epoch)
    _device_ok!(e, :params, e.ts.parameters) ||
        error("Optimizer update produced non-finite parameters in epoch $epoch (check the learning rate / optimizer).")
    return nothing
end

function SimpleFlows._engine_nll(e::ReactantEngine, which::Symbol)
    x, n = which === :train ? (e.xtrain, e.n_train) : (e.xval, e.n_val)
    ps = e.ts.parameters
    total = 0.0
    for start in 1:e.batch:n
        cols = start:min(start + e.batch - 1, n)
        data = _padded_batch(e, x, collect(cols))
        f = get!(e.compiled, which) do
            Reactant.@compile sync=true _eval_lp(e.obj, e.ts.model, ps, data)
        end
        lp = Array(f(e.obj, e.ts.model, ps, data))
        total += sum(Float64, @view lp[1:length(cols)]) + length(cols) * Float64(e.log_jac)
    end
    nll = -total / n
    isfinite(nll) || error("Non-finite negative log-likelihood ($nll) while evaluating the flow at fixed weights.")
    return nll
end

# Weights leave the device only here: an independent host copy.
SimpleFlows._engine_snapshot(e::ReactantEngine) = deepcopy(Lux.cpu_device()(e.ts.parameters))

end
