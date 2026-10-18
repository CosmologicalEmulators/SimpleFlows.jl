"""
    train_flow!(flow, data; n_epochs, lr, batch_size, verbose, opt,
                normalization, refit_normalizer,
                validation_data, rng, eval_every, history, backend) -> FlowDistribution

Train `flow` by minimising the negative log-likelihood of the normalised data
with mini-batched Adam + Zygote. Returns `flow` (mutated in place).

# Arguments
- `flow::FlowDistribution`: the flow to train.
- `data::AbstractMatrix`: training samples in physical coordinates, shape `(n_dims, n_samples)`.

# Keyword Arguments
- `n_epochs=1000`: number of passes over the training data (`0` only evaluates).
- `lr=1f-3`: Adam learning rate. `opt`: custom Optimisers.jl rule instead. The
  optimizer state is created afresh on every call, so calling `train_flow!`
  repeatedly is **not** an exact optimizer resume (no moment carry-over).
- `batch_size=256`: mini-batch size (also used for the full-data evaluations).
- `verbose=true`: log, at every evaluation epoch (see `eval_every`), the full-dataset
  fixed-weight physical-coordinate train (and validation) NLL. This runs the same
  evaluation as `history`; there is no mini-batch loss logging. With the default
  `eval_every=1` that is one log line per epoch, so use a larger interval for long runs.
- `rng=Random.default_rng()`: RNG used to shuffle the batches; it is advanced in
  place. Same seed, data and initial flow give identical weights and history.

## Backend
- `backend=:zygote` (default): CPU training with Zygote.
- `backend=:reactant`: optional, needs `using Reactant, Enzyme` first (otherwise
  `ArgumentError`). Training runs through Lux's native `Training.TrainState` with
  `AutoEnzyme()`. Preprocessing, validation of inputs, batch order (same `rng`
  stream as `:zygote`), history, best-checkpoint selection, rollback and the CPU
  `FlowDistribution` return are shared with the Zygote path; the optimizer state
  and device weights are internal and only host copies of the final/best weights
  leave the device. The normalised training set lives on the device and each
  step sends only a padded index vector and 0/1 weights; the batch is gathered
  inside the compiled objective. Every batch is padded to one size with weight 0,
  so the exact final partial batch needs no extra compilation and the loss is the
  exact mean over the real samples. Finite loss/gradient flags are computed on
  the device and synchronised as one scalar per step before the update, and the
  weights are checked after the last update of every epoch. `opt` must be a rule
  supported on the device. The coupling masks stay static (not traced).

## Normalizer
- `normalization=nothing`: default. A flow without a normalizer gets a fitted
  `:minmax`; a flow that already has one **reuses it unchanged**, so repeated
  training does not move the coordinates the weights were trained in.
- `normalization=:minmax | :whiten`: explicit kind. If it matches the existing
  normalizer it is reused. If it conflicts, `refit_normalizer=true` is required.
- `refit_normalizer=false`: with `true` the normalizer is refitted on `data`
  (with the explicit kind, else the existing kind). This deliberately changes
  the coordinates in which the current weights operate, so the current density
  changes until retrained.
Training and validation data are always normalised with the normalizer fitted on
the **training** data only (or the reused one).

## Validation, history and best weights
- `validation_data=nothing`: held-out samples `(n_dims, n_val)`. When given, the
  validation NLL is evaluated at epoch 0, every `eval_every` epochs and at the
  final epoch, and at the end the weights with the **lowest validation NLL**
  (the initial weights are a candidate; ties keep the earliest) are restored as
  an independent deep copy. Without validation the final weights are kept.
- `eval_every=1`: positive evaluation interval in epochs.
- `history=nothing`: optional caller-provided `Vector` (e.g. `Any[]`) to which one
  `(; epoch, train_nll, validation_nll, best_epoch)` entry is appended per
  evaluation. NLLs are the full-dataset, fixed-weight **physical-coordinate**
  mean negative log-density (the normalizer's Jacobian included, equal to
  `-mean(logpdf(flow, data))`), computed in `Float64` and weighted per sample.
  `validation_nll` and `best_epoch` are `nothing` without validation. Epochs
  count from 0 within this call; the final entry is evaluated at the last
  trained weights even if the best earlier weights are then restored. Without
  `validation_data`, `history` and `verbose` no evaluation is done.

## Errors
Inputs are checked before anything is modified: wrong dimension, empty or
non-finite data (after conversion to the flow precision), non-positive
`n_epochs`/`batch_size`/`eval_every` (`n_epochs=0` is allowed), unknown
`normalization`, a conflicting normalizer without `refit_normalizer`, a failing
normalizer fit and an unusable `history` throw `ArgumentError`. A non-finite
loss or gradient leaf stops training with a descriptive `ErrorException`
before the optimizer update, and non-finite parameters produced by an update
are also an error (also for the very last update); bad gradients are never
skipped. An `lr` that is not finite and positive in the flow precision is an
`ArgumentError`. On any error the
flow's weights and normalizer are restored to their state at entry.
"""
function train_flow!(flow::FlowDistribution{T}, data::AbstractMatrix;
                     n_epochs::Int=1000,
                     lr::Union{Nothing, Real}=nothing,
                     batch_size::Int=256,
                     verbose::Bool=true,
                     opt=nothing,
                     normalization::Union{Nothing, Symbol}=nothing,
                     refit_normalizer::Bool=false,
                     validation_data::Union{Nothing, AbstractMatrix}=nothing,
                     rng::AbstractRNG=Random.default_rng(),
                     eval_every::Int=1,
                     history::Union{Nothing, AbstractVector}=nothing,
                     backend::Symbol=:zygote) where {T}
    # ── 1. Validate everything before touching the flow ──────────────────────
    backend in (:zygote, :reactant) ||
        throw(ArgumentError("Unknown backend $(repr(backend)); expected :zygote or :reactant."))
    backend === :zygote || !isempty(methods(_make_reactant_engine)) ||
        throw(ArgumentError("backend=:reactant requires the Reactant extension: run `using Reactant, Enzyme` first."))
    isnothing(normalization) || normalization in (:minmax, :whiten) ||
        throw(ArgumentError("Unknown normalization $(repr(normalization)); expected :minmax, :whiten or nothing."))
    n_epochs >= 0 || throw(ArgumentError("n_epochs must be non-negative, got $n_epochs."))
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive, got $batch_size."))
    eval_every >= 1 || throw(ArgumentError("eval_every must be positive, got $eval_every."))
    isnothing(lr) || (lr > 0 && isfinite(lr)) || throw(ArgumentError("lr must be positive and finite, got $lr."))

    if isnothing(opt) && !isnothing(lr)
        lr_T = T(lr)
        (isfinite(lr_T) && lr_T > 0) ||
            throw(ArgumentError("lr=$lr is not finite and positive in precision $T (converts to $lr_T)."))
    end

    data_T = _checked_samples(T, data, flow.n_dims, "data")
    val_T = isnothing(validation_data) ? nothing :
            _checked_samples(T, validation_data, flow.n_dims, "validation_data")

    if !isnothing(history)
        sample = isnothing(val_T) ? _history_entry(0, 0.0, nothing, nothing) : _history_entry(0, 0.0, 0.0, 0)
        sample isa eltype(history) ||
            throw(ArgumentError("history must be a Vector accepting NamedTuple entries, e.g. `Any[]`; got eltype $(eltype(history))."))
    end

    normalizer = _select_normalizer(flow, data_T, normalization, refit_normalizer)
    data_norm = SimpleFlows.normalize(normalizer, data_T)
    val_norm = isnothing(val_T) ? nothing : SimpleFlows.normalize(normalizer, val_T)
    all(isfinite, data_norm) || throw(ArgumentError("Normalised training data is not finite."))
    isnothing(val_norm) || all(isfinite, val_norm) ||
        throw(ArgumentError("Normalised validation data is not finite."))

    # From here on the call is transactional: restore the entry state on error.
    entry_ps, entry_normalizer = deepcopy(flow.ps), flow.normalizer
    try
        flow.normalizer = normalizer
        engine = backend === :zygote ?
            _make_zygote_engine(flow, data_norm, val_norm, normalizer; lr, batch_size, opt) :
            _make_reactant_engine(flow, data_norm, val_norm, normalizer; lr, batch_size, opt)
        _run_training!(engine, flow, size(data_norm, 2), !isnothing(val_norm);
                       n_epochs, batch_size, verbose, rng, eval_every, history)
    catch
        flow.ps = entry_ps
        flow.normalizer = entry_normalizer
        rethrow()
    end
    return flow
end

_history_entry(epoch, train_nll, validation_nll, best_epoch) =
    (; epoch, train_nll, validation_nll, best_epoch)

function _checked_samples(::Type{T}, x::AbstractMatrix, n_dims::Int, name::String) where {T}
    eltype(x) <: Real || throw(ArgumentError("$name must have a real element type, got $(eltype(x))."))
    size(x, 1) == n_dims ||
        throw(ArgumentError("$name has $(size(x, 1)) rows but the flow has $n_dims dimensions (expected shape (n_dims, n_samples))."))
    size(x, 2) >= 1 || throw(ArgumentError("$name contains no samples."))
    all(isfinite, x) || throw(ArgumentError("$name contains non-finite values."))
    xT = Matrix{T}(x)
    all(isfinite, xT) || throw(ArgumentError("$name is not finite after conversion to $T."))
    return xT
end

_normalizer_symbol(::MinMaxNormalizer) = :minmax
_normalizer_symbol(::WhiteningNormalizer) = :whiten

function _fit_normalizer(kind::Symbol, data::AbstractMatrix)
    try
        return kind == :minmax ? MinMaxNormalizer(data) : WhiteningNormalizer(data)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("Cannot fit the $kind normalizer on the training data: " * err.msg))
    end
end

# New flow -> minmax; fitted flow -> reuse; conflicts or refits must be explicit.
function _select_normalizer(flow::FlowDistribution, data, requested, refit::Bool)
    existing = flow.normalizer
    if isnothing(existing)
        return _fit_normalizer(isnothing(requested) ? :minmax : requested, data)
    end
    have = _normalizer_symbol(existing)
    if !isnothing(requested) && requested != have && !refit
        throw(ArgumentError("Flow already has a $have normalizer but normalization=$(repr(requested)) was requested; " *
                            "pass refit_normalizer=true to deliberately refit (this changes the coordinates of the current weights)."))
    end
    refit || return existing
    return _fit_normalizer(isnothing(requested) ? have : requested, data)
end

_all_finite(x::AbstractArray) = all(isfinite, x)
_all_finite(x::Number) = isfinite(x)
_all_finite(x::Union{NamedTuple, Tuple}) = all(_all_finite, values(x))
_all_finite(::Nothing) = true
_all_finite(::ChainRulesCore.AbstractZero) = true
# Unknown leaf types are an error rather than silently "finite".
_all_finite(x) = error("Cannot check finiteness of a parameter/gradient leaf of type $(typeof(x)).")

# Full-dataset, fixed-weight physical-coordinate mean NLL (Float64, per-sample weighted).
function _mean_physical_nll(model, ps, st, xnorm::AbstractMatrix, log_jac, batch_size::Int)
    n = size(xnorm, 2)
    total = 0.0
    for start in 1:batch_size:n
        cols = start:min(start + batch_size - 1, n)
        lp = log_prob(model, ps, st, xnorm[:, cols])
        total += sum(Float64, lp) + length(cols) * Float64(log_jac)
    end
    nll = -total / n
    isfinite(nll) || error("Non-finite negative log-likelihood ($nll) while evaluating the flow at fixed weights.")
    return nll
end

# ── Training engines ─────────────────────────────────────────────────────────
#
# `_run_training!` owns everything that must be identical across backends: batch
# order (from the caller's `rng`), evaluation schedule, history, best checkpoint
# and final-weight selection. An *engine* supplies:
#   _engine_step!(e, idx, epoch, batch_number)  one guarded optimizer update on
#                                               the samples `idx` of the training set
#   _engine_epoch_done!(e, epoch)               end-of-epoch validity check
#   _engine_nll(e, which)                       full-data fixed-weight physical NLL (Float64)
#   _engine_snapshot(e)                         independent host copy of the weights
# `_engine_snapshot` is the only way weights leave an engine, so device engines
# never alias host arrays.

mutable struct ZygoteEngine{T, F<:FlowDistribution{T}, D, V, O, J}
    flow       :: F
    data_norm  :: D
    val_norm   :: V
    opt_state  :: O
    log_jac    :: J
    batch_size :: Int
end

function _default_optimizer(::Type{T}, lr, opt) where {T}
    isnothing(opt) || return opt
    actual_lr = isnothing(lr) ? T(1f-3) : T(lr)
    return Optimisers.OptimiserChain(Optimisers.ClipGrad(T(1)), Optimisers.Adam(actual_lr))
end

function _make_zygote_engine(flow::FlowDistribution{T}, data_norm, val_norm, normalizer; lr, batch_size, opt) where {T}
    opt_state = Optimisers.setup(_default_optimizer(T, lr, opt), flow.ps)
    return ZygoteEngine(flow, data_norm, val_norm, opt_state, normalizer.log_jac, batch_size)
end

function _engine_step!(e::ZygoteEngine, idx, epoch, batch_number)
    flow = e.flow
    batch = e.data_norm[:, idx]
    loss, (dps,) = Zygote.withgradient(flow.ps) do ps
        lp = log_prob(flow.model, ps, flow.st, batch)
        -mean(lp)
    end
    isfinite(loss) ||
        error("Non-finite training loss ($loss) at epoch $epoch, batch $batch_number; stopping before the optimizer update.")
    (!isnothing(dps) && _all_finite(dps)) ||
        error("Non-finite or missing gradient at epoch $epoch, batch $batch_number; stopping before the optimizer update.")

    e.opt_state, new_ps = Optimisers.update!(e.opt_state, flow.ps, dps)
    flow.ps = new_ps
    _all_finite(new_ps) ||
        error("Optimizer update produced non-finite parameters at epoch $epoch, batch $batch_number (check the learning rate / optimizer).")
    return nothing
end

_engine_epoch_done!(::ZygoteEngine, epoch) = nothing

function _engine_nll(e::ZygoteEngine, which::Symbol)
    x = which === :train ? e.data_norm : e.val_norm
    return _mean_physical_nll(e.flow.model, e.flow.ps, e.flow.st, x, e.log_jac, e.batch_size)
end

# Optimisers.update! mutates arrays in place, hence the deep copy.
_engine_snapshot(e::ZygoteEngine) = deepcopy(e.flow.ps)

# Implemented by the Reactant extension (needs `using Reactant, Enzyme`).
function _make_reactant_engine end

function _run_training!(e, flow::FlowDistribution, n_train::Int, has_val::Bool;
                        n_epochs, batch_size, verbose, rng, eval_every, history)
    loader = DataLoader(collect(1:n_train); batchsize=batch_size, shuffle=true, rng=rng)
    track = has_val || !isnothing(history) || verbose
    best_nll = Inf
    best_epoch = nothing
    best_ps = nothing

    function evaluate!(epoch)
        track || return
        tr = _engine_nll(e, :train)
        va = has_val ? _engine_nll(e, :val) : nothing
        if !isnothing(va) && va < best_nll
            best_nll, best_epoch = va, epoch
            best_ps = _engine_snapshot(e)
        end
        isnothing(history) || push!(history, _history_entry(epoch, tr, va, best_epoch))
        if verbose
            @info "Epoch $(lpad(epoch, 5)) | full-data physical NLL: train $(round(tr; digits=5))" *
                  (isnothing(va) ? "" : " | validation $(round(va; digits=5)) | best epoch $best_epoch") epoch train_nll=tr validation_nll=va best_epoch
        end
        return
    end

    evaluate!(0)
    for epoch in 1:n_epochs
        for (b, idx) in enumerate(loader)
            _engine_step!(e, idx, epoch, b)
        end
        _engine_epoch_done!(e, epoch)
        if epoch % eval_every == 0 || epoch == n_epochs
            evaluate!(epoch)
        end
    end

    if has_val && best_epoch != n_epochs
        flow.ps = best_ps
    else
        flow.ps = _engine_snapshot(e)
    end
    return nothing
end
