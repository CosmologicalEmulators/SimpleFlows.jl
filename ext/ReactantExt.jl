module ReactantExt

using SimpleFlows
using Reactant, Enzyme
using Lux, Statistics, Random, Optimisers
using DifferentiationInterface
using MLUtils

# ── Adam update: @generated leaf-level functions (no lambda, no closure) ─────
#
# Inside Reactant.compile, passing a closure `f` to a tree-walker causes
# Reactant's make_tracer/call_with_reactant to wrap the closure with the outer
# data arguments as captured type parameters.  When the wrapped closure is then
# called inside the recursive tree-walk the argument count diverges from the
# expected 2- or 3-arg method signature, causing MethodError.
#
# Bypass this entirely: encode the Adam update equations as @generated functions
# that recurse on NamedTuple keys statically, calling an @inline leaf function
# directly — zero lambdas, zero captured context.

# mt_new = β₁·m + (1-β₁)·g
@inline _adam_mt(m::AbstractArray, g::AbstractArray, beta1::F) where F =
    @. beta1 * m + (1f0 - beta1) * g
@generated function _adam_mt(m::NamedTuple{K}, g::NamedTuple{K}, beta1) where K
    body = Expr(:tuple, [:(ReactantExt._adam_mt(m.$k, g.$k, beta1)) for k in K]...)
    :(NamedTuple{$K}($body))
end

# vt_new = β₂·v + (1-β₂)·g²
@inline _adam_vt(v::AbstractArray, g::AbstractArray, beta2::F) where F =
    @. beta2 * v + (1f0 - beta2) * g * g
@generated function _adam_vt(v::NamedTuple{K}, g::NamedTuple{K}, beta2) where K
    body = Expr(:tuple, [:(ReactantExt._adam_vt(v.$k, g.$k, beta2)) for k in K]...)
    :(NamedTuple{$K}($body))
end

# ps_new = p - lr·m / (√v + ε)
@inline _adam_ps(p::AbstractArray, m::AbstractArray, v::AbstractArray, lr::F, eps::F) where F =
    @. p - lr * m / (sqrt(v) + eps)
@generated function _adam_ps(p::NamedTuple{K}, m::NamedTuple{K}, v::NamedTuple{K}, lr, eps) where K
    body = Expr(:tuple, [:(ReactantExt._adam_ps(p.$k, m.$k, v.$k, lr, eps)) for k in K]...)
    :(NamedTuple{$K}($body))
end

# Initialise a zero-filled tree with the same structure as `x`.
@inline _zeros_like(x::AbstractArray) = Reactant.ConcreteRArray(zeros(Float32, size(x)))
@generated function _zeros_like(x::NamedTuple{K}) where K
    body = Expr(:tuple, [:(ReactantExt._zeros_like(x.$k)) for k in K]...)
    :(NamedTuple{$K}($body))
end

# ── Gradient clipping (global L2 norm, matching Optimisers.ClipGrad) ─────────
# Step 1: accumulate sum of squared elements across the entire gradient tree.
# Use @. g*g (no lambda) — lambdas cause MethodError inside Reactant tracing.
@inline _grad_norm_sq(g::AbstractArray) = sum(@. g * g)
@generated function _grad_norm_sq(g::NamedTuple{K}) where K
    isempty(K) && return :(zero(Float32))
    terms = [:(ReactantExt._grad_norm_sq(g.$k)) for k in K]
    :(+($(terms...)))
end

# Step 2: scale every leaf by `scale`.
@inline _scale_grads(g::AbstractArray, scale) = @. g * scale
@generated function _scale_grads(g::NamedTuple{K}, scale) where K
    body = Expr(:tuple, [:(ReactantExt._scale_grads(g.$k, scale)) for k in K]...)
    :(NamedTuple{$K}($body))
end

# ── Combined train step ───────────────────────────────────────────────────────
function train_step(ps, mt, vt, model, st, x, lr, beta1, beta2, eps)
    loss, grads = DifferentiationInterface.value_and_gradient(
        p -> -mean(SimpleFlows.log_prob(model, p, st, x)),
        AutoEnzyme(), ps
    )
    # Global gradient clipping matching Optimisers.ClipGrad(1):
    # scale = min(1, clip_norm / norm).  Use ifelse — XLA-safe, no branching.
    clip_norm = 1f0
    norm      = sqrt(_grad_norm_sq(grads))
    scale     = ifelse(norm > clip_norm, clip_norm / norm, 1f0)
    grads     = _scale_grads(grads, scale)
    mt_new = _adam_mt(mt, grads, beta1)
    vt_new = _adam_vt(vt, grads, beta2)
    ps_new = _adam_ps(ps, mt_new, vt_new, lr, eps)
    return loss, ps_new, mt_new, vt_new
end

# ── train_flow_reactant! ──────────────────────────────────────────────────────
function SimpleFlows.train_flow_reactant!(flow::FlowDistribution{T}, data::AbstractMatrix;
                     n_epochs::Int=1000,
                     lr::Union{Nothing, Real}=nothing,
                     batch_size::Int=256,
                     verbose::Bool=true) where {T}

    # Fit & Attach Normalizer
    flow.normalizer = MinMaxNormalizer(T.(data))
    data_norm = SimpleFlows.normalize(flow.normalizer, data)

    xdev = reactant_device(; force=true)
    cdev = cpu_device()

    ps_x = flow.ps |> xdev
    st_x = flow.st |> xdev

    actual_lr = isnothing(lr) ? 1f-3 : Float32(lr)
    beta1     = 0.9f0
    beta2     = 0.999f0
    eps       = 1f-8

    # Moment accumulators: same tree shape as ps_x, zero-initialised.
    mt = _zeros_like(ps_x)
    vt = _zeros_like(ps_x)

    n_batches_per_epoch = size(data_norm, 2) ÷ batch_size
    n_batches_per_epoch == 0 && error("Batch size is larger than dataset size.")
    maxiters = n_epochs * n_batches_per_epoch

    dataloader = DataLoader(data_norm; batchsize=batch_size, shuffle=false,
                            partial=false) |> xdev |> Iterators.cycle
    x_dummy = first(dataloader)

    if verbose
        @info "Compiling Reactant graph (combined grad + Adam)..."
    end
    compiled_step = Reactant.compile(
        train_step,
        (ps_x, mt, vt, flow.model, st_x, x_dummy, actual_lr, beta1, beta2, eps)
    )

    total_samples = 0
    start_time    = time()

    for (iter, x) in enumerate(dataloader)
        total_samples += size(x, ndims(x))

        loss, ps_x, mt, vt = compiled_step(
            ps_x, mt, vt, flow.model, st_x, x, actual_lr, beta1, beta2, eps
        )

        cpu_loss = Float32(loss)
        isnan(cpu_loss) && error("NaN loss encountered in iter $(iter)!")

        if verbose && (iter == 1 || iter == maxiters || iter % 1000 == 0)
            throughput = total_samples / (time() - start_time)
            @info "Iter: [$(lpad(iter, 6))/$(lpad(maxiters, 6))] | Training Loss: $(round(cpu_loss, digits=6)) | Throughput: $(round(throughput, digits=2)) samples/s"
        end

        iter ≥ maxiters && break
    end

    flow.ps = ps_x |> cdev
    flow.st = st_x |> cdev
    return flow
end

end
