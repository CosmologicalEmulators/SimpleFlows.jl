# MODIFIED: training.jl — flattened training loop using Iterators.cycle

"""
    train_flow!(flow, data; n_epochs, lr, batch_size, verbose) -> FlowDistribution

Fit a `MinMaxNormalizer` from `data`, then train `flow` by minimising the
negative log-likelihood on the normalised data.

# Arguments
- `flow::FlowDistribution`: the flow to train (mutated in-place).
- `data::AbstractMatrix`: training samples, shape `(n_dims, n_samples)`.

# Keyword Arguments
- `n_epochs=1000`: number of passes over the dataset.
- `lr=1f-3`: learning rate for Adam.
- `batch_size=256`: mini-batch size.
- `verbose=true`: print NLL every 100 epochs.
"""
function train_flow!(flow::FlowDistribution{T}, data::AbstractMatrix;
                     n_epochs::Int=1000,
                     lr::Union{Nothing, Real}=nothing,
                     batch_size::Int=256,
                     verbose::Bool=true,
                     opt=nothing) where {T}
    # Fit & Attach Normalizer
    flow.normalizer = MinMaxNormalizer(T.(data))
    data_norm = SimpleFlows.normalize(flow.normalizer, data)

    actual_opt = if isnothing(opt)
        actual_lr = isnothing(lr) ? T(1f-3) : T(lr)
        Optimisers.OptimiserChain(Optimisers.ClipGrad(T(1)), Optimisers.Adam(actual_lr))
    else
        opt
    end
    opt_state = Optimisers.setup(actual_opt, flow.ps)

    # Flatten the loop: use total iterations instead of nested epoch/batch loops.
    # partial=false avoids recompilation due to changing batch shapes.
    n_batches_per_epoch = size(data_norm, 2) ÷ batch_size
    maxiters = n_epochs * n_batches_per_epoch
    
    loader = Iterators.cycle(
        DataLoader(data_norm; batchsize=batch_size, shuffle=true, partial=false)
    )

    running_loss = zero(T)
    for (iter, batch) in enumerate(loader)
        # Compute gradient and update parameters
        loss, (dps,) = Zygote.withgradient(flow.ps) do ps
            lp = log_prob(flow.model, ps, flow.st, batch)
            -Statistics.mean(lp)
        end

        opt_state, new_ps = Optimisers.update!(opt_state, flow.ps, dps)
        flow.ps = new_ps

        running_loss += loss

        # Periodic logging at epoch boundaries
        if n_batches_per_epoch > 0 && iter % n_batches_per_epoch == 0
            epoch = iter ÷ n_batches_per_epoch
            if verbose && epoch % 100 == 0
                @info "Epoch $(lpad(epoch, 5)) | mean NLL: $(round(running_loss / n_batches_per_epoch; digits=4))"
            end
            running_loss = zero(T)
        end

        iter ≥ maxiters && break
    end

    return flow
end

"""
    train_flow_reactant!(flow, data; kwargs...) -> FlowDistribution

Train a flow using the Reactant (XLA) backend for high-performance execution.
Requires `using Reactant` and `using Enzyme` to be loaded in your environment.
"""
function train_flow_reactant! end

