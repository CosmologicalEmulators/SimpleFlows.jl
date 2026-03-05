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
    # Always fit and apply a min-max normalizer
    flow.normalizer = MinMaxNormalizer(T.(data))
    data_T = normalize(flow.normalizer, T.(data))

    actual_opt = if isnothing(opt)
        actual_lr = isnothing(lr) ? T(1f-3) : T(lr)
        Optimisers.OptimiserChain(Optimisers.ClipGrad(T(1)), Optimisers.Adam(actual_lr))
    else
        opt
    end
    opt_state = Optimisers.setup(actual_opt, flow.ps)

    loader = DataLoader(data_T; batchsize=batch_size, shuffle=true)

    for epoch in 1:n_epochs
        total_loss = zero(T)
        n_batches  = 0

        for batch in loader
            loss, (dps,) = Zygote.withgradient(flow.ps) do ps
                lp = log_prob(flow.model, ps, flow.st, batch)
                -mean(lp)
            end

            opt_state, new_ps = Optimisers.update!(opt_state, flow.ps, dps)
            flow.ps = new_ps

            total_loss += loss
            n_batches  += 1
        end

        if verbose && epoch % 100 == 0
            @info "Epoch $(lpad(epoch, 5)) | mean NLL: $(round(total_loss / n_batches; digits=4))"
        end
    end

    return flow
end
