module ReactantExt

using SimpleFlows
using Reactant, Enzyme
using Lux, Statistics, Random, Optimisers
using DifferentiationInterface
using MLUtils

function loss_function(model, ps, st, x)
    lp = SimpleFlows.log_prob(model, ps, st, x)
    loss = -mean(lp)
    return loss, st, (;)
end

function SimpleFlows.train_flow_reactant!(flow::FlowDistribution{T}, data::AbstractMatrix;
                     n_epochs::Int=1000,
                     lr::Union{Nothing, Real}=nothing,
                     batch_size::Int=256,
                     verbose::Bool=true) where {T}

    # Fit & Attach Normalizer
    flow.normalizer = MinMaxNormalizer(T.(data))
    data_norm = SimpleFlows.normalize(flow.normalizer, data)

    # Set up devices
    xdev = reactant_device(; force=true)
    cdev = cpu_device()

    # Move parameters and state to Reactant
    ps_x = flow.ps |> xdev
    st_x = flow.st |> xdev

    actual_lr = isnothing(lr) ? 1f-3 : Float32(lr)

    # Use Adam with explicitly Float32 arguments to prevent type conversion issues in XLA
    opt = Adam(actual_lr, (0.9f0, 0.999f0), 1f-8)

    train_state = Lux.Training.TrainState(flow.model, ps_x, st_x, opt)

    n_batches_per_epoch = size(data_norm, 2) ÷ batch_size
    if n_batches_per_epoch == 0
        error("Batch size is larger than dataset size.")
    end

    maxiters = n_epochs * n_batches_per_epoch

    # Construct dataloader mapped to the reactant device
    dataloader = DataLoader(data_norm; batchsize=batch_size, shuffle=false, partial=false) |> xdev |> Iterators.cycle

    total_samples = 0
    start_time = time()

    for (iter, x) in enumerate(dataloader)
        total_samples += size(x, ndims(x))

        (_, loss, _, train_state) = Lux.Training.single_train_step!(
            AutoEnzyme(), loss_function, x, train_state; return_gradients=Val(false)
        )

        isnan(loss) && error("NaN loss encountered in iter $(iter)!")

        if verbose && (iter == 1 || iter == maxiters || iter % 1000 == 0)
            throughput = total_samples / (time() - start_time)
            cpu_loss = Float32(loss)
            @info "Iter: [$(lpad(iter, 6))/$(lpad(maxiters, 6))] | Training Loss: $(round(cpu_loss, digits=6)) | Throughput: $(round(throughput, digits=2)) samples/s"
        end

        iter ≥ maxiters && break
    end



    # Move trained parameters back to CPU and update the flow object
    flow.ps = train_state.parameters |> cdev
    flow.st = train_state.states |> cdev

    return flow
end

end
