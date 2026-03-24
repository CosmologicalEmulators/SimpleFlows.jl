using SimpleFlows
using SimpleFlows: log_prob, draw_samples
using Lux, Random, Statistics, Optimisers, Printf
using Reactant, Enzyme
using DifferentiationInterface

# 1. Setup Backend and Devices
try
    Reactant.set_default_backend("cpu")
    println("Reactant backend explicitly set to 'cpu'.")
catch e
    @warn "Could not set Reactant backend. Relying on env vars. Error: $e"
end

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
println("Reactant device: ", xdev)
println("CPU device: ", cdev)


# 2. Toy Dataset & Loss Function
function generate_mixture_data(n, rng=Random.default_rng())
    x1 = randn(rng, Float32, 2, n ÷ 2) .- 2.0f0
    x2 = randn(rng, Float32, 2, n ÷ 2) .+ 2.0f0
    return hcat(x1, x2)
end

# Lux.Training expects the loss function to return (loss, st, stats)
function loss_function(model, ps, st, x)
    lp = log_prob(model, ps, st, x)
    loss = -mean(lp)
    return loss, st, (;)
end

function main()
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    D = 2
    n_transforms = 4
    hidden_dims = 32
    println("Initializing RealNVP...")

    model = RealNVP(; n_transforms=n_transforms, dist_dims=D, hidden_layer_sizes=[hidden_dims, hidden_dims])

    println("\n--- Performance Comparison: Original train_flow! (Zygote on CPU) ---")
    ps_cpu, st_cpu = Lux.setup(rng, model)
    ps_cpu = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, ps_cpu)

    # Generate a fixed dataset for the CPU training
    n_train_samples = 512 * 10 # 10 batches per epoch
    x_train_cpu = generate_mixture_data(n_train_samples, rng)

    # Wrap in FlowDistribution
    flow_cpu = FlowDistribution(model, ps_cpu, st_cpu, D, [hidden_dims, hidden_dims], MinMaxNormalizer(x_train_cpu))

    # Train for 100 epochs (100 * 10 = 1000 steps)
    t_start_cpu = time()
    train_flow!(flow_cpu, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)
    t_end_cpu = time()

    t_start_cpu = time()
    train_flow!(flow_cpu, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)
    t_end_cpu = time()

    @time train_flow!(flow_cpu, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)

    total_samples_cpu = n_train_samples * 100
    throughput_cpu = total_samples_cpu / (t_end_cpu - t_start_cpu)
    @printf("Original train_flow! completed 1000 steps in %.2fs | Throughput: %.2f samples/s\n", (t_end_cpu - t_start_cpu), throughput_cpu)

    println("\n--- Performance Comparison: Reactant (Enzyme on XLA) ---")

    # Move model parameters and state to the reactant device
    ps, st = Lux.setup(rng, model) |> xdev


    # Use Float32 for all optimizer hyperparameters to avoid TracedRNumber type conversion issues
    opt = Adam(1f-3, (0.9f0, 0.999f0), 1f-8)

    # Create the training state
    train_state = Lux.Training.TrainState(model, ps, st, opt)

    @printf("Total Trainable Parameters: %d\n", Lux.parameterlength(ps))

    println("\n--- Starting Training Loop (100 steps for brevity) ---")

    total_samples = 0
    start_time = time()
    batchsize = 512
    maxiters = 10000

    for iter in 1:maxiters
        # Generate batch and move to device
        x_batch = generate_mixture_data(batchsize, rng) |> xdev
        total_samples += size(x_batch, ndims(x_batch))

        # Use Lux.Training's single_train_step! which handles compilation and optimizer updates gracefully
        (_, loss, _, train_state) = Lux.Training.single_train_step!(
            AutoEnzyme(), loss_function, x_batch, train_state; return_gradients=Val(false)
        )

        if isnan(loss)
            error("NaN loss encountered in iter $iter!")
        end

        if iter == 1 || iter == maxiters || iter % 1000 == 0
            throughput = total_samples / (time() - start_time)
            @printf("Step %4d | Loss: %8.4f | Throughput: %8.2f samples/s\n", iter, loss, throughput)
        end
    end

    @time for iter in 1:maxiters
        # Generate batch and move to device
        x_batch = generate_mixture_data(batchsize, rng) |> xdev
        total_samples += size(x_batch, ndims(x_batch))

        # Use Lux.Training's single_train_step! which handles compilation and optimizer updates gracefully
        (_, loss, _, train_state) = Lux.Training.single_train_step!(
            AutoEnzyme(), loss_function, x_batch, train_state; return_gradients=Val(false)
        )

        if isnan(loss)
            error("NaN loss encountered in iter $iter!")
        end

        if iter == 1 || iter == maxiters || iter % 1000 == 0
            throughput = total_samples / (time() - start_time)
            @printf("Step %4d | Loss: %8.4f | Throughput: %8.2f samples/s\n", iter, loss, throughput)
        end
    end

    println("\nTraining complete!")

    # Move parameters back to CPU for sampling
    ps_final = train_state.parameters |> cdev
    st_final = train_state.states |> cdev

    samples = draw_samples(rng, Float32, model, ps_final, st_final, 1000)
    println("Samples generated: ", size(samples))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
