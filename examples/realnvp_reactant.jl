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

    # Train for 10 epochs (10 * 10 = 100 steps)
    #t_start_cpu = time()
    #train_flow!(flow_cpu, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)
    #t_end_cpu = time()

    #t_start_cpu = time()
    #train_flow!(flow_cpu, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)
    #t_end_cpu = time()

    #@time train_flow!(flow_cpu, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)

    #total_samples_cpu = n_train_samples * 10
    #throughput_cpu = total_samples_cpu / (t_end_cpu - t_start_cpu)
    #@printf("Original train_flow! completed 100 steps in %.2fs | Throughput: %.2f samples/s\n", (t_end_cpu - t_start_cpu), throughput_cpu)

    #println("\n--- Performance Comparison: Reactant (Enzyme on XLA) ---")

    # Instantiate a fresh FlowDistribution for the Reactant run
    ps_xla, st_xla = Lux.setup(rng, model)
    ps_xla = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, ps_xla)
    flow_xla = FlowDistribution(model, ps_xla, st_xla, D, [hidden_dims, hidden_dims], MinMaxNormalizer(x_train_cpu))

    println("Starting train_flow_reactant! (first step includes XLA compilation time)...")
    t_start_xla = time()
    train_flow_reactant!(flow_xla, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=true)
    t_end_xla = time()

    # Run a second time to see the pure execution speed without compilation overhead
    println("\nRunning train_flow_reactant! again (pure XLA execution speed)...")
    t_start_xla_pure = time()
    train_flow_reactant!(flow_xla, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)
    t_end_xla_pure = time()
    @time train_flow_reactant!(flow_xla, x_train_cpu; n_epochs=10000, lr=1f-3, batch_size=512, verbose=false)

    throughput_xla_pure = (n_train_samples * 100) / (t_end_xla_pure - t_start_xla_pure)
    @printf("Reactant train_flow_reactant! completed 1000 steps in %.2fs | Throughput: %.2f samples/s\n", (t_end_xla_pure - t_start_xla_pure), throughput_xla_pure)

    speedup = throughput_xla_pure / throughput_cpu
    @printf("\n--> Reactant speedup vs Zygote: %.2fx\n", speedup)

    println("\nTraining complete!")

    samples = draw_samples(rng, Float32, flow_xla.model, flow_xla.ps, flow_xla.st, 10000)
    println("Samples generated: ", size(samples))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
