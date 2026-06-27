#!/usr/bin/env julia
# examples/compile_reactant_example.jl
#
# A complete standalone example demonstrating how to train a normalizing flow
# using the standard pipeline, port the trained weights to Reactant, and JIT-compile
# it for high-performance inference/evaluation.

using SimpleFlows
using Random
using Statistics
using Distributions: logpdf
using Reactant

function main()
    # 1. Setup seed and device backend
    try
        Reactant.set_default_backend("cpu")
        println("Using Reactant CPU-XLA backend.")
    catch
        println("Using Reactant default GPU/TPU accelerator backend.")
    end

    rng = Random.MersenneTwister(42)

    # 2. Generate a synthetic 2D dataset
    n_samples = 10_000
    data = randn(rng, Float32, 2, n_samples)
    
    # 3. Initialize a 2D RealNVP Flow
    println("\nInitializing RealNVP Normalizing Flow...")
    flow = FlowDistribution(Float32;
        architecture = :RealNVP,
        n_transforms = 4,
        dist_dims = 2,
        hidden_layer_sizes = [16, 16],
        rng = rng
    )

    # 4. Fit normalizer and train the model using standard CPU pipeline
    println("\nTraining with the standard CPU pipeline (100 epochs)...")
    train_flow!(flow, data;
        n_epochs = 100,
        batch_size = 256,
        verbose = true
    )

    # 5. Port the trained flow weights and normalizer state to Reactant
    println("\nMoving trained flow model parameters and state to Reactant...")
    flow_react = to_reactant(flow)

    # 6. Evaluate log-probabilities on new data using Reactant XLA Compilation
    println("Running a JIT-compiled Reactant evaluation on new data...")
    new_data = randn(rng, Float32, 2, 5)
    x_react = Reactant.to_rarray(new_data)
    
    # JIT-compile the logpdf function
    compiled_logpdf = Reactant.@compile logpdf(flow_react, x_react)
    
    # Run the compiled function (first call compiles, subsequent calls are near-instant)
    log_probs = compiled_logpdf(flow_react, x_react)
    
    println("Data:")
    display(new_data)
    println("Log probabilities (computed via compiled Reactant):")
    display(Array(log_probs))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
