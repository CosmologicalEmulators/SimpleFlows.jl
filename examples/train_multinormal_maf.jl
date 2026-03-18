# examples/train_multinormal_maf.jl
using SimpleFlows
using Random
using Statistics
using Distributions, LinearAlgebra, Random, Statistics, Printf
using Turing
using Optimisers

function run_maf_example()
    # ── 1. Setup Data ──────────────────────────────────────────────────────────
    rng = Random.default_rng()
    Random.seed!(rng, 123)
    
    dist_dims = 4
    # Create a correlated 4D Gaussian
    μ_true = [1.0, -0.5, 2.0, 0.3]
    Σ_true = [1.0  0.5  0.2  0.1;
              0.5  1.0  0.4  0.2;
              0.2  0.4  1.0  0.5;
              0.1  0.2  0.5  1.0]
    
    true_dist = MvNormal(μ_true, Σ_true)
    n_samples = 5000
    target_data = rand(rng, true_dist, n_samples)
    
    println("Target mean:     ", round.(mean(target_data, dims=2)[:], digits=3))
    println("Target std:      ", round.(std(target_data, dims=2)[:], digits=3))
    println()

    # ── 2. Initialize MAF Flow ───────────────────────────────────────────────
    # MAF is generally more expressive than RealNVP for same number of layers
    n_transforms = 4
    hidden_layer_sizes = [32, 32]
    
    println("Training MAF ($n_transforms transforms, $(hidden_layer_sizes) units, 500 epochs)…")
    
    dist = FlowDistribution(Float32; 
        architecture=:MAF,
        n_transforms=n_transforms, 
        dist_dims=dist_dims,
        hidden_layer_sizes=hidden_layer_sizes,
        rng=rng
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    # We use a smaller learning rate for MAF as it can be more sensitive
    train_flow!(dist, target_data; n_epochs=500, batch_size=200, opt=Optimisers.Adam(1e-3))

    # ── 4. Evaluate Density ──────────────────────────────────────────────────
    test_data = rand(rng, true_dist, 1000)
    lp_true = logpdf(true_dist, test_data)
    lp_flow = logpdf(dist, test_data)

    println("\n── Density Fit ──────────────────────────────────────────")
    println("Mean log-pdf (true distribution): ", round(mean(lp_true), digits=4))
    println("Mean log-pdf (trained MAF flow):  ", round(mean(lp_flow), digits=4))
    println("Difference:                        ", round(abs(mean(lp_true) - mean(lp_flow)), digits=4))

    samples = rand(rng, dist, 5000)
    println("\nFlow sample mean: ", round.(mean(samples, dims=2)[:], digits=3))
    println("Flow sample std:  ", round.(std(samples, dims=2)[:], digits=3))

    # ── 5. Serialization ─────────────────────────────────────────────────────
    save_path = joinpath(@__DIR__, "../trained_flows/maf_mvn_4d")
    save_trained_flow(save_path, dist)
    
    # Reload and verify
    dist_reloaded = load_trained_flow(save_path)
    lp_reloaded = logpdf(dist_reloaded, test_data)
    println("Mean log-pdf after reload: ", round(mean(lp_reloaded), digits=4))
    if isapprox(mean(lp_flow), mean(lp_reloaded), atol=1e-5)
        println("Round-trip OK ✓")
    else
        println("Round-trip FAILED ✗")
    end

    # ── 6. Turing.jl Integration ───────────────────────────────────────────
    println("\n── Turing Inference ─────────────────────────────────────────")
    # Observed data: one sample from θ[1] with noise
    θ_true = μ_true
    y_obs = θ_true[1] + 0.1 * randn(rng)
    println("Observed y: ", round(y_obs, digits=4), "  (true θ[1] = ", θ_true[1], ")")

    @model function linear_model(y, prior_dist)
        θ ~ prior_dist
        y ~ Normal(θ[1], 0.1)
    end

    # Sampling with exact prior
    println("\nSampling with exact MvNormal prior (4 chains × 1000 samples)…")
    chain_exact = sample(linear_model(y_obs, true_dist), HMC(0.1, 10), MCMCThreads(), 1000, 4; progress=false)

    # Sampling with MAF prior
    println("Sampling with trained MAF prior (4 chains × 1000 samples)…")
    chain_maf = sample(linear_model(y_obs, dist), HMC(0.1, 10), MCMCThreads(), 1000, 4; progress=false)

    println("\n── Posterior comparison (θ[1]) ──────────────────────────────")
    exact_θ1 = vec(chain_exact[:, "θ[1]", :])
    maf_θ1   = vec(chain_maf[:, "θ[1]", :])

    @printf "  True θ[1]:              %.4f\n" θ_true[1]
    @printf "  Observed y:             %.4f\n" y_obs
    @printf "  Posterior mean (exact): %.4f  ± %.4f\n" mean(exact_θ1) std(exact_θ1)
    @printf "  Posterior mean (MAF):   %.4f  ± %.4f\n" mean(maf_θ1) std(maf_θ1)

    println("\n✨ End of script: Turing MCMC sampling with MAF finished successfully!")
end

if abspath(PROGRAM_FILE) == joinpath(@__DIR__, "train_multinormal_maf.jl")
    run_maf_example()
end
