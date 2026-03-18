"""
Train a Neural Spline Flow (NSF) normalizing flow to approximate a 4-dimensional correlated Gaussian,
then use Turing.jl to run NUTS inference under both the exact prior and the NF prior.

Usage:
    julia --project=. -t 4 examples/train_multinormal_nsf.jl

Note: Turing.jl must be installed in your global environment or activated environment.
"""

using SimpleFlows
using Distributions, LinearAlgebra, Random, Statistics, Printf
using Turing

rng = Random.MersenneTwister(42)

# ── 1. Define a non-trivial 4D target distribution ───────────────────────────

μ = [1.0, -0.5, 2.0, 0.3]
Σ = [1.00  0.50  0.10  0.00;
     0.50  1.00  0.30  0.20;
     0.10  0.30  1.00  0.40;
     0.00  0.20  0.40  1.00]
target = MvNormal(μ, Σ)

# ── 2. Draw training data ─────────────────────────────────────────────────────

N_train = 10_000
data    = Float64.(rand(rng, target, N_train))   # 4 × 10_000

println("Target mean:     ", round.(mean(data; dims=2)[:]; digits=3))
println("Target std:      ", round.(std(data; dims=2)[:]; digits=3))

# ── 3. Build and train the flow ───────────────────────────────────────────────

# Note: NSF typically requires fewer transforms but slightly larger hidden layers 
# or just more expressive bins (K).
flow = FlowDistribution(Float64;
    architecture        = :NSF,
    n_transforms        = 4,
    dist_dims           = 4,
    hidden_layer_sizes  = [32, 32], 
    K                   = 8,
    tail_bound          = 3.0,
    rng,
)

println("\nTraining NSF (4 transforms, K=8, [32, 32] hidden units, 500 epochs)…")
train_flow!(flow, data;
    n_epochs   = 500,
    lr         = 1e-3,
    batch_size = 512,
    verbose    = true,
)

# ── 4. Evaluate fit ───────────────────────────────────────────────────────────

N_test      = 5_000
x_test      = Float64.(rand(rng, target, N_test))

lp_true  = mean(logpdf(target, Float64.(x_test)))
lp_flow  = mean(logpdf(flow, x_test))

println("\n── Density Fit ──────────────────────────────────────────")
println("Mean log-pdf (true distribution): ", round(lp_true; digits=4))
println("Mean log-pdf (trained NSF flow):  ", round(Float64(lp_flow); digits=4))
println("Difference:                        ", round(abs(lp_true - lp_flow); digits=4))

samples = Distributions.rand(rng, flow, N_test)   # 4 × N_test
println("\nFlow sample mean: ", round.(mean(samples; dims=2)[:]; digits=3))
println("Flow sample std:  ", round.(std(samples; dims=2)[:]; digits=3))

# ── 5. Save the trained flow ─────────────────────────────────────────────────

save_dir = joinpath(@__DIR__, "..", "trained_flows", "nsf_mvn_4d")
save_trained_flow(save_dir, flow)
println("\nFlow saved to $save_dir")

# ── 6. Reload and verify round-trip ──────────────────────────────────────────

flow2 = load_trained_flow(save_dir; rng)
lp_reloaded = mean(logpdf(flow2, x_test))
println("Mean log-pdf after reload: ", round(Float64(lp_reloaded); digits=4))
@assert lp_reloaded ≈ lp_flow  atol=1f-3  "Round-trip failed!"
println("Round-trip OK ✓")

# ── 7. Turing.jl inference demo ───────────────────────────────────────────────

println("\n── Turing Inference ─────────────────────────────────────────")

θ_true = [1.0, -0.5, 2.0, 0.3]
y_obs  = θ_true[1] + 0.5 * randn(rng)
println("Observed y: ", round(y_obs; digits=4), "  (true θ[1] = $(θ_true[1]))")

@model function inference_model(y_obs, prior)
    θ ~ prior
    y_obs ~ Normal(θ[1], 0.5)
end

n_samples = 1000
n_chains  = 4

# ── 7a. Exact MvNormal prior ─────────────────────────────────────────────────
println("\nSampling with exact MvNormal prior ($n_chains chains × $n_samples samples)…")
chain_exact = sample(
    inference_model(y_obs, target),
    NUTS(), MCMCThreads(), n_samples, n_chains;
    progress = false,
)

# ── 7b. Trained NSF prior ────────────────────────────────────────────────────
println("Sampling with trained NSF prior ($n_chains chains × $n_samples samples)…")
chain_nf = sample(
    inference_model(y_obs, flow2),
    NUTS(), MCMCThreads(), n_samples, n_chains;
    progress = false,
)

# ── 8. Compare posteriors ────────────────────────────────────────────────────

println("\n── Posterior comparison (θ[1]) ──────────────────────────────")
exact_θ1 = vec(chain_exact[:, "θ[1]", :])
nf_θ1    = vec(chain_nf[:, "θ[1]", :])

@printf "  True θ[1]:              %.4f\n" θ_true[1]
@printf "  Observed y:             %.4f\n" y_obs
@printf "  Posterior mean (exact): %.4f  ± %.4f\n" mean(exact_θ1) std(exact_θ1)
@printf "  Posterior mean (NSF):   %.4f  ± %.4f\n" mean(nf_θ1) std(nf_θ1)

println("\n✨ End of script: Turing MCMC sampling with NSF finished successfully!")
