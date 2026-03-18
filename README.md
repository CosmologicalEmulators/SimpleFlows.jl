# SimpleFlows.jl

A Julia package for training normalizing flows and using them as
[`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl)
distributions inside [`Turing.jl`](https://turinglang.org/) probabilistic programs.

## Features

- **Architectures**:
  - **RealNVP** (Real-valued Non-Volume Preserving)
  - **NSF** (Neural Spline Flow with Rational Quadratic Splines)
  - **MAF** (Masked Autoregressive Flow with MADE)
- Full `Distributions.jl` interface: `logpdf`, `rand`, `length`
- `Bijectors.jl` compatible → plug directly into Turing models as a prior
- Training via `Optimisers.Adam` + `Zygote.jl` autodiff, mini-batched with `MLUtils`
- **Save / Load**: architecture stored as JSON, weights as NPZ (Python-readable via `numpy.load`)

## Quick Start

```julia
using SimpleFlows, Distributions, LinearAlgebra

# 1. Build a 4-dim flow (options: :RealNVP, :NSF, :MAF)
flow = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=6, dist_dims=4, hidden_layer_sizes=[64, 64, 64])

# 2. Sample training data from your target distribution
data = Float32.(rand(MvNormal(zeros(4), I), 10_000))

# 3. Train
train_flow!(flow, data; n_epochs=500, lr=1f-3)

# 4. Use like any Distributions.jl distribution
logpdf(flow, randn(4))
rand(flow)

# 5. Save / Load (weights in NPZ — readable from Python with numpy.load)
save_trained_flow("my_flow/", flow)
flow2 = load_trained_flow("my_flow/")
```

## Turing.jl Integration

```julia
using Turing

@model function my_model(y)
    θ ~ flow2            # ← trained flow as a prior
    y ~ MvNormal(θ, I)
end
```

## File Format

Saved directories contain:
```
my_flow/
├── flow_setup.json   # architecture (n_transforms, dist_dims, etc.)
└── weights.npz       # flat parameter arrays (numpy-compatible)
```

## Architectures

| Architecture | Status  |
|---|---|
| RealNVP      | ✅ Done |
| MAF          | ✅ Done |
| NSF          | ✅ Done |

## Running Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

## Running the Examples

```bash
julia --project=. examples/train_multinormal.jl
julia --project=. examples/train_multinormal_nsf.jl
julia --project=. examples/train_multinormal_maf.jl
```
