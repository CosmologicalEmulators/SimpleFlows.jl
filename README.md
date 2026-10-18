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
- Input normalization: `:minmax` (default) or `:whiten` (mean + Cholesky covariance)
- **Save / Load**: architecture stored as JSON, weights as NPZ (Python-readable via `numpy.load`);
  versioned artifacts record precision, activation, normalizer kind and optional parameter names

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

### Whitening and parameter names

```julia
# Normalise with the sample mean and Cholesky factor of the covariance
# (Σ = L Lᵀ, z = L \ (x - μ)) instead of per-coordinate min-max scaling.
flow = FlowDistribution(Float64; architecture=:NSF, n_transforms=4, dist_dims=3,
                        hidden_layer_sizes=[64, 64], parameter_names=["omega_m", "h", "b1"])
train_flow!(flow, data; normalization=:whiten)   # default is :minmax

save_trained_flow("my_flow/", flow)              # names are saved with the flow
flow2 = load_trained_flow("my_flow/")
flow2.parameter_names                            # ["omega_m", "h", "b1"]
```

`WhiteningNormalizer` rejects non-finite data, fewer than `n_dims + 1` samples,
zero-variance coordinates and numerically collinear coordinates (no jitter or
dropped dimensions). The check is unit invariant, so coordinates with very
different scales are accepted. `logpdf` includes the fixed Jacobian
`-sum(log, diag(L))`, i.e. densities are in physical coordinates.
The normalizer is fitted on the first training call and reused on later calls
unless explicitly refitted.

### Initialization

`FlowDistribution` now defaults to `identity_init=true`: conditioner output
heads start at identity-transform parameters, while hidden layers remain random.
This avoids exponential amplification of tails by random affine layers before
training. Pass `identity_init=false` to reproduce the legacy random initialization.
Saved models retain their stored weights regardless of this constructor option.

### Validation, history and repeated training

```julia
history = Any[]
train_flow!(flow, train; n_epochs=500, validation_data=val, eval_every=10,
            history=history, rng=Random.Xoshiro(1))
history[end]   # (; epoch, train_nll, validation_nll, best_epoch)
```

With `validation_data` the weights with the lowest validation NLL (the initial
model included) are restored; NLLs are full-dataset physical-coordinate values
(normalizer Jacobian included). An already fitted normalizer is **reused** by
later `train_flow!` calls; changing kind or refitting requires
`refit_normalizer=true`, which changes the coordinates of the current weights.
Adam state restarts on every call (not an exact optimizer resume). See
`?train_flow!`.

### Reactant training (optional)

```julia
using Reactant, Enzyme          # activates the extension; Enzyme must be loaded too
train_flow!(flow, data; backend=:reactant, n_epochs=200, batch_size=256,
            validation_data=val, history=history)
```

Same API, batch order, history and CPU return value as the default Zygote
backend; weights and optimizer state live on the device during the call and the
final or best-validation weights are copied back. One scalar is synchronised per
step for the finite-value checks (conservative, no speed claim). See `?train_flow!`.

### Persistence guarantees

`save_trained_flow` writes `format_version`, `precision` (`Float32`/`Float64`),
the actual activation, the normalizer kind (`none`/`minmax`/`whiten`) and
optional `parameter_names`. Only a whitelist of built-in activations (e.g.
`gelu`, `relu`, `tanh`, `swish`, ...) can be saved; other callables are
rejected, and artifact strings are never evaluated. Artifacts written by older
versions (no `format_version`, or version 1) are loaded assuming GELU, with the
precision inferred from the weights. Unknown versions, activations or normalizer
kinds raise an `ArgumentError`.

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
├── flow_setup.json   # architecture, format_version, precision, activation,
│                     # normalizer kind, optional parameter_names
└── weights.npz       # flat parameter arrays and normalizer arrays (numpy-compatible)
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

## Reactant JIT Acceleration (Optional)

`SimpleFlows.jl` supports compiling density evaluation and input gradients on the GPU/TPU/CPU using the [Reactant.jl](https://github.com/JuliaScience/Reactant.jl) XLA compiler.

### Workflow: Compile and Evaluate a Trained Flow

```julia
using SimpleFlows
using Reactant
using Distributions: logpdf

# 1. Load your trained flow from CPU
flow_cpu = load_trained_flow("my_flow/")

# 2. Port the parameters and normalizer state to Reactant device memory
flow_react = to_reactant(flow_cpu)

# 3. Create input data on the device
x_cpu = randn(Float32, length(flow_cpu), 8)
x_react = Reactant.to_rarray(x_cpu)

# 4. Compile the logpdf evaluation function
compiled_logpdf = Reactant.@compile logpdf(flow_react, x_react)

# 5. Run JIT-compiled inference (near-instant execution after initial compilation compile-time)
log_probs = compiled_logpdf(flow_react, x_react)
```

For compiling reverse-mode input gradients on-device, `Enzyme.jl` can be used directly on the compiled Reactant code. See `test/test_reactant.jl` for reference.
