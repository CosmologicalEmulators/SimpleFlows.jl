# SimpleFlows.jl

Train RealNVP, neural spline flows (NSF), and masked autoregressive flows (MAF),
then use their `logpdf`, `rand`, and `length` methods through Distributions.jl.
Flows also support Bijectors.jl's unconstrained-prior interface for Turing.jl.
Integration is tested with Turing 0.43 and Bijectors 0.15; compatibility with newer
Turing/Bijectors major or minor API changes is not claimed.

Training uses mini-batches with Zygote by default. The default optimizer is
`OptimiserChain(ClipGrad(1), Adam(lr))`: elementwise gradient clipping followed
by Adam. Optional Reactant/Enzyme **CPU** training uses the same public API.
Normalization is fitted on training data only: min-max by default, or whitening
with a mean and Cholesky covariance factor. Reported densities include its Jacobian.

## Setup

Julia 1.10 or later is supported by the package. The optional Reactant examples
below were validated on Julia 1.12.6, Reactant 0.2.295 and Enzyme 0.13.216.
From the repository root, with Julia 1.12, prepare the local example environment:

```sh
julia --project="$(pwd)/examples/training" -e 'using Pkg; Pkg.instantiate()'
julia --project="$(pwd)/examples/training"
```

The Julia blocks below are independent examples. The Turing section uses the
separate environment described there. Short training budgets demonstrate the API,
not convergence or a guaranteed fit.
Float64 inputs to Float32 weights use Lux's generic mixed-precision matrix
multiply, which can be slower than matched-precision BLAS. Match precision for
throughput; Float64 inference inputs remain a supported gradient-test case.
Blocks 1–3 need only the core package and Distributions.jl and also work on Julia
1.10/1.11; the setup above is for the optional examples on Julia 1.12.

## Quick start: samples are columns

```julia
using SimpleFlows, Distributions, Random

data = randn(Xoshiro(1), Float32, 4, 256) # (dimensions, samples)
flow = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=2,
    dist_dims=4, hidden_layer_sizes=[16, 16], rng=Xoshiro(2))
train_flow!(flow, data; n_epochs=5, lr=1f-3, verbose=false, rng=Xoshiro(3))
logpdf(flow, zeros(Float32, 4))
samples = rand(Xoshiro(4), flow, 16)     # 4 × 16

artifact = joinpath(mktempdir(), "flow")
save_trained_flow(artifact, flow)
restored = load_trained_flow(artifact; rng=Xoshiro(5))
@assert logpdf(restored, data) == logpdf(flow, data)
```

## Whitening and parameter names

```julia
using SimpleFlows, Distributions, Random

data = randn(Xoshiro(1), Float64, 3, 96)
flow = FlowDistribution(Float64; architecture=:NSF, n_transforms=2,
    dist_dims=3, hidden_layer_sizes=[8, 8], rng=Xoshiro(2),
    parameter_names=["ω_m", "h", "b1"])
train_flow!(flow, data; normalization=:whiten, n_epochs=2,
    verbose=false, rng=Xoshiro(3))
artifact = joinpath(mktempdir(), "whitened_flow")
save_trained_flow(artifact, flow)
restored = load_trained_flow(artifact; rng=Xoshiro(4))
@assert restored.parameter_names == ["ω_m", "h", "b1"]
```

Whitening uses `Σ = L Lᵀ`, `z = L \ (x - μ)` and a density correction
`-sum(log, diag(L))`. It rejects non-finite data, fewer than `n_dims + 1`
samples, zero-variance coordinates and numerical collinearity, without adding
jitter or dropping coordinates. Its conditioning check is invariant to coordinate units.

`identity_init=true` is the constructor default. Conditioner output heads start
at identity-transform parameters; hidden layers remain random. This avoids
exponential tail amplification before training. Use `identity_init=false` for
legacy random initialization. Loading always preserves stored weights.

## Validation, history and repeated training

```julia
using SimpleFlows, Random

train = randn(Xoshiro(1), Float32, 2, 128)
validation = randn(Xoshiro(2), Float32, 2, 64)
flow = FlowDistribution(Float32; architecture=:MAF, n_transforms=2,
    dist_dims=2, hidden_layer_sizes=[8, 8], rng=Xoshiro(3))
history = Any[]
train_flow!(flow, train; n_epochs=3, validation_data=validation,
    eval_every=1, history, verbose=false, rng=Xoshiro(4))
history[end] # (; epoch, train_nll, validation_nll, best_epoch)
@assert length(history) == 4
```

With validation data, the best weights among evaluated epochs are restored, including
the initial model as a candidate. History contains full-dataset physical NLLs,
not mini-batch losses; its final entry describes the last epoch before restoration.
`verbose=true` also requests full-data evaluation, every epoch by default. Use
`eval_every` to control that cost. With no validation, history or logging, no
extra full-data evaluation is performed.

A fitted normalizer is reused on later calls. Changing kind or refitting requires
`refit_normalizer=true`, which changes the current weights' coordinates.
Optimizer state restarts on each call. A custom `opt` replaces the **whole**
default chain, including clipping; specify its rate inside the rule, not via `lr`.
Training exceptions restore entry weights and normalizer, not the best-so-far
weights. This includes Ctrl-C when delivered as an exception in an interactive
session, not forcible process termination. Deployment artifacts are not
optimizer-resume checkpoints.

## Optional Reactant/Enzyme CPU training

Use the dedicated environment from Setup. GPU/TPU training is not certified here.

```julia
using SimpleFlows, Random, Reactant, Enzyme

Reactant.set_default_backend("cpu")
train = randn(Xoshiro(1), Float32, 2, 64)
validation = randn(Xoshiro(2), Float32, 2, 32)
flow = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=2,
    dist_dims=2, hidden_layer_sizes=[8, 8], rng=Xoshiro(3))
history = Any[]
train_flow!(flow, train; backend=:reactant, n_epochs=2, batch_size=32,
    validation_data=validation, history, verbose=false, rng=Xoshiro(4))
@assert length(history) == 3
```

The backends share batch ordering, history semantics, and a CPU flow return value.
Numerics are close, not promised bitwise identical across backends. Parameters
and optimizer state remain on-device during the call; final or best-validation
weights are copied back. A scalar finite-value check synchronizes every update.
First-use compilation can dominate short runs, and a new call creates a new
compiled trainer. Prefer one continuous call for long schedules; no speedup is
claimed from the smoke examples.

## Compiled density evaluation

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) can compile flow evaluation.
This independent CPU example fits only the normalizer (`n_epochs=0`):

```julia
using SimpleFlows, Distributions, Random, Reactant

Reactant.set_default_backend("cpu")
flow = FlowDistribution(Float32; architecture=:RealNVP, n_transforms=2,
    dist_dims=2, hidden_layer_sizes=[8, 8], rng=Xoshiro(1))
train_flow!(flow, randn(Xoshiro(2), Float32, 2, 32);
    n_epochs=0, verbose=false)
device_flow = to_reactant(flow)
device_x = Reactant.to_rarray(randn(Xoshiro(3), Float32, 2, 8))
compiled_logpdf = Reactant.@compile sync=true logpdf(device_flow, device_x)
host_logpdf = Array(compiled_logpdf(device_flow, device_x))
@assert length(host_logpdf) == 8
```

Compilation specializes to input shapes and types; keep them fixed when reusing
the callable. Results are device arrays; `Array` transfers them to the host.
Reverse-mode input-gradient examples are in `test/test_reactant.jl`.

## Turing.jl integration

With Julia 1.12, prepare and activate the Turing examples environment first:

```sh
julia --project="$(pwd)/examples" -e 'using Pkg; Pkg.instantiate()'
julia --project="$(pwd)/examples"
```

```julia
using SimpleFlows, Turing, Distributions, Random

prior = FlowDistribution(Float64; architecture=:RealNVP, n_transforms=2,
    dist_dims=2, hidden_layer_sizes=[8, 8], rng=Xoshiro(1))
train_flow!(prior, randn(Xoshiro(2), 2, 128);
    n_epochs=2, verbose=false, rng=Xoshiro(3))
@model function flow_prior_model(y, prior)
    θ ~ prior
    y ~ Normal(θ[1], 0.5)
end
model = flow_prior_model(0.5, prior)
chain = sample(Xoshiro(4), model, NUTS(10, 0.8), 20; progress=false)
@assert size(chain, 1) == 20
```

This tiny chain exercises sampling and AD; it is not a convergence assessment.

## Persistence

`save_trained_flow` writes `flow_setup.json` and `weights.npz`. Metadata records
format version, precision, activation, normalizer kind and optional parameter
names. NPZ arrays can be read by NumPy. Only whitelisted built-in activations
(including GELU, ReLU, tanh and swish) are accepted; artifact strings are never
evaluated. Legacy artifacts without a version, or version 1, assume GELU and
infer precision from weights. Unknown versions, activations or normalizer kinds
raise `ArgumentError`.

## Longer training and examples

[The three-flow synthetic recipe](examples/README_training.md) trains all three
architectures with ten staged learning rates in one continuous call. It includes
smoke and 10,000-epoch modes, validation-selected saving, loss CSVs and sampling.
More epochs or larger networks do not guarantee better validation loss.

The following longer Turing demonstrations use the Turing environment above:

```sh
julia --project="$(pwd)/examples" -t 4 examples/train_multinormal_nvp.jl
julia --project="$(pwd)/examples" -t 4 examples/train_multinormal_nsf.jl
julia --project="$(pwd)/examples" -t 4 examples/train_multinormal_maf.jl
```

## Tests

```sh
julia --project="$(pwd)" -e 'using Pkg; Pkg.test()'
```
