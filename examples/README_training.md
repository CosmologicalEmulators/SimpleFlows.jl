# Three normalizing flows, one training recipe

[`train_all_flows_long.jl`](train_all_flows_long.jl) generates a reproducible
two-dimensional correlated, curved distribution. It trains **RealNVP, NSF and
MAF** independently, using identical training/validation/test draws. No scientific
datasets, private paths, Python, or plotting packages are required.

With Julia 1.12, from the repository root, prepare the dedicated local training
environment (separate from the legacy Turing examples):

```sh
julia --project="$(pwd)/examples/training" -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Start with the short end-to-end smoke run:

```sh
julia --project="$(pwd)/examples/training" examples/train_all_flows_long.jl --smoke --output /path/to/new/smoke-output
```

Run the full schedule:

```sh
julia --project="$(pwd)/examples/training" examples/train_all_flows_long.jl --output /path/to/new/full-output
```

The default is **Reactant + Enzyme on CPU**. Add `--zygote` for the ordinary Julia
training backend. GPU performance/support is not claimed by this example.
First-use Julia precompilation and Reactant compilation can dominate the short
smoke run; it is a correctness check, not a performance benchmark. The examples
environment declares minimum Reactant/Enzyme versions for this training recipe.

## What the code demonstrates

- Three transforms per flow, with two 32-unit hidden layers and GELU activation.
  NSF uses the package defaults K=8 and tail bound 3.
- Float32, whitening fitted only on training data, batch size 256.
- **10 stages × 1000 epochs = 10,000 epochs per architecture.** Each stage holds
  one rate; the ten rates are logarithmically spaced from 1e-3 to 1e-6. The
  full example has 16 batches per epoch, hence 160,000 updates per flow.
- One `train_flow!` call keeps Adam moments, weights, shuffle RNG and compiled
  caches continuous across stages. A small `Optimisers.AbstractRule`, passed
  through the public `opt` keyword, advances its counter dynamically on device.
  This is learning-rate decay, not weight decay, and does not restart training.
- Full-dataset physical training/validation NLL every ten epochs, including epoch
  zero. `history` is written to each flow's `losses.csv` after its training call.
- Global best-validation weights are restored by `train_flow!`, saved, reloaded,
  and checked against the original model. The independent test set is used only
  for the final reported score, never checkpoint selection.
- Drawing 2000 finite samples from the restored flow. Logged training-call time
  includes compilation and evaluation; it is not a prepared-update benchmark.

`--smoke` keeps all ten stages but uses two epochs per stage and smaller datasets.
The default output is `examples/training_output/` (gitignored); existing output
directories are never overwritten. No automatic early stopping is enabled:
the requested budget runs in full, but only the best-validation network is exported.
This example saves deployment artifacts, not resumable full optimizer checkpoints.
More epochs or capacity do not guarantee improved validation loss.
