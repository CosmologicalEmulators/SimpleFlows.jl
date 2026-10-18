# SimpleFlows: ELG2 EFT validation

## Acceptance and scope

The repaired package passed **1322/1322 tests** (`validation_logs/identity_branch_independent_full.log`). The final local comparison completed **24/24 cells**, and a separate process loaded and checked all 24 artifacts (`validation_logs/elg2_identity_aggregate.log`). No cluster runs or production inference changes were made. Referenced validation logs and scientific artifacts are retained locally, not included in the repository.

An additional independent check on actual final fitted weights passed **15/15 checks**: Zygote weight gradients against four ForwardDiff directions per architecture, promoting the stored Float32 weights to Float64 for the numerical reference (`validation_logs/independent_identity_trained_eft_gradients.log`). This supplements, rather than replaces, the suite's Float32/64 compiled Enzyme gradient and matched-update checks.

Reference: 16 NUTS chains × 2000 retained draws, eight EFT coordinates from the 13-dimensional ELG2 posterior. All 50 divergence-flagged draws are retained. Chains 1–10 supply 20,000 training draws; 11–13 supply 6000 validation draws; 14–16 supply 6000 assessment draws. Training/validation/assessment contain 16/0/34 flagged draws respectively. This learns an **EFT marginal**, not a cosmology-conditioned EFT density or the full cosmology+EFT posterior.

Each cell: Float32, three transforms, hidden widths [32,32], batch 256, 150 epochs, clipped Adam with learning rate 0.001, validation every ten epochs, and restoration of the best validation weights. Architectures: RealNVP, NSF, MAF. Normalizers: minmax and whitening. Training backends: Zygote and Reactant/Enzyme. Seeds: 42 and 43. Final initialization is identity, with random hidden layers retained. Per-cell configuration includes source hashes and input-file hashes.

## Accuracy

### Post-review correctness gate

Independent Computer Scientist review (GPT-6 Astra, High reasoning) identified
two additional correctness gaps after the original campaign: exact NSF tail
endpoint input gradients, and stale density-normalization constants when reusing
a compiled function with a replacement fitted normalizer. Both are corrected.
Exact endpoints now use the identity tail; device-backed normalizers derive their
Jacobian from dynamic arrays, while ordinary CPU evaluation retains the cached
scalar. The analogous inherited min-max compiled-reuse issue is also covered.

The repaired tree passed **1498/1498 tests** in a full `Pkg.test()` run
(`validation_logs/pr3_corrections_full.log`). Independent reviewer reproductions
verified endpoint scores across AD backends, zero density error in the factor-two
normalizer-replacement control, 21/21 named dynamic-flow checks, and no accessor
dispatch ambiguities. The earlier 24-fit and benchmark results below describe
the original campaign; those campaigns were not rerun for these corrections.

Within each architecture, the following cell is selected by **validation NLL only**, not assessment accuracy. All selected cells use whitening. NLL is mean negative log-density in physical EFT coordinates, including the normalizer Jacobian; lower is better.

| Architecture | Selected backend / seed | Validation NLL | Assessment NLL | Maximum marginal KS | Maximum absolute mean shift / reference SD | SD ratio range | Maximum correlation difference |
|---|---|---:|---:|---:|---:|---:|---:|
| RealNVP | Reactant / 42 | 5.4739 | 5.4065 | 0.0403 | 0.0770 | 1.0127–1.0382 | 0.0901 |
| NSF | Reactant / 43 | 6.0950 | 5.7906 | 0.0370 | 0.0770 | 0.9985–1.1281 | 0.0711 |
| MAF | Zygote / 42 | 5.3604 | 5.2986 | 0.0366 | 0.0364 | 1.0046–1.0602 | 0.0747 |

The train-fitted full-covariance Gaussian has validation NLL 9.5548 and assessment NLL 9.1671. MAF with whitening is the strongest likelihood fit **within this bounded experiment**, not a universal architecture ranking. RealNVP/MAF whitening runs agree between training backends to the displayed precision; tiny differences are not meaningful backend rankings. Minmax trajectories show larger backend-dependent optimization differences despite independently checked local gradient/update parity.

Sliced W2 in training-whitened coordinates is 0.0770 / 0.0767 / 0.0798 for the three selected flows, compared with 0.1268 for the Gaussian. Training-chain versus assessment-chain W2 is 0.0672, and validation-chain versus assessment-chain W2 is 0.1127. These are finite-sample, autocorrelated-chain comparisons, **not rigorous error bars or equivalence certification**. NLL and W2 need not rank models identically. Marginal and triangle plots show the learned nonlinear structure; residual width and correlation differences remain, so “exact posterior replacement” is not supported.

An unbounded normalizing flow does not enforce hard prior support automatically. Use of these artifacts as proposals or surrogates still requires the original target/support and independent inference validation. Do not treat an EFT marginal fit as a new prior in a cosmology analysis without accounting for how it was obtained.

## Failures found and fixed

1. Persistence previously lost activation/precision/parameter-name information. Round-trip and fresh-process tests now cover all architectures and fitted normalizers; legacy artifacts remain supported with explicit validation.
2. Whitening conditioning must be assessed in unit-invariant correlation coordinates, not by rejecting a covariance merely because coordinates have different physical units.
3. Training must reject non-finite final updates, restore state on failure, report full physical-coordinate NLL, and restore the best validation checkpoint.
4. An absolute epsilon in the NSF inverse quadratic root caused approximately 1.37% log-Jacobian round-trip inconsistency on valid narrow bins. Stable root evaluation and the exact positive Jacobian removed this bias; the original parent reproduction improved to about 7.4e-8. Independent 256-bit bisection references cover conditioning and edge cases.
5. The original random RealNVP initialization amplified valid whitened tail samples from maximum magnitude 11 to 7.9e25; the Float32 Gaussian square overflowed before training. All four original RealNVP+whitening cells failed across both seeds/backends. Identity output-head initialization fixes this without clipping/filtering data. `identity_init=false` preserves legacy random initialization.
6. Exact identity exposed a compiled inverse-root adjoint failure: selecting between quotients yielded NaNs at a=0. The parent reduced this to a two-row quadratic expression and fixed it by selecting numerator/denominator **before one division**. Unconditional Float32/64, forward/inverse, identity/nonidentity tests compare Enzyme, Zygote and ForwardDiff at smooth points, retaining separate exact-knot checks. At perturbed knots, left/right derivatives differ; this was independently demonstrated rather than concealed by relaxed tolerances.

The original matrix and four failed cells remain in `/home/marcobonici/Desktop/work/simpleflows_elg2_validation_20261017`. Final artifacts, full tables, plots and provenance are in `/home/marcobonici/Desktop/work/simpleflows_elg2_validation_20261017_identity`.

## Controlled CPU performance

All **12 ABBA measurements completed**, using the same saved `ARCH_whiten_zygote_seed42` weights for both backends. Hardware: Intel i7-13700H, pinned physical cores 0 and 2, two Julia threads, one BLAS thread; Float32, batch 256. Each measurement follows explicit warm-up and cooling to at most 65°C, with 50 BenchmarkTools samples and `evals=1`. Each timed training step starts with restored weights and fresh optimizer state while retaining compiled caches. The timed region includes post-update validity checks and synchronization. Reset reproducibility and compiled-cache retention are checked, not assumed.

The table gives the **range of per-process minima** across the two runs of each backend; these are hot short-burst measurements, not sustained throughput.

| Architecture | Zygote training step (ms) | Reactant training step (ms) | Matched minimum-time speed ratio | Additional first-step compile/setup: Zygote / Reactant (s) |
|---|---:|---:|---:|---:|
| RealNVP | 0.989–1.002 | 0.400–0.401 | 2.47–2.50× | 1.25–1.27 / 52.47–52.90 |
| NSF | 4.434–4.460 | 1.964–1.979 | 2.25–2.26× | 1.23–1.24 / 78.03–78.11 |
| MAF | 0.697–1.012 | 0.417–0.425 | 1.67–2.38× | 1.19 / 48.96–49.02 |

Per-process median step times were 1.011–1.026 vs 0.481–0.483 ms for RealNVP; 4.612–4.640 vs 2.046–2.063 ms for NSF; and 0.706–1.061 vs 0.480–0.483 ms for MAF. **MAF's Zygote repetitions vary substantially**, despite similar sampled temperatures; do not turn that range into a precise universal speedup. Training-step temperatures before/after measurement were 54–59°C. One separate NSF input-gradient measurement ended at 87°C; that diagnostic is not used for the training-speed comparison, and the process cooled again before measuring training.

The first-step numbers are incremental after forward/input-gradient warm-up, not clean-process startup totals. Compilation costs matter: 150 epochs with 20,000 samples and batch 256 require 11,850 updates. At these sizes, the measured hot-step savings alone do **not** amortize Reactant's extra compilation cost. Thus Reactant is faster per prepared update here, but this does not establish faster short end-to-end training. Larger models, longer training, compilation reuse, and GPUs require their own measurements.

An initial campaign using BenchmarkTools' longer default bursts was stopped when a run reached 90°C. Its partial results remain in `benchmarks/` and are excluded. Accepted, consistently configured ABBA results are in `benchmarks_short_burst/`. Operational fitting wall times are not substituted for these measurements. No GPU performance claim is made.
