# 0.2.0

- Identity output-head initialization is now the default. This changes initial
  weights and training trajectories for a given seed; use `identity_init=false`
  for the legacy initialization. Hidden-layer random draws are unchanged.
- Training reuses a fitted normalizer. Explicit `refit_normalizer=true` is needed
  to refit or change kind; this changes the coordinates of the current weights.
- `verbose=true` now logs full-dataset physical-coordinate NLL, not a mini-batch
  average. The default evaluation interval is every epoch; choose `eval_every`
  explicitly for long runs. Validation/history also request full-data evaluation.
- Shuffle order uses the caller's `rng`. Each call creates fresh optimizer state;
  repeated calls are not exact optimizer resumes. Custom `opt` replaces clipping
  and Adam; explicit `lr` and `opt` together are rejected before mutation.
- Whitening, validation-selected weights, versioned deployment artifacts and
  optional Reactant/Enzyme CPU training are supported. The tested optional floor
  is Reactant 0.2.295 / Enzyme 0.13.216; examples use Julia 1.12.
- **Do not load 0.2 artifacts with SimpleFlows 0.1.** The old loader can silently
  discard whitening or misinterpret non-GELU activations, producing incorrect
  densities. Version 0.2 can read supported legacy artifacts; the converse is
  not guaranteed. Artifacts do not contain resumable optimizer state.
- Activations outside the persistence whitelist now raise an error rather than
  being mislabeled GELU. Non-finite or incorrectly shaped training data are
  rejected, and any delivered training exception restores the entry state.
- `WhiteningNormalizer` and `load_parameter_names` are new exports. The core
  Julia compatibility floor is 1.10; Turing integration is tested with 0.43 and
  Bijectors 0.15. ForwardDiff is test-only rather than a runtime dependency.
