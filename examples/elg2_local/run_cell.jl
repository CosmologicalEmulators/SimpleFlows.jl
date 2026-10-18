# One ELG2 validation cell = one fresh process:
#   architecture x normalization x backend x seed.
#
#   /home/marcobonici/.juliaup/bin/julia --project=<this dir> run_cell.jl \
#       --arch NSF --normalization whiten --backend zygote --seed 42 [--epochs 150] \
#       [--out-root DIR] [--smoke]
#
# `--identity-init true|false` (default true, the package default) selects the output-head
# identity initialization; false reproduces the legacy fully random initialization.
# Training and validation draws only are used for fitting and model selection
# (best *validation* weights are restored by train_flow!). The reference file is
# read once at the start, so the assessment draws are in memory from the
# beginning, but nothing touches them: the training/validation matrices are
# built by `training_split`, and *assessment evaluation* is deferred until the
# artifact is saved and a `training_complete.json` marker is written. They are
# never used for any choice.

ENV["GKSwstype"] = "100"        # headless GR for Plots

using SimpleFlows
using Reactant, Enzyme          # needed for backend=:reactant (activates the extension)
using Random, Statistics, LinearAlgebra, Printf, JSON, NPZ, Dates
using Distributions: logpdf    # SimpleFlows.FlowDistribution implements the Distributions.jl interface
import Zygote
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "plots.jl"))

# ── arguments ────────────────────────────────────────────────────────────────
function parse_args(argv)
    opts = Dict{String, Any}("arch" => nothing, "normalization" => nothing, "backend" => "zygote",
                             "seed" => nothing, "epochs" => 150, "out-root" => DEFAULT_OUT_ROOT,
                             "smoke" => false, "n-transforms" => 3, "hidden" => "32,32", "K" => 8,
                             "batch-size" => 256, "lr" => 1e-3, "eval-every" => 10, "n-samples" => 10000,
                             "identity-init" => "true")
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument $a")
        key = a[3:end]
        key in keys(opts) || error("unknown option --$key")
        if key == "smoke"
            opts[key] = true; i += 1
        else
            i < length(argv) || error("missing value for --$key")
            default = opts[key]
            v = argv[i + 1]
            opts[key] = default isa Int ? parse(Int, v) : default isa Float64 ? parse(Float64, v) : v
            i += 2
        end
    end
    for k in ("arch", "normalization", "seed")
        isnothing(opts[k]) && error("--$k is required")
    end
    opts["seed"] = opts["seed"] isa Integer ? opts["seed"] : parse(Int, opts["seed"])
    opts["arch"] in ("RealNVP", "NSF", "MAF") || error("--arch must be RealNVP, NSF or MAF")
    opts["normalization"] in ("minmax", "whiten") || error("--normalization must be minmax or whiten")
    opts["backend"] in ("zygote", "reactant") || error("--backend must be zygote or reactant")
    opts["identity-init"] in ("true", "false") || error("--identity-init must be true or false")
    return opts
end

# SHA-256 of every file that defines the model/training/runner code (reproducibility record).
function source_hashes()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    files = String[]
    for d in ("src", "ext"), f in sort(readdir(joinpath(root, d))); endswith(f, ".jl") && push!(files, joinpath(d, f)); end
    push!(files, "Project.toml")
    hashes = Dict{String, String}(f => file_sha256(joinpath(root, f)) for f in files)
    here = joinpath("examples", "elg2_local")
    for f in sort(readdir(@__DIR__)); (endswith(f, ".jl") || f in ("Project.toml", "Manifest.toml", "launch_cells.sh")) && (hashes[joinpath(here, f)] = file_sha256(joinpath(@__DIR__, f))); end
    return hashes
end

function main(argv)
    o = parse_args(argv)
    smoke = o["smoke"]
    o["backend"] == "reactant" && Reactant.set_default_backend("cpu")   # CPU PJRT only for now
    epochs = smoke && !any(==("--epochs"), argv) ? 2 : o["epochs"]
    n_samples = smoke && !any(==("--n-samples"), argv) ? 500 : o["n-samples"]
    o["n-samples"] = n_samples
    cell_id = @sprintf("%s_%s_%s_seed%d", o["arch"], o["normalization"], o["backend"], o["seed"])
    root = joinpath(o["out-root"], smoke ? "smoke" : "cells")
    mkpath(root)
    cell_dir = joinpath(root, cell_id)
    mkdir(cell_dir)             # fails if it exists: cells are never overwritten or resumed
    status_path = joinpath(cell_dir, "status.json")
    write_status(state; extra...) = open(status_path, "w") do io
        JSON.print(io, merge(Dict("state" => state, "cell" => cell_id, "smoke" => smoke, "pid" => getpid(),
                                  "updated" => stamp()), Dict{String, Any}(string(k) => v for (k, v) in extra)), 2)
    end

    config = Dict{String, Any}(
        "cell" => cell_id, "kind" => smoke ? "smoke" : "full", "architecture" => o["arch"],
        "normalization" => o["normalization"], "backend" => o["backend"], "seed" => o["seed"],
        "epochs" => epochs, "n_transforms" => o["n-transforms"],
        "hidden_layer_sizes" => parse.(Int, split(o["hidden"], ",")), "K" => o["K"], "tail_bound" => 3.0,
        "activation" => "gelu (FlowDistribution default)", "precision" => "Float32",
        "batch_size" => o["batch-size"], "lr" => o["lr"], "eval_every" => o["eval-every"],
        "n_generated_samples" => n_samples, "assessment_policy" => "reference file read upfront; assessment draws unused for fitting/selection, evaluated only after training_complete.json", "optimizer" => "ClipGrad(1) + Adam (train_flow! default)",
        "julia_version" => string(VERSION), "threads" => Threads.nthreads(),
        "versions" => Dict("SimpleFlows" => string(pkgversion(SimpleFlows)), "Reactant" => string(pkgversion(Reactant)),
                           "Enzyme" => string(pkgversion(Enzyme))),
        "identity_init" => o["identity-init"] == "true",
        "identity_init_note" => "true: package default (output heads start as the identity, hidden layers random); false: legacy random initialization. Training provenance only; artifacts store the weights.",
        "source_sha256" => source_hashes(),
        "started" => stamp(), "data_dir" => DATA_DIR,
    )
    write_status("running"; phase="start")
    try
        run_cell(o, config, cell_dir, write_status, smoke, epochs)
        write_status("complete"; phase="done")
        write(joinpath(cell_dir, "COMPLETE"), stamp() * "\n")
        logmsg("CELL COMPLETE ", cell_id)
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        logmsg("CELL FAILED ", cell_id, ": ", first(split(msg, '\n')))
        write_status("failed"; error=msg)
        write(joinpath(cell_dir, "FAILED"), msg)
        exit(1)
    end
end

function run_cell(o, config, cell_dir, write_status, smoke, epochs)
    cell_id = config["cell"]
    T = Float32
    logmsg("cell ", cell_id, smoke ? "  [SMOKE RUN - not a validation result]" : "", "  epochs=", epochs)

    # ── data (training + validation only) ───────────────────────────────────
    ref = load_reference()
    logmsg("reference verified: sha256 ok, names ok, shape ", size(ref.samples), ", ", sum(ref.flags), " divergent draws retained")
    sp = training_split(ref; smoke)
    config["data_hashes"] = ref.hashes
    config["split"] = Dict("training_chains" => collect(TRAIN_CHAINS), "validation_chains" => collect(VAL_CHAINS),
                           "assessment_chains" => collect(ASSESS_CHAINS), "n_train" => size(sp.train, 2),
                           "n_validation" => size(sp.val, 2), "train_divergent" => sum(sp.train_flags),
                           "validation_divergent" => sum(sp.val_flags),
                           "divergences_per_chain" => EXPECTED_DIVERGENCES_PER_CHAIN,
                           "smoke_subsample" => smoke)
    config["parameter_names"] = EFT_NAMES
    logmsg("train ", size(sp.train), " (", sum(sp.train_flags), " divergent), validation ", size(sp.val), " (", sum(sp.val_flags), " divergent)")
    open(joinpath(cell_dir, "config.json"), "w") do io; JSON.print(io, config, 2); end

    # ── model and training ───────────────────────────────────────────────────
    kw = (; architecture=Symbol(o["arch"]), n_transforms=o["n-transforms"], dist_dims=8, identity_init=(o["identity-init"] == "true"),
          hidden_layer_sizes=config["hidden_layer_sizes"], rng=Random.Xoshiro(o["seed"]),
          parameter_names=EFT_NAMES)
    flow = o["arch"] == "NSF" ? FlowDistribution(T; kw..., K=o["K"], tail_bound=3.0) : FlowDistribution(T; kw...)
    history = Any[]
    write_status("running"; phase="training")
    logmsg("training start (backend=", o["backend"], ", normalization=", o["normalization"], ")")
    t0 = time_ns()
    train_flow!(flow, sp.train; n_epochs=epochs, lr=o["lr"], batch_size=o["batch-size"], verbose=true,
                normalization=Symbol(o["normalization"]), validation_data=sp.val, history,
                eval_every=o["eval-every"], rng=Random.Xoshiro(o["seed"] + 1), backend=Symbol(o["backend"]))
    # Wall time of the whole call including any compilation; not a performance benchmark.
    config["training_wall_seconds_including_compile"] = (time_ns() - t0) / 1e9
    best = history[end].best_epoch
    logmsg("training done; best validation epoch ", best, ", final history entry ", history[end])
    open(joinpath(cell_dir, "history.json"), "w") do io; JSON.print(io, json_safe.(history), 2); end

    # ── artifact (package API) + reload check ────────────────────────────────
    art = joinpath(cell_dir, "flow")
    save_trained_flow(art, flow; parameter_names=EFT_NAMES)
    flow2 = load_trained_flow(art)
    probe = T.(sp.val[:, 1:min(64, size(sp.val, 2))])      # deployment dtype, cast BEFORE any logpdf
    lp1, lp2 = logpdf(flow, probe), logpdf(flow2, probe)
    s1 = rand(Random.Xoshiro(7), flow, 16); s2 = rand(Random.Xoshiro(7), flow2, 16)
    reload = Dict("logpdf_equal" => lp1 == lp2, "same_seed_samples_equal" => s1 == s2,
                  "parameter_names_preserved" => flow2.parameter_names == EFT_NAMES,
                  "normalizer_kind" => string(nameof(typeof(flow2.normalizer))))
    reload["logpdf_equal"] && reload["same_seed_samples_equal"] && reload["parameter_names_preserved"] ||
        error("save/load check failed: $reload")
    eltype(probe) == T || error("probe dtype changed")
    npzwrite(joinpath(cell_dir, "reload_probe.npz"), Dict("x" => probe, "logpdf" => Float64.(lp1)))
    config["reload_check"] = reload
    logmsg("artifact saved to ", art, " and reloaded: ", reload)
    open(joinpath(cell_dir, "training_complete.json"), "w") do io
        JSON.print(io, Dict("completed" => stamp(), "best_epoch" => best, "history_entries" => length(history)), 2)
    end

    # ── assessment evaluation (deferred until now); never used for selection ─
    write_status("running"; phase="assessment")
    logmsg("assessment phase (held-out chains ", ASSESS_CHAINS, ")")
    asm = assessment_split(ref; smoke)
    results = assess(flow2, sp, asm, history, config, o, cell_dir)
    results["cell"] = cell_id; results["kind"] = smoke ? "smoke" : "full"
    gen_samples = pop!(results, "_samples")
    open(joinpath(cell_dir, "results.json"), "w") do io; JSON.print(io, json_safe(results), 2); end
    open(joinpath(cell_dir, "config.json"), "w") do io; JSON.print(io, config, 2); end

    logmsg("writing plots")
    make_plots(cell_dir, results, history, asm.assess, gen_samples; title=(smoke ? "SMOKE " : "") * cell_id)
    return nothing
end

# Non-finite density values or generated samples mean the fitted density is not valid:
# record the counts and fail the cell instead of assessing a finite subset.
function scientific_failure(cell_dir, kind, counts)
    open(joinpath(cell_dir, "scientific_failure.json"), "w") do io
        JSON.print(io, merge(Dict("kind" => kind), counts), 2)
    end
    error("SCIENTIFIC FAILURE: $kind $(counts)")
end

function assess(flow, sp, asm, history, config, o, cell_dir)
    n_gen = o["n-samples"]
    T = Float32
    base = GaussianBaseline(sp.train)
    A = Float64.(asm.assess)
    nd = .!asm.assess_flags
    lp = Float64.(logpdf(flow, T.(asm.assess)))
    nbad = count(!isfinite, lp)
    nbad == 0 || scientific_failure(cell_dir, "nonfinite_assessment_logpdf",
                                     Dict("n_nonfinite" => nbad, "n_assessment" => length(lp),
                                          "n_nonfinite_among_divergent_draws" => count(!isfinite, lp[asm.assess_flags])))
    lpg = gaussian_logpdf(base, A)
    nll(v) = -mean(v)
    per_chain = Dict(string(c) => Dict("n" => count(==(c), asm.assess_chain),
                                       "divergent" => sum(asm.assess_flags[asm.assess_chain .== c]),
                                       "flow_nll" => nll(lp[asm.assess_chain .== c]),
                                       "gaussian_nll" => nll(lpg[asm.assess_chain .== c])) for c in ASSESS_CHAINS)
    heldout = Dict(
        "assessment_flow_nll_all_draws" => nll(lp), "assessment_gaussian_nll_all_draws" => nll(lpg),
        "assessment_flow_minus_gaussian_nll_all_draws" => nll(lp) - nll(lpg),
        "assessment_flow_nll_nondivergent" => nll(lp[nd]), "assessment_gaussian_nll_nondivergent" => nll(lpg[nd]),
        "n_assessment" => length(lp), "n_assessment_divergent" => sum(asm.assess_flags),
        "per_chain" => per_chain,
        "validation_flow_nll_best" => history[end].best_epoch === nothing ? nothing :
            history[findfirst(h -> h.epoch == history[end].best_epoch, history)].validation_nll,
        "validation_gaussian_nll" => nll(gaussian_logpdf(base, sp.val)),
        "train_gaussian_nll" => nll(gaussian_logpdf(base, sp.train)),
        "nondivergent_note" => "The *_nondivergent entries are supplemental; primary metrics use all draws, divergent NUTS draws included.",
        "note" => "Marginal EFT density of the NUTS reference; the Gaussian is a separate diagnostic baseline fitted on the training chains. These NLLs are not comparable to the joint cosmology+EFT log posterior.",
    )
    logmsg("assessment NLL: flow ", heldout["assessment_flow_nll_all_draws"], "  gaussian ", heldout["assessment_gaussian_nll_all_draws"])

    # Generated samples (same seed for every cell of a given seed).
    gen = Float64.(rand(Random.Xoshiro(o["seed"] + 1000), flow, n_gen))
    finite = vec(all(isfinite, gen; dims=1))
    logmsg("generated ", n_gen, " samples, ", count(!, finite), " non-finite")
    all(finite) || scientific_failure(cell_dir, "nonfinite_generated_samples",
                                       Dict("n_nonfinite" => count(!, finite), "n_generated" => n_gen))
    gen_f = gen
    marg = marginal_summary(gen_f, A)
    covs = covariance_summary(gen_f, A)

    # Joint two-sample diagnostics in train-whitened coordinates.
    zg, za = whiten(base, gen_f), whiten(base, A)
    ztrain = whiten(base, sp.train); zval = whiten(base, sp.val)
    sw_seed = 20261017
    gauss_samples = whiten(base, rand(Random.Xoshiro(o["seed"] + 2000), base, n_gen))
    sw = Dict(
        "flow_vs_assessment" => sliced_wasserstein(zg, za; seed=sw_seed),
        "gaussian_baseline_vs_assessment" => sliced_wasserstein(gauss_samples, za; seed=sw_seed),
        "training_chains_vs_assessment" => sliced_wasserstein(ztrain, za; seed=sw_seed),
        "validation_chains_vs_assessment" => sliced_wasserstein(zval, za; seed=sw_seed),
        "algorithm" => "Monte-Carlo sliced W2 in coordinates z = L^-1 (x - mean) with L the Cholesky factor of the training-chain covariance; 256 random unit directions (Xoshiro seed $sw_seed), 2000 matched quantiles per direction. Estimate with Monte-Carlo/finite-sample error; NUTS draws are autocorrelated, so the training/validation-chain rows give the reference noise level between independent NUTS chain groups.",
        "n_projections" => 256, "n_quantiles" => 2000, "seed" => sw_seed)
    ks_all = marg["ks"]
    results = Dict{String, Any}(
        "heldout" => heldout, "marginals" => marg, "covariance" => covs, "sliced_wasserstein_train_whitened" => sw,
        "n_generated" => n_gen, "n_generated_nonfinite" => 0,
        "ks_note" => "Two-sample KS statistic of generated samples against the pooled assessment NUTS draws (no p-values: NUTS draws are autocorrelated).",
        "reference_for_standardization" => "assessment NUTS draws (chains 14-16)",
        "selection" => "best validation NLL epoch only; assessment evaluation was deferred until after training and nothing was selected with it",
        "best_epoch" => history[end].best_epoch,
        "final_train_nll" => history[end].train_nll, "final_validation_nll" => history[end].validation_nll,
        "max_ks" => maximum(ks_all), "_samples" => gen_f)
    return results
end

main(ARGS)
