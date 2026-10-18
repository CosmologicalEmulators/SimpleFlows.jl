# Aggregate all completed cells under <out-root>/cells (or smoke/ with --smoke).
# One row per cell: seeds are never averaged or blended into one "model".
# Ranking is by *validation* NLL only; assessment columns are reported alongside.
#
#   julia --project=<this dir> aggregate.jl [--out-root DIR] [--smoke]
#
# Run as a fresh process: every artifact is reloaded with load_trained_flow and
# its physical log-density is compared with the values stored at save time
# (same Float32 probe inputs). Exit status: 0 only if there is at least one complete
# cell, no FAILED cell, and every reload check matched; otherwise non-zero (the
# summary files are still written first).

ENV["GKSwstype"] = "100"
using SimpleFlows, JSON, NPZ, Printf, Statistics
using Distributions: logpdf
include(joinpath(@__DIR__, "common.jl"))

function main(argv)
    root = DEFAULT_OUT_ROOT; smoke = false
    i = 1
    while i <= length(argv)
        if argv[i] == "--out-root"; root = argv[i + 1]; i += 2
        elseif argv[i] == "--smoke"; smoke = true; i += 1
        else error("unknown argument $(argv[i])") end
    end
    dir = joinpath(root, smoke ? "smoke" : "cells")
    isdir(dir) || error("no such directory $dir")
    rows = Dict{String, Any}[]
    incomplete = String[]; reload_failures = String[]
    for cell in sort(readdir(dir))
        cd_ = joinpath(dir, cell)
        isdir(cd_) || continue
        if !isfile(joinpath(cd_, "COMPLETE"))
            push!(incomplete, cell * (isfile(joinpath(cd_, "FAILED")) ? " (FAILED)" : " (not complete)"))
            continue
        end
        res = JSON.parsefile(joinpath(cd_, "results.json")); cfg = JSON.parsefile(joinpath(cd_, "config.json"))
        # fresh-process reload check
        flow = load_trained_flow(joinpath(cd_, "flow"))
        probe = npzread(joinpath(cd_, "reload_probe.npz"))
        eltype(probe["x"]) == Float32 || error("reload probe of $cell is not Float32 ($(eltype(probe["x"])))")
        reload_ok = Float64.(logpdf(flow, probe["x"])) == probe["logpdf"] && flow.parameter_names == EFT_NAMES
        reload_ok || push!(reload_failures, cell)
        h = res["heldout"]; sw = res["sliced_wasserstein_train_whitened"]
        push!(rows, Dict{String, Any}(
            "cell" => cell, "kind" => res["kind"], "architecture" => cfg["architecture"],
            "normalization" => cfg["normalization"], "backend" => cfg["backend"], "seed" => cfg["seed"],
            "epochs" => cfg["epochs"], "identity_init" => get(cfg, "identity_init", "legacy-unrecorded"), "best_epoch" => res["best_epoch"],
            "val_nll_best" => h["validation_flow_nll_best"], "val_gaussian_nll" => h["validation_gaussian_nll"],
            "assess_flow_nll" => h["assessment_flow_nll_all_draws"], "assess_gaussian_nll" => h["assessment_gaussian_nll_all_draws"],
            "assess_flow_minus_gaussian" => h["assessment_flow_minus_gaussian_nll_all_draws"],
            "assess_flow_nll_nondivergent" => h["assessment_flow_nll_nondivergent"],
            "max_abs_mean_shift_sd" => maximum(abs, res["marginals"]["mean_shift_over_ref_sd"]),
            "sd_ratio_min" => minimum(res["marginals"]["sd_ratio"]), "sd_ratio_max" => maximum(res["marginals"]["sd_ratio"]),
            "max_ks" => res["max_ks"], "max_abs_corr_diff" => res["covariance"]["max_abs_correlation_difference"],
            "sw2_flow" => sw["flow_vs_assessment"], "sw2_gaussian" => sw["gaussian_baseline_vs_assessment"],
            "sw2_train_vs_assess" => sw["training_chains_vs_assessment"], "sw2_val_vs_assess" => sw["validation_chains_vs_assessment"],
            "nonfinite_samples" => res["n_generated_nonfinite"], "train_wall_s_incl_compile" => cfg["training_wall_seconds_including_compile"],
            "fresh_process_reload_ok" => reload_ok))
    end
    sort!(rows; by=r -> (r["val_nll_best"] === nothing ? Inf : r["val_nll_best"]))   # validation only
    out = joinpath(root, smoke ? "smoke_summary" : "summary"); mkpath(out)
    open(joinpath(out, "summary.json"), "w") do io; JSON.print(io, (; rows, incomplete), 2); end
    cols = ["cell", "identity_init", "val_nll_best", "assess_flow_nll", "assess_gaussian_nll", "max_abs_mean_shift_sd", "sd_ratio_min", "sd_ratio_max",
            "max_ks", "max_abs_corr_diff", "sw2_flow", "sw2_gaussian", "sw2_val_vs_assess", "fresh_process_reload_ok"]
    fmt(v) = v isa AbstractFloat ? @sprintf("%.4g", v) : string(v)
    open(joinpath(out, "summary.md"), "w") do io
        println(io, "# ELG2 local validation summary", smoke ? " (SMOKE RUNS - not results)" : "")
        println(io, "\nOne row per cell (seeds are separate models, never averaged). Sorted by validation NLL; assessment columns are reported, not used for ranking.")
        println(io, "\n| ", join(cols, " | "), " |\n|", repeat("---|", length(cols)))
        for r in rows; println(io, "| ", join((fmt(r[c]) for c in cols), " | "), " |"); end
        isempty(incomplete) || println(io, "\nIncomplete or failed cells: ", join(incomplete, ", "))
    end
    println(read(joinpath(out, "summary.md"), String))
    logmsg("wrote ", out)
    nfailed = count(endswith("(FAILED)"), incomplete)
    isempty(rows) && (logmsg("ERROR: no complete cells"); exit(1))
    isempty(reload_failures) || (logmsg("ERROR: fresh-process reload mismatch: ", join(reload_failures, ", ")); exit(3))
    nfailed == 0 || (logmsg("ERROR: ", nfailed, " FAILED cell(s): ", join(filter(endswith("(FAILED)"), incomplete), ", ")); exit(2))
end
main(ARGS)
