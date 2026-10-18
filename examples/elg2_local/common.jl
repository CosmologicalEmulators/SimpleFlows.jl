# Shared helpers for the local ELG2 validation runner (data, split, Gaussian
# baseline, metrics). Everything here is plain Julia; no Turing, no package edits.

using NPZ, JSON, SHA, Statistics, LinearAlgebra, Random, Printf, Dates

const DATA_DIR = "/home/marcobonici/Desktop/work/turing_mooncake_validation_20261008/eft_pf_sampler_results/ELG2/nuts_reference16_20261014"
const DEFAULT_OUT_ROOT = "/home/marcobonici/Desktop/work/simpleflows_elg2_validation_20261017"

# Provenance verified against the parent's elg2_provenance.json / sha256sum of the files.
const EXPECTED_SHA256 = Dict(
    "samples.npy"           => "ac94d657725c10c0076074aecc6027efcbdcd62f68803088ae3ce3f5ca65a354",
    "names.txt"             => "7f5ee13164e70c65d45933aa31312eca2488abfa9f70acfbda6f597a15531efd",
    "divergence_flags.npy"  => "5f17967a60357ebbbca6495153b1cc60c02084a72dac7d566df72c9cd4e8a234",
    "metadata.json"         => "9af96105ce0c0638e7a8e2865e7accf8b42be4624adc87c9a56c0678bdf18241",
)
const ALL_NAMES = ["ln10As", "ns", "h", "wb", "wc", "B1[1]", "B2[1]", "BK[1]", "Btd[1]", "A0[1]", "A2[1]", "etaSN0[1]", "etaSN2[1]"]
const EFT_NAMES = ["B1[1]", "B2[1]", "BK[1]", "Btd[1]", "A0[1]", "A2[1]", "etaSN0[1]", "etaSN2[1]"]
const EFT_COLUMNS = 6:13
const TRAIN_CHAINS = 1:10
const VAL_CHAINS = 11:13
const ASSESS_CHAINS = 14:16
const EXPECTED_DIVERGENCES_PER_CHAIN = [0, 0, 0, 10, 0, 1, 0, 4, 0, 1, 0, 0, 0, 33, 1, 0]

stamp() = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
function logmsg(args...)
    println(stamp(), " | ", args...)
    flush(stdout)
end

file_sha256(path) = bytes2hex(open(sha256, path))

"""
    load_reference() -> (; samples, flags, hashes, names)

Load the ELG2 NUTS reference (`samples` has shape `(draw, parameter, chain)`),
verify provenance (hashes, names, shape, finiteness, divergence counts) and keep
every draw, including the flagged divergent ones.
"""
function load_reference(dir::String=DATA_DIR)
    hashes = Dict(f => file_sha256(joinpath(dir, f)) for f in keys(EXPECTED_SHA256))
    for (f, h) in EXPECTED_SHA256
        hashes[f] == h || error("sha256 mismatch for $f: got $(hashes[f]), expected $h")
    end
    names = filter(!isempty, strip.(readlines(joinpath(dir, "names.txt"))))
    names == ALL_NAMES || error("names.txt does not match the expected parameter names: $names")
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    meta["total_retained"] == 32000 && meta["chains"] == 16 && meta["retained_per_chain"] == 2000 ||
        error("metadata.json does not describe 16 chains x 2000 draws")
    samples = NPZ.npzread(joinpath(dir, "samples.npy"))
    flags = NPZ.npzread(joinpath(dir, "divergence_flags.npy"))
    size(samples) == (2000, 13, 16) || error("unexpected samples shape $(size(samples))")
    size(flags) == (2000, 16) || error("unexpected divergence_flags shape $(size(flags))")
    eltype(samples) == Float64 || error("unexpected samples eltype $(eltype(samples))")
    all(isfinite, samples) || error("samples contain non-finite values")
    per_chain = vec(sum(flags; dims=1))
    per_chain == EXPECTED_DIVERGENCES_PER_CHAIN ||
        error("divergence counts per chain $(per_chain) differ from the provenance $(EXPECTED_DIVERGENCES_PER_CHAIN)")
    sum(flags) == 50 || error("expected 50 divergent draws, found $(sum(flags))")
    return (; samples=Array{Float64}(samples), flags=Array{Bool}(flags), hashes, names, meta)
end

"""Stack the EFT columns of the given chains into an `8 x (draws*chains)` matrix (chain-major)."""
function eft_matrix(samples, chains)
    return reduce(hcat, [permutedims(samples[:, EFT_COLUMNS, c]) for c in chains])
end
flag_vector(flags, chains) = reduce(vcat, [flags[:, c] for c in chains])
chain_labels(chains, ndraw=2000) = reduce(vcat, [fill(c, ndraw) for c in chains])

"""
    training_split(ref; smoke=false) -> (; train, val, ...)

Training and validation matrices (and only those). Assessment draws are *not*
touched here: `assessment_split` builds them, and is only called after training is done.
In smoke mode, deterministic evenly spaced subsets (50 train, 32 validation) are used.
"""
function training_split(ref; smoke::Bool=false)
    train = eft_matrix(ref.samples, TRAIN_CHAINS)
    val = eft_matrix(ref.samples, VAL_CHAINS)
    tidx = collect(1:size(train, 2)); vidx = collect(1:size(val, 2))
    if smoke
        tidx = round.(Int, range(1, size(train, 2); length=50))
        vidx = round.(Int, range(1, size(val, 2); length=32))
    end
    return (; train=train[:, tidx], val=val[:, vidx], train_index=tidx, val_index=vidx,
            train_flags=flag_vector(ref.flags, TRAIN_CHAINS)[tidx],
            val_flags=flag_vector(ref.flags, VAL_CHAINS)[vidx])
end

"""Assessment draws; call only after training. Smoke mode keeps 500 evenly spaced draws."""
function assessment_split(ref; smoke::Bool=false)
    a = eft_matrix(ref.samples, ASSESS_CHAINS); f = flag_vector(ref.flags, ASSESS_CHAINS)
    ch = chain_labels(ASSESS_CHAINS)
    idx = smoke ? round.(Int, range(1, size(a, 2); length=500)) : collect(1:size(a, 2))
    return (; assess=a[:, idx], assess_flags=f[idx], assess_chain=ch[idx])
end

# ── Gaussian baseline (fitted on the training draws only) ────────────────────

struct GaussianBaseline
    mean::Vector{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    logdet::Float64
end

function GaussianBaseline(x::AbstractMatrix)
    x64 = Float64.(x)
    μ = vec(mean(x64; dims=2))
    L = cholesky(Symmetric(cov(x64; dims=2))).L
    return GaussianBaseline(μ, LowerTriangular(Matrix(L)), 2 * sum(log, diag(L)))
end
function gaussian_logpdf(g::GaussianBaseline, x::AbstractMatrix)
    z = g.L \ (Float64.(x) .- g.mean)
    d = size(x, 1)
    return -0.5 .* (vec(sum(abs2, z; dims=1)) .+ d * log(2π) .+ g.logdet)
end
Base.rand(rng::AbstractRNG, g::GaussianBaseline, n::Int) = g.L * randn(rng, length(g.mean), n) .+ g.mean
whiten(g::GaussianBaseline, x) = g.L \ (Float64.(x) .- g.mean)

# ── Metrics ───────────────────────────────────────────────────────────────────

const QUANTILE_LEVELS = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]

"""Two-sample Kolmogorov-Smirnov statistic (supremum distance of the empirical CDFs)."""
function ks_statistic(a::AbstractVector, b::AbstractVector)
    sa, sb = sort(a), sort(b)
    na, nb = length(sa), length(sb)
    i = j = 1; d = 0.0
    while i <= na && j <= nb
        x = min(sa[i], sb[j])
        while i <= na && sa[i] <= x; i += 1; end
        while j <= nb && sb[j] <= x; j += 1; end
        d = max(d, abs((i - 1) / na - (j - 1) / nb))
    end
    return d
end

"""
    sliced_wasserstein(x, y; n_proj=256, n_quantiles=2000, seed=0, p=2)

Monte-Carlo sliced Wasserstein-`p` distance between the columns of `x` and `y`
(both `d × n`, already in the coordinates in which it should be measured).
Each of `n_proj` random unit directions gives a 1-D problem solved by matching
`n_quantiles` equally spaced quantiles of the two projected samples (equal sample
sizes are not required); the result is `(mean_directions mean_quantiles |qx-qy|^p)^(1/p)`.
It is an estimate with Monte-Carlo and finite-sample error, not an exact distance.
"""
function sliced_wasserstein(x::AbstractMatrix, y::AbstractMatrix; n_proj::Int=256,
                            n_quantiles::Int=2000, seed::Int=0, p::Int=2)
    rng = Random.Xoshiro(seed)
    d = size(x, 1)
    levels = ((1:n_quantiles) .- 0.5) ./ n_quantiles
    total = 0.0
    for _ in 1:n_proj
        θ = randn(rng, d); θ ./= norm(θ)
        qx = quantile(vec(θ' * x), levels)
        qy = quantile(vec(θ' * y), levels)
        total += mean(abs.(qx .- qy) .^ p)
    end
    return (total / n_proj)^(1 / p)
end

function marginal_summary(flow_x::AbstractMatrix, ref_x::AbstractMatrix)
    ref_mean = vec(mean(ref_x; dims=2)); ref_sd = vec(std(ref_x; dims=2))
    flow_mean = vec(mean(flow_x; dims=2)); flow_sd = vec(std(flow_x; dims=2))
    qref = reduce(hcat, [quantile(ref_x[i, :], QUANTILE_LEVELS) for i in axes(ref_x, 1)])   # levels x dims
    qflow = reduce(hcat, [quantile(flow_x[i, :], QUANTILE_LEVELS) for i in axes(flow_x, 1)])
    return Dict(
        "mean_shift_over_ref_sd" => (flow_mean .- ref_mean) ./ ref_sd,
        "sd_ratio" => flow_sd ./ ref_sd,
        "quantile_levels" => QUANTILE_LEVELS,
        "flow_quantiles_standardized" => (qflow .- ref_mean') ./ ref_sd',
        "ref_quantiles_standardized" => (qref .- ref_mean') ./ ref_sd',
        "quantile_difference_standardized" => (qflow .- qref) ./ ref_sd',
        "ks" => [ks_statistic(flow_x[i, :], ref_x[i, :]) for i in axes(ref_x, 1)],
        "flow_mean" => flow_mean, "flow_sd" => flow_sd, "ref_mean" => ref_mean, "ref_sd" => ref_sd,
    )
end

function covariance_summary(flow_x::AbstractMatrix, ref_x::AbstractMatrix)
    cf, cr = cov(flow_x; dims=2), cov(ref_x; dims=2)
    rf, rr = cor(flow_x; dims=2), cor(ref_x; dims=2)
    return Dict(
        "flow_covariance" => cf, "ref_covariance" => cr,
        "flow_correlation" => rf, "ref_correlation" => rr,
        "max_abs_correlation_difference" => maximum(abs, rf .- rr),
        "relative_frobenius_covariance_difference" => norm(cf .- cr) / norm(cr),
    )
end

json_safe(x::AbstractArray) = [json_safe(v) for v in eachrow_or_vec(x)]
eachrow_or_vec(x::AbstractVector) = x
eachrow_or_vec(x::AbstractMatrix) = [collect(r) for r in eachrow(x)]
json_safe(x::Real) = isfinite(x) ? x : string(x)
json_safe(x::AbstractString) = x
json_safe(x::Dict) = Dict(k => json_safe(v) for (k, v) in x)
json_safe(x::NamedTuple) = Dict(string(k) => json_safe(v) for (k, v) in pairs(x))
json_safe(x::Bool) = x
json_safe(::Nothing) = nothing
json_safe(x) = x
