using Test
using SimpleFlows
using Random, Distributions, LinearAlgebra
using Lux, Zygote, Bijectors, ForwardDiff, JSON, NPZ

rng = Random.MersenneTwister(42)

# `Pkg.test(test_args=["io", "whitening"])` runs only the matching test_<name>.jl
# files; with no arguments the whole suite runs.
const ALL_TESTS = ["layers", "realnvp", "training", "io", "normalizer", "whitening", "initialization",
                   "splines", "nsf", "maf", "ad", "reactant", "reactant_training", "reactant_identity"]
selected = isempty(ARGS) ? ALL_TESTS : ARGS
# Opt-in reproduction scripts, only run when selected explicitly.
const OPTIONAL_TESTS = ["reactant_trisolve_repro"]
for name in selected
    name in ALL_TESTS || name in OPTIONAL_TESTS || error("Unknown test file selector \"$name\"")
end

@testset "SimpleFlows.jl" begin
    for name in ALL_TESTS
        name in selected && include("test_$name.jl")
    end
    for name in OPTIONAL_TESTS
        name in selected && include("$name.jl")
    end
end
