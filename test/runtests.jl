using Test
using SimpleFlows
using Random, Distributions, LinearAlgebra

rng = Random.MersenneTwister(42)

@testset "SimpleFlows.jl" begin
    include("test_layers.jl")
    include("test_realnvp.jl")
    include("test_io.jl")
    include("test_normalizer.jl")
    include("test_splines.jl")
    include("test_nsf.jl")
end
