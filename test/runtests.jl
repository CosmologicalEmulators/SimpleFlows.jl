using Test
using SimpleFlows
using Random, Distributions, LinearAlgebra

rng = Random.MersenneTwister(42)

@testset "SimpleFlows.jl" begin
    include("test_layers.jl")
    include("test_realnvp.jl")
    include("test_io.jl")
    include("test_normalizer.jl")
end
