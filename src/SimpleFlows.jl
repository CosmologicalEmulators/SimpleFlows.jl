module SimpleFlows

using Lux, Random, Statistics, MLUtils, ConcreteStructs
using Distributions
using Bijectors
using JSON, NPZ
using Optimisers, Zygote
using ChainRulesCore
using LinearAlgebra

include("layers.jl")
include("realnvp.jl")
include("normalizer.jl")
include("splines.jl")
include("nsf.jl")
include("made.jl")
include("maf.jl")
include("generic_ops.jl")
include("distribution.jl")
include("training.jl")
include("io.jl")

export RealNVP, NeuralSplineFlow, MaskedAutoregressiveFlow, FlowDistribution
export MinMaxNormalizer, WhiteningNormalizer
export train_flow!, save_trained_flow, load_trained_flow, load_parameter_names, normalize, denormalize
export unconstrained_rational_quadratic_spline
export to_reactant

function to_reactant(x)
    error("Reactant.jl is not loaded. Please run `using Reactant` to use `to_reactant`.")
end

end
