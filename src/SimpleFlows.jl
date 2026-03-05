module SimpleFlows

using Lux, Random, Statistics, MLUtils, ConcreteStructs
using Distributions
using Bijectors
using JSON, NPZ
using Optimisers, Zygote

include("layers.jl")
include("realnvp.jl")
include("normalizer.jl")
include("distribution.jl")
include("training.jl")
include("io.jl")

export RealNVP, FlowDistribution
export MinMaxNormalizer
export train_flow!, save_trained_flow, load_trained_flow

end
