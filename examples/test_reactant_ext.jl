using SimpleFlows
using Lux, Random, Statistics
using Reactant, Enzyme

# Setup
rng = Random.default_rng()
Random.seed!(rng, 42)

D = 2
n_transforms = 2
hidden_dims = 16

model = RealNVP(; n_transforms=n_transforms, dist_dims=D, hidden_layer_sizes=[hidden_dims])
ps, st = Lux.setup(rng, model)
ps = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, ps)

# Generate toy data
n_train_samples = 512
x1 = randn(rng, Float32, 2, n_train_samples ÷ 2) .- 2.0f0
x2 = randn(rng, Float32, 2, n_train_samples ÷ 2) .+ 2.0f0
x_train = hcat(x1, x2)

# Wrap in FlowDistribution
flow = FlowDistribution{Float32, typeof(model), typeof(ps), typeof(st)}(model, ps, st, D, [hidden_dims], MinMaxNormalizer(x_train))

println("Starting train_flow_reactant! ...")
train_flow_reactant!(flow, x_train; n_epochs=2, lr=1f-3, batch_size=256, verbose=true)
println("Done!")
