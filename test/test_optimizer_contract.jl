using Optimisers

@testset "Optimizer keyword conflicts fail before mutation" begin
    flow = FlowDistribution(Float32; n_transforms=2, dist_dims=2,
        hidden_layer_sizes=[8], rng=Random.Xoshiro(401))
    data = randn(Random.Xoshiro(402), Float32, 2, 32)
    original_ps = deepcopy(flow.ps)
    for lr in (1f-3, 1e-3)
        @test_throws ArgumentError train_flow!(flow, data; lr,
            opt=Optimisers.Adam(1f-3), n_epochs=1, verbose=false)
        @test flow.ps == original_ps
        @test isnothing(flow.normalizer)
    end
end
