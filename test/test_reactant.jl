using SimpleFlows
using SimpleFlows: log_prob, draw_samples
using Lux, Random, Statistics, Test
using Reactant, Enzyme

@testset "Reactant Compilation" begin
    # Use a fixed seed for reproducibility
    rng = Random.default_rng()
    Random.seed!(rng, 123)

    D = 2
    n_transforms = 2
    hidden_dims = 16
    
    model = RealNVP(; n_transforms=n_transforms, dist_dims=D, 
                      hidden_layer_sizes=[hidden_dims])
    ps, st = Lux.setup(rng, model)
    ps = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, ps)
    
    x = randn(rng, Float32, D, 10)

    @testset "HLO Tracing" begin
        # @code_hlo proves that the Julia code can be converted to XLA IR.
        # This is the strongest test for Reactant compatibility without needing a device.
        hlo = @code_hlo log_prob(model, ps, st, x)
        @test hlo isa Reactant.XLA.HLOModule || hlo === nothing # In some versions it might just print
        @test true # If it didn't throw, tracing succeeded
    end

    @testset "Function Compilation" begin
        # Test that we can compile a full training step
        function nll_loss(p, model, st, x)
            lp = log_prob(model, p, st, x)
            return -mean(lp)
        end

        function train_step(p, model, st, x)
            l, grads = Reactant.value_and_gradient(p_inner -> nll_loss(p_inner, model, st, x), p)
            return l, grads
        end

        # This will attempt to trace and compile the gradient calculation
        # If this succeeds, the code is fully XLA-compatible for training
        try
            compiled_fn = Reactant.compile(train_step, (ps, model, st, x))
            @test true
        catch e
            @test false "Reactant compilation failed: $e"
        end
    end
end
