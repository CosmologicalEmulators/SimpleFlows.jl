using Test
using SimpleFlows
using Reactant
using Random
using Distributions
using Zygote
using Enzyme

# A wrapper function to be compiled by Reactant.
# Reactant rejects closures over device arrays, so the flow model must be passed as an argument.
function evaluate_flow_logpdf(x, flow)
    return logpdf(flow, x)
end

# A loss function for gradient testing
function sum_logpdf_loss(x, flow)
    return sum(logpdf(flow, x))
end

# Enzyme gradient helper function
function compile_gradient_wrt_x(x, flow)
    return Enzyme.gradient(Reverse, sum_logpdf_loss, x, Const(flow))[1]
end

@testset "Reactant Extension Tests" begin
    # Note: On CPU-only environments, Reactant may output CUDA probing/initialization
    # warning logs (e.g., "failed call to cuInit"). These warnings are benign and
    # the suite will run successfully on CPU.
    Reactant.set_default_backend("cpu")
    
    # Setup dimension and sample size
    dist_dims = 3
    n_samples = 10
    rng = Random.MersenneTwister(42)
    
    # Generate input data
    x_cpu = randn(rng, Float32, dist_dims, n_samples)
    x_react = Reactant.to_rarray(x_cpu)
    
    architectures = [
        (:RealNVP, (architecture = :RealNVP, n_transforms = 2, dist_dims = dist_dims, hidden_layer_sizes = [16, 16], rng = rng)),
        (:NSF, (architecture = :NSF, n_transforms = 2, dist_dims = dist_dims, hidden_layer_sizes = [16, 16], K = 4, tail_bound = 3.0, rng = rng)),
        (:MAF, (architecture = :MAF, n_transforms = 2, dist_dims = dist_dims, hidden_layer_sizes = [16, 16], rng = rng))
    ]
    
    for (arch_name, kwargs) in architectures
        @testset "Architecture: $arch_name" begin
            # 1. Construct flow on CPU and fit normalizer
            flow_cpu = FlowDistribution(Float32; identity_init=false, kwargs...)
            
            # Fit min-max normalizer manually to test normalizer compatibility
            data_fit = randn(rng, Float32, dist_dims, 100)
            flow_cpu.normalizer = MinMaxNormalizer(data_fit)
            
            # 2. Convert flow to Reactant device representation
            flow_react = to_reactant(flow_cpu)
            
            @test flow_react.ps isa NamedTuple
            @test !isnothing(flow_react.normalizer)
            @test flow_react.normalizer.x_min isa Reactant.ConcretePJRTArray
            
            # 3. Evaluate CPU reference logpdf
            logpdf_ref = logpdf(flow_cpu, x_cpu)
            
            # 4. Compile and evaluate forward pass on Reactant
            f_compiled = Reactant.@compile sync=true evaluate_flow_logpdf(x_react, flow_react)
            logpdf_react = f_compiled(x_react, flow_react)
            Reactant.synchronize(logpdf_react)
            
            # 5. Check Parity
            @test Array(logpdf_react) ≈ logpdf_ref atol=1e-5 rtol=1e-5
            
            # 6. Evaluate CPU reference gradient w.r.t x using Zygote
            grad_ref = Zygote.gradient(x -> sum(logpdf(flow_cpu, x)), x_cpu)[1]
            
            # 7. Compile and evaluate gradient on Reactant using Enzyme
            g_compiled = Reactant.@compile sync=true compile_gradient_wrt_x(x_react, flow_react)
            grad_react = g_compiled(x_react, flow_react)
            Reactant.synchronize(grad_react)
            
            # 8. Check Gradient Parity
            @test Array(grad_react) ≈ grad_ref atol=1e-4 rtol=1e-4
        end
    end

    @testset "to_reactant without normalizer" begin
        flow_cpu = FlowDistribution(Float32; identity_init=false, architecture=:RealNVP, n_transforms=1,
                                    dist_dims=2, hidden_layer_sizes=[4], rng=rng)
        flow_react = to_reactant(flow_cpu)
        @test flow_react.normalizer === nothing

        x_cpu = randn(rng, Float32, 2, 4)
        x_react = Reactant.to_rarray(x_cpu)
        f = Reactant.@compile sync=true evaluate_flow_logpdf(x_react, flow_react)
        @test Array(f(x_react, flow_react)) ≈ logpdf(flow_cpu, x_cpu) rtol=1e-5 atol=1e-5
    end

    @testset "Loaded trained flow can be converted and compiled" begin
        # 1. Create a flow on CPU
        flow_cpu = FlowDistribution(Float32; identity_init=false, architecture=:RealNVP, n_transforms=2,
                                    dist_dims=3, hidden_layer_sizes=[8, 8], rng=rng)
        # Attach a normalizer
        flow_cpu.normalizer = MinMaxNormalizer(randn(rng, Float32, 3, 10))

        # 2. Save it to a temporary directory (avoids committing binary files to repository)
        tmpdir = mktempdir()
        save_trained_flow(tmpdir, flow_cpu)

        # 3. Load it back
        flow_loaded = load_trained_flow(tmpdir; rng=rng)
        
        # 4. Convert and compile
        x_cpu = randn(rng, Float32, 3, 8)
        flow_react = to_reactant(flow_loaded)
        x_react = Reactant.to_rarray(x_cpu)

        ref = logpdf(flow_loaded, x_cpu)
        f = Reactant.@compile sync=true evaluate_flow_logpdf(x_react, flow_react)
        got = f(x_react, flow_react)

        @test Array(got) ≈ ref rtol=1e-5 atol=1e-5
    end
end


@testset "Reactant whitening normalizer" begin
    Reactant.set_default_backend("cpu")
    for T in (Float32, Float64), arch in (:RealNVP, :NSF, :MAF)
        @testset "$arch $T" begin
            rng = Random.MersenneTwister(31)
            kw = (; architecture=arch, n_transforms=2, dist_dims=3, hidden_layer_sizes=[8], rng)
            flow_cpu = arch == :NSF ? FlowDistribution(T; identity_init=false, kw..., K=4, tail_bound=3.0) : FlowDistribution(T; identity_init=false, kw...)
            A = T.([1.0 0.3 0.0; -0.2 0.8 0.1; 0.0 0.2 1.5])
            flow_cpu.normalizer = WhiteningNormalizer(A * randn(rng, T, 3, 60) .+ T[1, -1, 0])

            tmp = mktempdir()                     # exercise a loaded flow, not only a fresh one
            save_trained_flow(tmp, flow_cpu)
            flow_cpu = load_trained_flow(tmp)
            flow_react = to_reactant(flow_cpu)
            @test flow_react.normalizer isa WhiteningNormalizer
            @test flow_react.normalizer.chol isa Reactant.ConcretePJRTArray

            rtol = T == Float32 ? 1e-4 : 1e-9
            atol = T == Float32 ? 1e-4 : 1e-9
            x1 = randn(rng, T, 3, 6); x1r = Reactant.to_rarray(x1)
            f = Reactant.@compile sync=true evaluate_flow_logpdf(x1r, flow_react)
            @test Array(f(x1r, flow_react)) ≈ logpdf(flow_cpu, x1) rtol=rtol atol=atol

            g = Reactant.@compile sync=true compile_gradient_wrt_x(x1r, flow_react)
            @test Array(g(x1r, flow_react)) ≈ Zygote.gradient(v -> sum(logpdf(flow_cpu, v)), x1)[1] rtol=10rtol atol=10atol

            # new input values through the same compiled functions
            x2 = randn(rng, T, 3, 6) .* T(0.5); x2r = Reactant.to_rarray(x2)
            @test Array(f(x2r, flow_react)) ≈ logpdf(flow_cpu, x2) rtol=rtol atol=atol
            @test Array(g(x2r, flow_react)) ≈ Zygote.gradient(v -> sum(logpdf(flow_cpu, v)), x2)[1] rtol=10rtol atol=10atol
        end
    end
end
