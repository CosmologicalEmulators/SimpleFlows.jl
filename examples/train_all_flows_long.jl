# Self-contained synthetic example: RealNVP, NSF and MAF with staged LR decay.
# See examples/README_training.md for setup and commands. No external data needed.
using SimpleFlows, Random, Statistics, Distributions, Optimisers
using Reactant, Enzyme  # Both are required for backend=:reactant.

# A dynamic optimizer-state counter avoids rebuilding/recompiling the trainer
# when the rate changes. This scales Adam's update, not its moment estimates.
struct StageMultiplier{T} <: Optimisers.AbstractRule
    log_ratio::T
    updates_per_stage::Int
    last_stage::Int
end
Optimisers.init(::StageMultiplier, x::AbstractArray{T}) where {T} =
    fill!(similar(x, T, 1), zero(T))
function Optimisers.apply!(rule::StageMultiplier, counter, x, dx)
    stage = min.(floor.(counter ./ rule.updates_per_stage), rule.last_stage)
    factor = exp.(rule.log_ratio .* stage)
    return counter .+ one(eltype(x)), dx .* factor
end

function banana_samples(rng, n)
    x = randn(rng, Float32, 2, n)
    x[2, :] .= 0.8f0 .* x[1, :] .+ 0.25f0 .* (x[1, :] .^ 2 .- 1) .+ 0.5f0 .* x[2, :]
    return x
end

function main(args=ARGS)
    smoke = "--smoke" in args
    backend = "--zygote" in args ? :zygote : :reactant
    idx = findfirst(==("--output"), args)
    !isnothing(idx) && idx == length(args) && error("--output requires a directory")
    output = isnothing(idx) ? joinpath(@__DIR__, "training_output") : abspath(args[idx + 1])
    ispath(output) && error("Refusing to overwrite $output; choose a new --output directory")
    mkpath(output)
    backend == :reactant && Reactant.set_default_backend("cpu")

    # Change these settings to explore capacity, duration or schedule.
    stages = 10
    epochs_per_stage = smoke ? 2 : 1000
    total_epochs = stages * epochs_per_stage
    batch_size = smoke ? 64 : 256
    initial_lr, final_lr = 1f-3, 1f-6
    train = banana_samples(Xoshiro(1), smoke ? 256 : 4096)
    validation = banana_samples(Xoshiro(2), smoke ? 64 : 1024)
    test = banana_samples(Xoshiro(3), smoke ? 64 : 1024)
    updates_per_stage = epochs_per_stage * cld(size(train, 2), batch_size)

    for architecture in (:RealNVP, :NSF, :MAF)
        flow = FlowDistribution(Float32; architecture, n_transforms=3,
            hidden_layer_sizes=[32, 32], dist_dims=2, rng=Xoshiro(42),
            parameter_names=["x", "y"])
        schedule = StageMultiplier(log(final_lr / initial_lr) / (stages - 1),
                                   updates_per_stage, stages - 1)
        optimizer = Optimisers.OptimiserChain(Optimisers.ClipGrad(1f0),
                                             Optimisers.Adam(initial_lr), schedule)
        history = Any[]
        started = time_ns()
        # ONE call: weights, Adam moments, RNG and compiled caches persist across
        # all ten stages. Do not replace this with ten independent training calls.
        train_flow!(flow, train; backend, opt=optimizer, n_epochs=total_epochs,
            batch_size, normalization=:whiten, validation_data=validation,
            history, eval_every=smoke ? 1 : 10, verbose=false, rng=Xoshiro(43))
        elapsed = (time_ns() - started) / 1e9  # Includes compilation/evaluation; not a hot-loop benchmark.

        dir = joinpath(output, string(architecture))
        mkpath(dir)
        open(joinpath(dir, "losses.csv"), "w") do io
            println(io, "epoch,learning_rate,train_nll,validation_nll,best_epoch")
            for h in history
                stage = clamp(cld(max(h.epoch, 1), epochs_per_stage), 1, stages)
                rate = initial_lr * (final_lr / initial_lr)^((stage - 1) / (stages - 1))
                println(io, join((h.epoch, rate, h.train_nll, h.validation_nll, h.best_epoch), ','))
            end
        end
        # train_flow! has already restored the global best-validation weights,
        # not the final epoch. The normalizer is fitted on training data only.
        save_trained_flow(joinpath(dir, "flow"), flow)
        restored = load_trained_flow(joinpath(dir, "flow"))
        @assert logpdf(restored, test) == logpdf(flow, test)
        generated = rand(Xoshiro(44), restored, 2000)
        @assert all(isfinite, generated) && all(isfinite, logpdf(restored, generated))
        test_nll = -mean(Float64.(logpdf(restored, test)))
        @assert isfinite(test_nll)
        @info "Completed synthetic example" architecture backend total_epochs best_epoch=history[end].best_epoch best_validation_nll=minimum(h.validation_nll for h in history) test_nll training_call_seconds=elapsed
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
