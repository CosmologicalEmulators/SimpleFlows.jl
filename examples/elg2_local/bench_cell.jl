# Optional BenchmarkTools timings for a saved cell, kept separate from the
# validation run. NOT to be run while other jobs use the CPU, and it makes no
# speed claim by itself: any comparison needs pinned cores, thermal control and
# matched pairs. CPU only.
#
#   julia --project=<this dir> bench_cell.jl --cell <cell dir> [--backend zygote|reactant] [--batch-size N]
#
# What is measured (all through BenchmarkTools, evals=1 for the training step):
#   logpdf_cpu                 logpdf of a prepared Float32 batch
#   input_gradient_cpu_zygote  DifferentiationInterface gradient with a *prepared* AutoZygote backend
#   training_step              ONE optimizer step followed by the post-update validity check
#                              (_engine_step! + _engine_epoch_done!). For the Reactant engine the update
#                              is asynchronous after the pre-update scalar check, so the epoch-end check,
#                              which synchronises the updated parameters, is inside the timed region; for
#                              the Zygote engine the post-update finite check already runs inside the step
#                              (and _engine_epoch_done! is a no-op). The same wrapper is used for both.
# Every timed sample starts from the *same* parameters and a fresh optimizer state (the `setup`
# phase, excluded from timing, restores them); compiled caches are kept. The first training step
# (compile + setup) is reported once as a single labelled call and is not a performance number.

ENV["GKSwstype"] = "100"
using SimpleFlows, Reactant, Enzyme, BenchmarkTools, JSON, Random, Printf, Statistics
using Lux, Optimisers
import Zygote, DifferentiationInterface as DI
using ADTypes: AutoZygote
using Distributions: logpdf
include(joinpath(@__DIR__, "common.jl"))

flat(x::AbstractArray) = vec(Array(x))
flat(x::NamedTuple) = reduce(vcat, map(flat, values(x)))

# ── engine state reset (outside the timed region) ───────────────────────────
function make_reset(engine::SimpleFlows.ZygoteEngine, ps0, rule)
    return function ()
        engine.flow.ps = deepcopy(ps0)
        engine.opt_state = Optimisers.setup(rule, engine.flow.ps)
        return nothing
    end
end

function make_reset(engine, ps0, rule)        # Reactant engine
    warm = engine.ts                           # carries the compiled gradient/update functions
    return function ()
        fresh = Lux.Training.TrainState(warm.model, Reactant.to_rarray(deepcopy(ps0)), (;), rule)
        engine.ts = Lux.Training.TrainState(warm.cache, warm.objective_function, fresh.allocator_cache, fresh.model,
                                            fresh.parameters, fresh.states, fresh.optimizer, fresh.optimizer_state, 0)
        return nothing
    end
end

function timed_step!(engine, idx)
    SimpleFlows._engine_step!(engine, idx, 1, 1)
    SimpleFlows._engine_epoch_done!(engine, 1)    # post-update check; synchronises the Reactant update
    return nothing
end

function cpu_temperature()
    temperatures = Float64[]
    for dir in readdir("/sys/class/hwmon"; join=true)
        name = joinpath(dir, "name")
        isfile(name) && strip(read(name, String)) == "coretemp" || continue
        for file in readdir(dir)
            occursin(r"^temp\d+_input$", file) || continue
            push!(temperatures, parse(Float64, strip(read(joinpath(dir, file), String))) / 1000)
        end
    end
    isempty(temperatures) && error("No coretemp telemetry; controlled benchmark cannot proceed")
    return maximum(temperatures)
end

function cool_before!(out, name)
    start = time_ns()
    consecutive = 0
    while consecutive < 3
        temperature = cpu_temperature()
        consecutive = temperature <= 65 ? consecutive + 1 : 0
        (time_ns() - start) / 1e9 < 180 || error("CPU did not stabilize below 65 C")
        sleep(1)
    end
    out[name * "_temperature_before_c"] = cpu_temperature()
end

function check_after!(out, name)
    temperature = cpu_temperature()
    out[name * "_temperature_after_c"] = temperature
    temperature < 90 || error("Rejecting thermally uncontrolled measurement: $temperature C")
end

function main(argv)
    cell = nothing; backend = "zygote"; batch = 256; output = nothing
    i = 1
    while i <= length(argv)
        if argv[i] == "--cell"; cell = argv[i + 1]; i += 2
        elseif argv[i] == "--backend"; backend = argv[i + 1]; i += 2
        elseif argv[i] == "--batch-size"; batch = parse(Int, argv[i + 1]); i += 2
        elseif argv[i] == "--output"; output = argv[i + 1]; i += 2
        else error("unknown argument $(argv[i])") end
    end
    isnothing(cell) && error("--cell is required")
    output = isnothing(output) ? joinpath(cell, "benchmark_$(backend).json") : output
    ispath(output) && error("Refusing to overwrite benchmark output: $output")
    backend in ("zygote", "reactant") || error("--backend must be zygote or reactant")
    backend == "reactant" && Reactant.set_default_backend("cpu")
    flow = load_trained_flow(joinpath(cell, "flow"))
    ref = load_reference()
    sp = training_split(ref)                       # training/validation only
    x = Float32.(sp.val[:, 1:batch])
    out = Dict{String, Any}("cell" => basename(cell), "backend" => backend, "batch" => batch,
                            "benchmarktools" => string(pkgversion(BenchmarkTools)), "threads" => Threads.nthreads(),
                            "cpu" => Sys.CPU_NAME, "affinity" => get(ENV, "BENCHMARK_CPUSET", "unrecorded"),
                            "caveat" => "no speed claim without pinned cores, thermal control and matched pairs")

    logmsg("logpdf (CPU, batch ", batch, ")")
    logpdf(flow, x)  # compile before cooling
    cool_before!(out, "logpdf_cpu")
    out["logpdf_cpu"] = bench_summary(@benchmark logpdf($flow, $x) evals=1 samples=50)
    check_after!(out, "logpdf_cpu")

    logmsg("input gradient (CPU, DifferentiationInterface prepared AutoZygote, batch ", batch, ")")
    f = let flow = flow; v -> sum(logpdf(flow, v)); end
    ad = AutoZygote()
    prep = DI.prepare_gradient(f, ad, x)
    g_ref = Zygote.gradient(f, x)[1]
    DI.gradient(f, prep, ad, x) == g_ref || error("DifferentiationInterface gradient differs from Zygote")
    cool_before!(out, "input_gradient_cpu_zygote")
    out["input_gradient_cpu_zygote"] = bench_summary(@benchmark DI.gradient($f, $prep, $ad, $x) evals=1 samples=50)
    check_after!(out, "input_gradient_cpu_zygote")

    # Hot training step from identical state each sample.
    nrm = flow.normalizer
    xn = SimpleFlows.normalize(nrm, Float32.(sp.train))
    idx = collect(1:batch)
    rule = SimpleFlows._default_optimizer(Float32, 1e-3, nothing)
    work = deepcopy(flow)
    ps0 = deepcopy(work.ps)
    engine_fn = backend == "reactant" ? SimpleFlows._make_reactant_engine : SimpleFlows._make_zygote_engine
    engine = engine_fn(work, xn, nothing, nrm; lr=1e-3, batch_size=batch, opt=nothing)
    t0 = time_ns()
    timed_step!(engine, idx)       # first call: compile + setup (single call, NOT a performance number)
    out["first_step_seconds_compile_and_setup"] = (time_ns() - t0) / 1e9
    # Built only AFTER the warm-up so the Reactant TrainState it captures already holds the compiled caches.
    reset! = make_reset(engine, ps0, rule)
    cache_before = backend == "reactant" ? engine.ts.cache : nothing
    reset!()

    # The reset must be exact and independent of the timed work.
    snap() = SimpleFlows._engine_snapshot(engine)
    p0 = flat(snap()); @assert p0 == flat(ps0)
    timed_step!(engine, idx); p1 = flat(snap()); @assert p1 != p0
    @assert flat(ps0) == p0 "the saved initial parameters were mutated"
    reset!(); @assert flat(snap()) == p0 "reset did not restore the parameters"
    timed_step!(engine, idx); @assert flat(snap()) == p1 "the step from a reset state is not reproducible (optimizer state not reset?)"
    reset!()
    if backend == "reactant"
        ex0, ex1 = cache_before.extras, engine.ts.cache.extras
        ex1.compiled_gradient_function === ex0.compiled_gradient_function ||
            error("reset lost/replaced the cached compiled gradient function")
        hasproperty(ex1, :update_function) && ex1.update_function === ex0.update_function ||
            error("reset lost/replaced the cached compiled update function")
    end
    out["reset_verified"] = true

    logmsg("hot training step (", backend, "), one step per sample from a reset state")
    cool_before!(out, "training_step")
    out["training_step"] = bench_summary(@benchmark timed_step!($engine, $idx) setup=($reset!()) evals=1 samples=50)
    check_after!(out, "training_step")
    open(output, "w") do io; JSON.print(io, out, 2); end
    logmsg("wrote ", output)
end

bench_summary(t) = Dict("median_ns" => median(t).time, "minimum_ns" => minimum(t).time, "mean_ns" => mean(t).time,
                        "samples" => length(t.times), "allocs" => t.allocs, "memory_bytes" => t.memory)
main(ARGS)
