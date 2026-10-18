# Minimal reproduction (opt-in): Reactant + Enzyme reverse pass through a
# triangular solve. Run with
#   Pkg.test(test_args=["reactant_trisolve_repro"])
# It reports versions and records, rather than asserts, the status of the plain
# `LowerTriangular(L) \ y` gradient; the forward-substitution used by
# SimpleFlows is asserted to work.
using Reactant, Enzyme, LinearAlgebra

@testset "Reactant triangular solve reverse-mode repro" begin
    Reactant.set_default_backend("cpu")
    @info "versions" julia=VERSION Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
    L = Float32[2 0 0; 0.5 1 0; -0.3 0.2 3]
    x = Float32[1 2; 3 4; 5 6]
    Lr, xr = Reactant.to_rarray(L), Reactant.to_rarray(x)

    solve_loss(x, L) = sum(LowerTriangular(L) \ x)
    fwd = Reactant.@compile sync=true solve_loss(xr, Lr)
    @test Float32(fwd(xr, Lr)) ≈ sum(LowerTriangular(L) \ x)    # forward compiles

    grad(x, L) = Enzyme.gradient(Reverse, solve_loss, x, Const(L))[1]
    status = try
        g = Reactant.@compile sync=true grad(xr, Lr)
        Array(g(xr, Lr)) ≈ (LowerTriangular(L)' \ ones(Float32, 3, 2)) ? "WORKS (workaround may be removable)" : "WRONG RESULT"
    catch err
        "FAILS: " * join(filter(l -> occursin("adjoint", l), split(sprint(showerror, err), '\n')), " | ")
    end
    @info "reverse pass through LowerTriangular(L) \\ x" status
    @test true

    # The substitution used by SimpleFlows' Reactant extension differentiates.
    ext = Base.get_extension(SimpleFlows, :ReactantExt)
    sub_loss(x, L) = sum(ext._forward_substitute(L, x))
    gs(x, L) = Enzyme.gradient(Reverse, sub_loss, x, Const(L))[1]
    g2 = Reactant.@compile sync=true gs(xr, Lr)
    @test Array(g2(xr, Lr)) ≈ (LowerTriangular(L)' \ ones(Float32, 3, 2)) rtol=1e-5
end
