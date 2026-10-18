@testset "README Julia examples" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, readme)]
    @test length(blocks) == 6
    for (i, code) in enumerate(blocks)
        @testset "independent block $i" begin
            @test !isempty(strip(code))
            # A fresh process isolates imports, extensions and backend state as
            # well as globals. Run the code verbatim in Pkg.test's environment.
            project = dirname(Base.active_project())
            @info "Execute README block" block=i
            cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`
            @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
            @info "Finished README block" block=i
        end
    end
end
