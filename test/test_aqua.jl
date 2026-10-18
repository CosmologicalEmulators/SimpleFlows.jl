using Aqua
using Reactant

@testset "Aqua package quality" begin
    Aqua.test_all(SimpleFlows)
    @test isempty(Test.detect_ambiguities(Base.get_extension(SimpleFlows, :ReactantExt)))
end
