using Test
using CoherentStructures

@testset "CoherentStructures" begin
    
    include("fem_tests.jl")

    include("test_pullback_tensors.jl")

end
