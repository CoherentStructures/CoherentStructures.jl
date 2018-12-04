using Test, StaticArrays, OrdinaryDiffEq, Tensors, LinearAlgebra, Distributed
using CoherentStructures

@testset "periodic_diff" begin
    @test CoherentStructures.periodic_diff(5,3,10) ==  2
    @test CoherentStructures.periodic_diff(3,5,10) == -2
    @test CoherentStructures.periodic_diff(9,1,10) == -2
    @test CoherentStructures.periodic_diff(1,9,10) ==  2
end
