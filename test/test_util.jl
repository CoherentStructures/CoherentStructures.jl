using Test
using CoherentStructures, Tensors
import AxisArrays
const AA = AxisArrays

@testset "tensor invariants" begin
    T = one(SymmetricTensor{2,2})
    l1, l2, v1, v2, t, d = tensor_invariants(T)
    @test l1 == 1
    @test l2 == 1
    @test v1 == [0, 1]
    @test v2 == [1, 0]
    @test t == 2
    @test d == 1

    Ts = AA.AxisArray(fill(T, (10, 10)))
    L1, L2, V1, V2, T, D = tensor_invariants(Ts)
    @test all(L1 .== l1)
    @test all(L2 .== l1)
    @test all(V1 .== [v1])
    @test all(V2 .== [v2])
    @test all(T .== t)
    @test all(D .== d)
end

@testset "periodic_diff" begin
    @test CoherentStructures.periodic_diff(5,3,10) ==  2
    @test CoherentStructures.periodic_diff(3,5,10) == -2
    @test CoherentStructures.periodic_diff(9,1,10) == -2
    @test CoherentStructures.periodic_diff(1,9,10) ==  2
end
