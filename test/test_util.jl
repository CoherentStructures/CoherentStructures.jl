using Test
using CoherentStructures, Tensors, StaticArrays
import AxisArrays
const AA = AxisArrays

@testset "utils" begin
    @testset "tensor invariants" begin
        T = one(SymmetricTensor{2,2})
        l1, l2, v1, v2, t, d = @inferred tensor_invariants(T)
        @test l1 == 1
        @test l2 == 1
        @test v1 == [0, 1]
        @test v2 == [1, 0]
        @test t == 2
        @test d == 1

        Ts = AA.AxisArray(fill(T, (10, 10)))
        L1, L2, V1, V2, T, D = @inferred tensor_invariants(Ts)
        @test all(L1 .== l1)
        @test all(L2 .== l1)
        @test all(V1 .== [v1])
        @test all(V2 .== [v2])
        @test all(T .== t)
        @test all(D .== d)
    end

    @testset "skewdot" begin
        x = SVector{2}(rand(2))
        @test iszero(CoherentStructures.skewdot(x, x))
    end
end

@testset "SEBA" begin
    @test SEBA(Matrix(1.0I, 5, 5)) == Matrix(1.0I, 5, 5)
    @test_throws ErrorException SEBA(randn(10, 5), maxiter=1)
end
