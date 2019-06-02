using Test, CoherentStructures, StaticArrays, Distances, BenchmarkTools

x = y = [@SVector rand(2) for _ in 1:21]
@testset "spatiotemporal metric" begin
    @test @inferred evaluate(STmetric(Euclidean(), 1), x, y) ≈ 0
    @test @inferred evaluate(STmetric(Euclidean(), 2), x, y) ≈ 0
    @test @inferred evaluate(STmetric(SqEuclidean(), 1), x, y) ≈ 0
    @test @inferred evaluate(STmetric(Euclidean(), -1), x, y) ≈ 0
    @test @inferred evaluate(STmetric(Euclidean(), -2), x, y) ≈ 0
    @test @inferred evaluate(STmetric(Euclidean(), Inf), x, y) ≈ 0
    @test @inferred evaluate(STmetric(Euclidean(), -Inf), x, y) ≈ 0
    b = @benchmark evaluate($(STmetric()), $x, $y)
    @test b.allocs == 0
end

@testset "pairwise spatiotemporal metric" begin
    @test (@inferred pairwise(STmetric(), [x], [y])) == reshape([0.0], 1, 1)
    @test (@inferred pairwise(STmetric(), [x])) == reshape([0.0], 1, 1)
end

Y = [[@SVector rand(2) for _ in 1:21] for _ in 1:10]
d = zeros(10)
@testset "colwise spatiotemporal metric" begin
    dist = map(y -> evaluate(STmetric(), x, y), Y)
    @test colwise!(d, STmetric(), x, Y) ≈ dist
    b = @benchmark colwise!($d, $(STmetric()), $x, $Y)
    @test b.allocs == 0
end
