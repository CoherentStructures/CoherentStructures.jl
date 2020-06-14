using Test, CoherentStructures, StaticArrays, Distances, BenchmarkTools

m = 10
x = [@SVector rand(2) for _ in 1:m]
y = [@SVector rand(2) for _ in 1:m]
@testset "spatiotemporal metric" begin
    @test @inferred STmetric(Euclidean(), 1)(x, y) ≈ sum(Euclidean().(x, y))/m
    @test @inferred STmetric(Euclidean(), 2)(x, y) ≈ sqrt(sum(abs2, Euclidean().(x, y))/m)
    @test @inferred STmetric(SqEuclidean(), 1)(x, y) ≈ sum(SqEuclidean().(x, y))/m
    @test @inferred STmetric(Euclidean(), -1)(x, y) ≈ inv(sum(inv, Euclidean().(x, y))/m)
    @test @inferred STmetric(Euclidean(), -2)(x, y) ≈ 1/sqrt(sum(x -> x^(-2), Euclidean().(x, y))/m)
    @test @inferred STmetric(Euclidean(), Inf)(x, y) ≈ maximum(Euclidean().(x, y))
    @test @inferred STmetric(Euclidean(), -Inf)(x, y) ≈ minimum(Euclidean().(x, y))
    b = @benchmarkable STmetric()($x, $y)
    @test run(b, samples=4).allocs == 0
end

X = [[@SVector rand(2) for _ in 1:21] for _ in 1:10]
Y = [[@SVector rand(2) for _ in 1:21] for _ in 1:11]
R = zeros(10, 10)
r = zeros(length(X), length(Y))
@testset "pairwise(!) spatiotemporal metric" begin
    @test @inferred pairwise(STmetric(), [x], [y]) == reshape([0.0], 1, 1)
    @test @inferred pairwise(STmetric(), [x]) == reshape([0.0], 1, 1)
    @test @inferred pairwise!(R, STmetric(), X) == @inferred pairwise(STmetric(), X)
    b = @benchmarkable pairwise!($R, $(STmetric()), $X)
    @test run(b, samples=4).allocs == 0
    @test size(pairwise(STmetric(), X, Y)) == (length(X), length(Y))
    b = @benchmarkable pairwise!($r, $(STmetric()), $X, $Y)
    @test run(b, samples=4).allocs == 0
end

Y = [[@SVector rand(2) for _ in 1:21] for _ in 1:10]
d = zeros(10)
@testset "colwise spatiotemporal metric" begin
    dist = map(y -> STmetric()(x, y), Y)
    @test colwise!(d, STmetric(), x, Y) == colwise(STmetric(), x, Y)
    @test d == dist
    @test colwise!(d, STmetric(), Y, x) == colwise(STmetric(), Y, x)
    @test d == dist
    @test colwise!(d, STmetric(), Y, Y) == colwise(STmetric(), Y, Y)
    @test d == zero(dist)
    b = @benchmarkable colwise!($d, $(STmetric()), $x, $Y)
    @test run(b, samples=4).allocs == 0
end
