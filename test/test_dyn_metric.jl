using Test, CoherentStructures, StaticArrays, Distances, BenchmarkTools, Random

Random.seed!(1234)
m = 20
x = [@SVector rand(2) for _ in 1:m]
xfake = [@SVector rand(2) for _ in 1:(m+1)]
xempty = eltype(x)[]
y = [@SVector rand(2) for _ in 1:m]
p = 1 + rand()
@testset "spatiotemporal metric" begin
    @test @inferred(STmetric(Euclidean(), 1)(x, y)) == stmetric(x, y, Euclidean(), 1) ≈ sum(Euclidean().(x, y))/m
    @test @inferred(STmetric(Euclidean(), 2)(x, y)) == stmetric(x, y, Euclidean(), 2) ≈ sqrt(sum(abs2, Euclidean().(x, y))/m)
    @test @inferred(STmetric(SqEuclidean(), 1)(x, y)) == stmetric(x, y, SqEuclidean(), 1) ≈ sum(SqEuclidean().(x, y))/m
    @test @inferred(STmetric(Euclidean(), -1)(x, y)) == stmetric(x, y, Euclidean(), -1) ≈ inv(sum(inv, Euclidean().(x, y))/m)
    @test @inferred(STmetric(Euclidean(), -2)(x, y)) == stmetric(x, y, Euclidean(), -2) ≈ 1/sqrt(sum(x -> x^(-2), Euclidean().(x, y))/m)
    @test @inferred(STmetric(Euclidean(), Inf)(x, y)) == stmetric(x, y, Euclidean(), Inf) == maximum(Euclidean().(x, y))
    @test @inferred(STmetric(Euclidean(), -Inf)(x, y)) == stmetric(x, y, Euclidean(), -Inf) == minimum(Euclidean().(x, y))
    @test @inferred(STmetric(Euclidean(), p)(x, y)) == stmetric(x, y, Euclidean(), p) == (sum(x -> x^p, Euclidean().(x, y))/m)^(1/p)
    b = @benchmarkable STmetric()($x, $y)
    @test run(b, samples=4).allocs == 0
    @test_throws DimensionMismatch stmetric(xfake, y, Euclidean(), 1)
    @test stmetric(xempty, xempty, Euclidean(), 1) === 0.0
end

X = [[@SVector rand(2) for _ in 1:m] for _ in 1:10]
Y = [[@SVector rand(2) for _ in 1:m] for _ in 1:11]
R = zeros(10, 10)
r = zeros(length(X), length(Y))
@testset "pairwise(!) spatiotemporal metric" begin
    @test @inferred pairwise(STmetric(), [x], [y]) == reshape([STmetric()(x, y)], 1, 1)
    @test @inferred pairwise(STmetric(), [x]) == reshape([0.0], 1, 1)
    @test @inferred pairwise!(R, STmetric(), X) == @inferred pairwise(STmetric(), X)
    b = @benchmarkable pairwise!($R, $(STmetric()), $X)
    @test run(b, samples=4).allocs == 0
    @test size(pairwise(STmetric(), X, Y)) == (length(X), length(Y))
    @test pairwise(STmetric(), X, Y) == STmetric().(X, permutedims(Y))
    b = @benchmarkable pairwise!($r, $(STmetric()), $X, $Y)
    @test run(b, samples=4).allocs == 0
end

Y = [[@SVector rand(2) for _ in 1:m] for _ in 1:10]
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
