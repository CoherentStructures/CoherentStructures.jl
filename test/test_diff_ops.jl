using Test, CoherentStructures, Distances, LinearMaps
using Distributed, Statistics

include("define_vector_fields.jl")

t_initial = 0.0
t_final = 3600.0
q = 19
tspan = range(t_initial, stop=t_final, length=q)
xmin, xmax, ymin, ymax = 0., 6.371π, -3., 3.
m = 50
n = 31
N = m*n
x = range(xmin, stop=xmax, length=m)
y = range(ymin, stop=ymax, length=n)
p0 = collect(Iterators.product(x, y))
metric = PeriodicEuclidean([xmax, Inf])
dist = STmetric(metric, 1)
f = let tspan=tspan
    u -> flow(bickleyJet, u, tspan)
end

f(first(p0))
sol = map(f, p0)
n_coords = 7

(nprocs() == 1) && addprocs()
@everywhere using CoherentStructures

@testset "sparse_diff_op_family" begin
    ε = 1e-3
    kernel = gaussian(ε)
    sparsify = Neighborhood(gaussiancutoff(10ε))
    P = sparse_diff_op_family(sol, sparsify, kernel; metric=metric)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)

    P = sparse_diff_op_family(sol, sparsify, kernel, mean; metric=metric)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)

    ε = 0.2
    sparsify = Neighborhood(ε)
    P = sparse_diff_op_family(sol, sparsify, Base.one, row_normalize!∘unionadjacency; α=0, metric=metric)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)
end

@testset "SparsificationMethods" begin
    ε = 5e-1
    k = 10

    kernel = gaussian(ε)
    sparsify = Neighborhood(gaussiancutoff(ε))
    P = sparse_diff_op(sol, sparsify, kernel; metric=dist)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)

    P = sparse_diff_op(sol, MutualKNN(k), kernel; metric=dist)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)

    P = sparse_diff_op(sol, KNN(k), kernel; metric=dist)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)
end

rmprocs(2:nprocs())
