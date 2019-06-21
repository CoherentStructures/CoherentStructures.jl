using Test, Distributed, CoherentStructures, Statistics, StaticArrays

t_initial = 0.0
t_final = 3600.0
q = 19
tspan = range(t_initial, stop=t_final, length=q)
xmin, xmax, ymin, ymax = 0., 6.371π, -3., 3.
m = 50
n = 31
N = m*n
p0 = vec(SVector{2}.(range(xmin, stop=xmax, length=m), range(ymin, stop=ymax, length=n)'))
metric = PEuclidean([xmax, Inf])
dist = STmetric(metric, 1)
f = u -> flow(bickleyJet, u, tspan)

sol = map(f, p0)
n_coords = 7

(nprocs() == 1) && addprocs()
@everywhere using CoherentStructures

@testset "sparse_diff_op_family" begin
    ε = 1e-3
    @everywhere @eval kernel = x -> exp(-abs2(x) / $(float(4ε)))
    sparsify = Neighborhood(gaussiancutoff(ε))
    P = sparse_diff_op_family(sol, sparsify, kernel; metric=metric)
    @test P isa CoherentStructures.LinMaps

    P = sparse_diff_op_family(sol, sparsify, kernel, Statistics.mean; metric=metric)
    @test P isa CoherentStructures.LinMaps

    ε = 0.2
    sparsify = Neighborhood(ε)
    P = sparse_diff_op_family(sol, sparsify, Base.one, P -> max.(P...); α=0, metric=metric)
    @test P isa CoherentStructures.LinMaps
end

@testset "SparsificationMethods" begin
    ε = 5e-1
    k = 10

    kernel = gaussian(ε)
    sparsify = Neighborhood(gaussiancutoff(ε))
    P = sparse_diff_op(sol, sparsify, kernel; metric=dist)
    @test P isa CoherentStructures.LinMaps

    P = sparse_diff_op(sol, MutualKNN(k), kernel; metric=dist)
    @test P isa CoherentStructures.LinMaps

    P = sparse_diff_op(sol, KNN(k), kernel; metric=dist)
    @test P isa CoherentStructures.LinMaps
end

rmprocs(2:nprocs())
