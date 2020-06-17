using Test, StreamMacros, CoherentStructures, StaticArrays, Distances, LinearMaps
using Distributed, Statistics

bickleyJet = @velo_from_stream stream begin
    stream = psi₀ + psi₁
    psi₀   = - U₀ * L₀ * tanh(y / L₀)
    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term

    re_sum_term =  Σ₁ + Σ₂ + Σ₃

    Σ₁  =  ε₁ * cos(k₁*(x - c₁*t))
    Σ₂  =  ε₂ * cos(k₂*(x - c₂*t))
    Σ₃  =  ε₃ * cos(k₃*(x - c₃*t))

    k₁ = 2/r₀      ; k₂ = 4/r₀    ; k₃ = 6/r₀

    ε₁ = 0.0075    ; ε₂ = 0.15    ; ε₃ = 0.3
    c₂ = 0.205U₀   ; c₃ = 0.461U₀ ; c₁ = c₃ + (√5-1)*(c₂-c₃)
    U₀ = 62.66e-6  ; L₀ = 1770e-3 ; r₀ = 6371e-3
end

t_initial = 0.0
t_final = 3600.0
q = 19
tspan = range(t_initial, stop=t_final, length=q)
xmin, xmax, ymin, ymax = 0., 6.371π, -3., 3.
m = 50
n = 31
N = m*n
p0 = vec(SVector{2}.(range(xmin, stop=xmax, length=m), range(ymin, stop=ymax, length=n)'))
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
    sparsify = Neighborhood(gaussiancutoff(ε))
    P = sparse_diff_op_family(sol, sparsify, kernel; metric=metric)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)

    P = sparse_diff_op_family(sol, sparsify, kernel, mean; metric=metric)
    @test P isa LinearMaps.MapOrMatrix
    @test size(P) == (N, N)

    ε = 0.2
    sparsify = Neighborhood(ε)
    P = sparse_diff_op_family(sol, sparsify, Base.one, P -> max.(P...); α=0, metric=metric)
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
