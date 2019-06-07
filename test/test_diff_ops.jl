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

for par in (false, true)
    if par && (nprocs() == 1)
        addprocs()
        @everywhere using CoherentStructures
    end
    @testset "sparse_diff_op_family, parallel=$par" begin
        ε = 1e-3
        @everywhere @eval kernel = x -> exp(-abs2(x) / $(float(4ε)))
        P = sparse_diff_op_family(sol, Neighborhood(gaussiancutoff(ε)), kernel;
                        metric=metric, parallel=par)
        @test P isa CoherentStructures.LinMaps

        P = sparse_diff_op_family(sol, Neighborhood(gaussiancutoff(ε)), kernel;
                        op_reduce=Statistics.mean, metric=metric, parallel=par)
        @test P isa CoherentStructures.LinMaps

        ε = 0.2
        P = sparse_diff_op_family(sol, Neighborhood(ε), Base.one;
                        op_reduce=(P -> max.(P...)), α=0, metric=metric, parallel=par)
        @test P isa CoherentStructures.LinMaps
    end

    @testset "SparsificationMethods, parallel=$par" begin
        ε = 5e-1
        k = 10

        kernel = gaussian(ε)
        P = sparse_diff_op(sol, Neighborhood(gaussiancutoff(ε)), kernel; metric=dist)
        @test P isa CoherentStructures.LinMaps

        P = sparse_diff_op(sol, MutualKNN(k), kernel; metric=dist)
        @test P isa CoherentStructures.LinMaps

        P = sparse_diff_op(sol, KNN(k), kernel; metric=dist)
        @test P isa CoherentStructures.LinMaps
    end
end
rmprocs(2:nprocs())
