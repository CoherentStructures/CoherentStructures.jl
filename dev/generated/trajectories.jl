using CoherentStructures, Tensors

n = 500
tspan = range(0, stop=1.0, length=20)
initial_points = [Vec{2}(rand(2)) for i in 1:n]
trajectories = [flow(rot_double_gyre, initial_points[i], tspan) for i in 1:n]

ctx, _ = irregularDelaunayGrid(initial_points)

S = adaptiveTOCollocationStiffnessMatrix(ctx, (i,ts) -> trajectories[i], tspan; flow_map_mode=1)
M = assembleMassMatrix(ctx)

using Arpack
λ, V = eigs(S, M; which=:SM, nev=6)

import Plots
plot_real_spectrum(λ)

using Clustering
function iterated_kmeans(numiterations, args...)
    best = kmeans(args...)
    for i in 1:(numiterations - 1)
        cur = kmeans(args...)
        if cur.totalcost < best.totalcost
            best = cur
        end
    end
    return best
end

n_partition = 3
res = iterated_kmeans(20, permutedims(V[:,2:n_partition]), n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i] * i for i in 1:n_partition])

fig = plot_u(ctx, u_combined, 400, 400;
    color=:viridis, colorbar=:none, title="$n_partition-partition of double gyre")

Plots.plot(fig)

using Distributed
(nprocs() == 1) && addprocs()

@everywhere using CoherentStructures
using LinearAlgebra, LinearMaps, StaticArrays, Distances, Plots

tspan = range(10*24*3600, stop=30*24*3600, length=21)
m = 100; n = 31; N = m*n
x = range(0.0, stop=6.371π, length=m)
y = range(-3.0, stop=3.0, length=n)
f = u -> flow(bickleyJet, u, tspan,  tolerance=1e-6)
p0 = vec(SVector{2}.(x, y'))
trajectories = pmap(f, p0; batch_size=m)

per = [6.371π, Inf]

#We calculate 6 diffusion coordinates for each example
n_coords=6

ε = 1e-3
kernel = gaussian(ε)
P = sparse_diff_op_family(trajectories, Neighborhood(gaussiancutoff(ε)), kernel;
                metric=PEuclidean(per))
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)

ε = 5e-1
k = 200
dist = STmetric(PEuclidean(per), 1)
kernel = gaussian(ε)
P = sparse_diff_op(trajectories, Neighborhood(gaussiancutoff(ε)), kernel; metric=dist
    )
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)

Ψ2 = SEBA(Ψ)

field = permutedims(reshape(Ψ2[:,1], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)
field = permutedims(reshape(Ψ2[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)

using Statistics
ε = 1e-3
kernel = gaussian(ε)
using CoherentStructures
P = sparse_diff_op_family(trajectories, Neighborhood(gaussiancutoff(ε)), kernel, mean; metric=PEuclidean(per)
    );
n_coords=6
@time λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)

ε = 0.2
P = sparse_diff_op_family(trajectories, Neighborhood(ε), Base.one, P -> max.(P...); α=0, metric=PEuclidean(per)
    );
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

