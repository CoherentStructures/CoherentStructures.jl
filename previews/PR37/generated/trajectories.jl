using Distributed
(nprocs() == 1) && addprocs()

@everywhere using CoherentStructures
using StaticArrays, Distances, Plots

tspan = range(10*24*3600.0, stop=30*24*3600.0, length=41)
m = 120; n = 41; N = m*n
x = range(0.0, stop=20.0, length=m)
y = range(-3.0, stop=3.0, length=n)
f = u -> flow(bickleyJet, u, tspan, tolerance=1e-4)
particles = vec(SVector{2}.(x, y'))
trajectories = pmap(f, particles; batch_size=m)

periods = [6.371π, Inf]
metric = PeriodicEuclidean(periods)

n_coords = 6

ε = 3e-1
kernel = gaussian(ε)
P = sparse_diff_op(trajectories, Neighborhood(gaussiancutoff(ε/5)), kernel; metric=STmetric(metric, 1))
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:, 2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

field = permutedims(reshape(Ψ[:, 3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

P = sparse_diff_op(trajectories, KNN(400), gaussian(10); metric=STmetric(metric, Inf))
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:, 2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

field = permutedims(reshape(Ψ[:, 3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

Ψ2 = SEBA(Ψ)

field = permutedims(reshape(Ψ2[:, 1], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

field = permutedims(reshape(Ψ2[:, 2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

import Statistics: mean
σ = 1e-3
kernel = gaussian(σ)
P = sparse_diff_op_family(trajectories, Neighborhood(gaussiancutoff(σ)), kernel, mean; metric=metric)
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:, 2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

field = permutedims(reshape(Ψ[:, 3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

ε = 0.2
P = sparse_diff_op_family(trajectories, Neighborhood(ε), Base.one, P -> row_normalize!(min.(sum(P), 1));
                            α=0, metric=metric)
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:, 2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

field = permutedims(reshape(Ψ[:, 3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

σ = 1e-3
kernel = gaussian(σ)
P = sparse_diff_op_family(trajectories, Neighborhood(gaussiancutoff(σ)), kernel; metric=metric)
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:, 2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

field = permutedims(reshape(Ψ[:, 3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1, color=:viridis)
Plots.plot(fig)

using CoherentStructures, StaticArrays, Tensors

n = 500
tspan = range(0, stop=1.0, length=20)
xs, ys = rand(n), rand(n)
particles = SVector{2}.(xs, ys)
trajectories = [flow(rot_double_gyre, p, tspan) for p in particles]

ctx, _ = irregularDelaunayGrid(Vec{2}.(particles))

S = adaptiveTOCollocationStiffnessMatrix(ctx, (i, ts) -> trajectories[i], tspan; flow_map_mode=1)
M = assembleMassMatrix(ctx)

using Arpack
λ, V = eigs(S, M; which=:SM, nev=6)

using Plots
fig = plot_real_spectrum(λ, label="")
Plots.plot(fig)

using Clustering
function iterated_kmeans(iterations, args...)
    best = kmeans(args...)
    for i in 1:(iterations - 1)
        cur = kmeans(args...)
        if cur.totalcost < best.totalcost
            best = cur
        end
    end
    return best.assignments
end

partitions = 3
clusters = iterated_kmeans(20, permutedims(V[:, 2:partitions]), partitions)

fig = scatter(xs, ys, zcolor=clusters[ctx.node_to_dof], markersize=8, labels="")
Plots.plot(fig)

fig = plot_u(ctx, float(clusters), 400, 400;
                color=:viridis,
                colorbar=:none,
                title="$partitions-partition of rotating double gyre")
Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

