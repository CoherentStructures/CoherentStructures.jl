#md # ```@meta
#md #   EditURL = "../../../examples/trajectories.jl"
#md # ```
# # Working with trajectories
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`trajectories.ipynb`](https://nbviewer.jupyter.org/github/CoherentStructures/CoherentStructures.jl/blob/gh-pages/dev/generated/trajectories.ipynb),
#md #     and as an executable julia file
#md #     [`trajectories.jl`](https://raw.githubusercontent.com/CoherentStructures/CoherentStructures.jl/gh-pages/dev/generated/trajectories.jl).
#md #

# Some of the methods in `CoherentStructures.jl` currently work with velocity fields only.
# However, there are also some methods that work on trajectories directly. These include
# the graph Laplacian based methods and the Transfer-Operator based methods for approximating the
# dynamic Laplacian.

# We show here how to use some of these methods.

# ## FEM adaptive TO method

# We first generate some trajectories on a set of `n` random points for the double gyre.

using CoherentStructures, Tensors

n = 500
tspan = range(0, stop=1.0, length=20)
initial_points = [Vec{2}(rand(2)) for i in 1:n]
trajectories = [flow(rot_double_gyre, initial_points[i], tspan) for i in 1:n]


# Generate a triangulation
# If this call fails or does not return, the initial points may not be unique.

ctx, _ = irregularDelaunayGrid(initial_points)

# Generate stiffness and mass matrices, solve eigenproblem.
# If this call fails or does not return the initial points may not be unique.
S = adaptiveTOCollocationStiffnessMatrix(ctx, (i,ts) -> trajectories[i], tspan; flow_map_mode=1)
M = assembleMassMatrix(ctx)

using Arpack
λ, V = eigs(S, M; which=:SM, nev=6)

# We can plot the spectrum obtained.

import Plots
plot_real_spectrum(λ)

# K-means clustering yields the coherent vortices.

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

# We plot the result. The result looks "messy" due to the fact that we used
# only few trajectories.

fig = plot_u(ctx, u_combined, 400, 400;
    color=:viridis, colorbar=:none, title="$n_partition-partition of double gyre")

DISPLAY_PLOT(fig, trajectories_fem)

# ## Graph Laplace based methods

# Optionally add processes to run things in parallel
using Distributed
(nprocs() == 1) && addprocs()

# We first load some dependencies and our package

@everywhere using CoherentStructures
using LinearAlgebra, LinearMaps, StaticArrays, Distances, Plots

# Setup the problem, and get some trajectories
# We use a regular grid at initial time to make plotting easier.
# We use relatively few trajectories, as some methods take very long with many trajectories
tspan = range(10*24*3600, stop=30*24*3600, length=21)
m = 100; n = 31; N = m*n
x = range(0.0, stop=6.371π, length=m)
y = range(-3.0, stop=3.0, length=n)
f = u -> flow(bickleyJet, u, tspan,  tolerance=1e-6)
p0 = vec(SVector{2}.(x, y'))
trajectories = pmap(f, p0; batch_size=m)

# The problem is periodic in x, but not y
per = [6.371π, Inf]

#We calculate 6 diffusion coordinates for each example
n_coords=6

# We now illustrate some of the different Graph-Laplace based methods

# ### time coupled diffusion coordinates  (Marshall & Hirn, 2018)
# Note that the results would look much better here if we had used more trajectories!
ε = 1e-3
kernel = gaussian(ε)
P = sparse_diff_op_family(trajectories, Neighborhood(gaussiancutoff(ε)), kernel;
                metric=PEuclidean(per))
λ, Ψ = diffusion_coordinates(P, n_coords)

# Plot some resulting eigenvectors
field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, mahirn18ev2)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, mahirn18ev3)

# ### L_1 time averaging with gaussian kernel Hadjighasem et al, 2016

ε = 5e-1
k = 200
dist = STmetric(PEuclidean(per), 1)
kernel = gaussian(ε)
P = sparse_diff_op(trajectories, Neighborhood(gaussiancutoff(ε)), kernel; metric=dist
    )
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, ha16ev2)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, ha16ev3)

# ### Use of SEBA to extract features
# We could have also used "SEBA" from Froyland, Rock & Sakellariou 2019
Ψ2 = SEBA(Ψ)

# We plot two of the SEBA features extracted
field = permutedims(reshape(Ψ2[:,1], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, ha16seba1)
field = permutedims(reshape(Ψ2[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, ha16seba2)


# ### space-time diffusion maps,  Banisch & Koltai, 2017
# Again here the results would look better with more trajectories
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
DISPLAY_PLOT(fig, bakoev2)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, bakoev3)


# ### L_-Inf time averaging, Padberg-Gehle & Schneide, 2018
ε = 0.2
P = sparse_diff_op_family(trajectories, Neighborhood(ε), Base.one, P -> max.(P...); α=0, metric=PEuclidean(per)
    );
λ, Ψ = diffusion_coordinates(P, n_coords)

field = permutedims(reshape(Ψ[:,2], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, pascheev2)
field = permutedims(reshape(Ψ[:,3], m, n))
fig = Plots.heatmap(x, y, field, aspect_ratio=1,color=:viridis)
DISPLAY_PLOT(fig, pascheev3)
