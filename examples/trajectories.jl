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

# We first generate some trajectories on a set of `n` random points for the double gyre.

using CoherentStructures, Tensors

n = 500
tspan = range(0, stop=1.0, length=20)
initial_points = [Vec{2}(rand(2)) for i in 1:n]
trajectories = [flow(rot_double_gyre, initial_points[i], tspan) for i in 1:n]

# ## FEM adaptive TO method

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
