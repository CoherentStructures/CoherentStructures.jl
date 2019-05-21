using CoherentStructures,Tensors

n = 500
tspan=range(0,stop=1.0, length=20)
initial_points = [Vec{2}(rand(2)) for i in 1:n]
trajectories = [
    flow(rot_double_gyre, initial_points[i],tspan) for i in 1:n
    ]

ctx, _ = irregularDelaunayGrid(initial_points)

#Generate stiffness and mass matrices, solve eigenproblem.

S = adaptiveTOCollocationStiffnessMatrix(ctx,(i,ts) -> trajectories[i], tspan; flow_map_mode=1
    )
M = assembleMassMatrix(ctx)

using Arpack
λ, V = eigs(S,M, which=:SM,nev=6)

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

#We plot the result. The result looks "messy" due to the

fig = plot_u(ctx, u_combined, 400, 400;
    color=:viridis, colorbar=:none, title="$n_partition-partition of double gyre")

Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

