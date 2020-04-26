using CoherentStructures, Arpack
LL, UR = (0.0, 0.0), (1.0, 1.0)
ctx, _ = regularTriangularGrid((50, 50), LL, UR)

A = x -> mean_diff_tensor(rot_double_gyre, x, [0.0, 1.0], 1.e-10, tolerance= 1.e-4)
K = assembleStiffnessMatrix(ctx, A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K, M, which=:SM);

import Plots
res = [plot_u(ctx, v[:,i], 100, 100, colorbar=:none, clim=(-3,3)) for i in 1:6];
fig = Plots.plot(res..., margin=-10Plots.px)

Plots.plot(fig)

spectrum_fig = Plots.scatter(1:6, real.(λ))

Plots.plot(spectrum_fig)

using Clustering

ctx2, _ = regularTriangularGrid((200, 200))
v_upsampled = sample_to(v, ctx, ctx2)

numclusters=2
res = kmeans(permutedims(v_upsampled[:,2:numclusters+1]), numclusters + 1)
u = kmeansresult2LCS(res)
res = Plots.plot([plot_u(ctx2, u[:,i], 200, 200, color=:viridis, colorbar=:none) for i in 1:3]...)

Plots.plot(res)

using Distributed
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq
using StaticArrays, AxisArrays
q = 21
tspan = range(0., stop=1., length=q)
nx = ny = 101
xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
δ = 1.e-6
mCG_tensor = let tspan=tspan, δ=δ
    u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, δ; tolerance=1e-6, solver=Tsit5())
end

C̅ = pmap(mCG_tensor, P; batch_size=ceil(Int, length(P)/nprocs()^2))
p = LCSParameters(0.5)
vortices, singularities = ellipticLCS(C̅, p; outermost=true)

using Plots
trace = tensor_invariants(C̅)[5]
fig = plot_vortices(vortices, singularities, (xmin, ymin), (xmax, ymax);
    bg=trace, title="DBS field and transport barriers", showlabel=true, clims=(0,5))
Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

