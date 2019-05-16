using CoherentStructures, Arpack
LL = [0.0, 0.0]; UR = [1.0, 1.0];
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
res = Plots.plot([plot_u(ctx2, u[:,i], 200, 200, color=:viridis, colorbar=:none) for i in [1,2,3]]...)

Plots.plot(res)

using Distributed
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq
using StaticArrays
using AxisArrays
q = 51
const tspan = range(0., stop=1., length=q)
nx = ny = 51
xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
const δ = 1.e-6
mCG_tensor = u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, δ;
        tolerance=1e-6, solver=Tsit5())

C̅ = pmap(mCG_tensor, P; batch_size=ny)
p = LCSParameters(0.5)
vortices, singularities = ellipticLCS(C̅, p; outermost=true)

using Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));
            aspect_ratio=1, color=:viridis, leg=true,
            title="DBS field and transport barriers",
            xlims=(xmin, xmax), ylims=(ymin, ymax))
scatter!([s.coords.data for s in singularities], color=:red, label="singularities")
scatter!([vortex.center.data for vortex in vortices], color=:yellow, label="vortex cores")
for vortex in vortices, barrier in vortex.barriers
    plot!(barrier.curve, w=2, label="T = $(round(barrier.p, digits=2))")
end
Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

