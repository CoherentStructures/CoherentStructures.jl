using CoherentStructures, Arpack
LL = [0.0, 0.0]; UR = [1.0, 1.0];
ctx, _ = regularTriangularGrid((50, 50), LL, UR)

A = x -> mean_diff_tensor(rot_double_gyre, x, [0.0, 1.0], 1.e-10, tolerance= 1.e-4)
K = assembleStiffnessMatrix(ctx, A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K, M, which=:SM);

import Plots
res = [plot_u(ctx, v[:,i], 100, 100, colorbar=:none, clim=(-3,3)) for i in 1:6];
Plots.plot(res..., margin=-10Plots.px)

Plots.scatter(1:6, real.(λ))

using Clustering

ctx2, _ = regularTriangularGrid((200, 200))
v_upsampled = sample_to(v, ctx, ctx2)

numclusters=2
res = kmeans(permutedims(v_upsampled[:,2:numclusters+1]), numclusters + 1)
u = kmeansresult2LCS(res)
Plots.plot([plot_u(ctx2, u[:,i], 200, 200, color=:viridis, colorbar=:none) for i in [1,2,3]]...)

using Distributed
import AxisArrays
const AA = AxisArrays
nprocs() == 1 && addprocs()

@everywhere begin
    using CoherentStructures, OrdinaryDiffEq, StaticArrays
    const q = 51
    const tspan = range(0., stop=1., length=q)
    ny = 101
    nx = 101
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    xspan = range(xmin, stop=xmax, length=nx)
    yspan = range(ymin, stop=ymax, length=ny)
    P = AA.AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
    const δ = 1.e-6
    mCG_tensor = u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, δ;
            tolerance=1e-6, solver=Tsit5())
end

C̅ = pmap(mCG_tensor, P; batch_size=ny)
p = LCSParameters(3*max(step(xspan), step(yspan)), 0.5, 60, 0.7, 1.5, 1e-4)
vortices, singularities = ellipticLCS(C̅, p; outermost=true)

using Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));
            aspect_ratio=1, color=:viridis, leg=true,
            title="DBS field and transport barriers")
scatter!(get_coords(singularities), color=:red)
for vortex in vortices
    plot!(vortex.curve, color=:yellow, w=3, label="T = $(round(vortex.p, digits=2))")
    scatter!(vortex.core, color=:yellow)
end
Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

