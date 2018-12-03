# Rotating Double Gyre

## Description

The rotating double gyre model was introduced by
[Mosovsky & Meiss](https://dx.doi.org/10.1137/100794110). It can be derived from
the stream function

$\psi(x,y,t)=(1−s(t))\psi_P +s(t)\psi_F \\ \psi_P (x, y) = \sin(2\pi x) \sin(\pi y) \\ \psi_F (x, y) = \sin(\pi x) \sin(2\pi y)$

where ``s`` is (usually taken to be) a cubic interpolating function satisfying
``s(0) = 0`` and ``s(1) = 1``. It therefore interpolates two double-gyre-type
velocity fields, from horizontally to vertically arranged counter-rotating gyres.
The corresponding velocity field is provided by the package and callable as
`rot_double_gyre`.

![](https://raw.githubusercontent.com/natschil/misc/db22aeef/images/double_gyre.gif)

## FEM-Based Methods

The following code-snippet shows how these methods can be used.
```@example 1
using CoherentStructures, Arpack
LL = [0.0, 0.0]; UR = [1.0, 1.0];
ctx, _ = regularTriangularGrid((50, 50), LL, UR)

A = x -> mean_diff_tensor(rot_double_gyre, x, [0.0, 1.0], 1.e-10, tolerance= 1.e-4)
K = assembleStiffnessMatrix(ctx, A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K, M, which=:SM);
```
This velocity field is given by the `rot_double_gyre` function. The third
argument to `mean_diff_tensor` is a vector of time instances at which we evaluate
(and subsequently average) the pullback diffusion tensors. The fourth parameter
is the step size δ used for the finite-difference scheme, `tolerance` is passed
to the ODE solver from [DifferentialEquations.jl](http://juliadiffeq.org/). In
the above, `A(x)` approximates the mean diffusion tensor given by

$A(x) = \sum_{t \in \mathcal T}(D\Phi^t(x))^{-1} (D\Phi^t x)^{-T}$

The eigenfunctions saved in `v` approximate those of $\Delta^{dyn}$
```@example 1
import Plots
res = [plot_u(ctx, v[:,i], 100, 100, colorbar=:none, clim=(-3,3)) for i in 1:6];
Plots.plot(res..., margin=-10Plots.px)
```
Looking at the spectrum, there appears a gap after the third eigenvalue:
```@example 1
Plots.scatter(1:6, real.(λ))
```
We can use the [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) package to compute coherent structures from the first two nontrivial eigenfunctions:
```@example 1
using Clustering

ctx2, _ = regularTriangularGrid((200, 200))
v_upsampled = sample_to(v, ctx, ctx2)

numclusters=2
res = kmeans(permutedims(v_upsampled[:,2:numclusters+1]), numclusters + 1)
u = kmeansresult2LCS(res)
Plots.plot([plot_u(ctx2, u[:,i], 200, 200, color=:viridis, colorbar=:none) for i in [1,2,3]]...)
```
## Geodesic vortices

Here, we demonstrate how to calculate black-hole vortices, see
[Geodesic elliptic material vortices](@ref) for references and details.
```
using Distributed
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
    P = SVector{2}.(xspan', yspan)
    const δ = 1.e-6
    mCG_tensor = u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, δ;
            tolerance=1e-6, solver=Tsit5())
end

C̅ = SymmetricTensorField((xspan, yspan), pmap(mCG_tensor, P; batch_size=ny))
p = LCSParameters(3*max(step(xspan), step(yspan)), 0.3, 60, 0.7, 1.5, 1e-4)
vortices, singularities = ellipticLCS(C̅, p; outermost=true)
```
The results are then visualized as follows.
```
using Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, log10.(traceT);
            aspect_ratio=1, color=:viridis, leg=true,
            title="DBS field and transport barriers")
foreach(vortices) do vortex
    Plots.plot!(vortex.curve, w=3, label="T = $(round(vortex.p, digits=2))")
end
scatter!(get_coords(singularities), color=:red)
Plots.plot(fig)
```
