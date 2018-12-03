# Bickley Jet

The Bickley jet flow is a kinematic idealized model of a meandering zonal jet
flanked above and below by counterrotating vortices. It was introduced by
[Rypina et al.](https://dx.doi.org/10.1175/JAS4036.1); cf. also [del‐Castillo‐Negrete and Morrison](https://doi.org/10.1063/1.858639).

The Bickley jet is described by a time-dependent velocity field arising from a
stream-function. The corresponding velocity field is provided by the package and
callable as `bickleyJet`.

Instead of using the `bickleyJet` function to get this velocity field, we could
also use the `@velo_from_stream` macro:
```@example 2
using CoherentStructures
bickley = @velo_from_stream stream begin
    stream = psi₀ + psi₁
    psi₀   = - U₀ * L₀ * tanh(y / L₀)
    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term

    re_sum_term =  Σ₁ + Σ₂ + Σ₃

    Σ₁  =  ε₁ * cos(k₁*(x - c₁*t))
    Σ₂  =  ε₂ * cos(k₂*(x - c₂*t))
    Σ₃  =  ε₃ * cos(k₃*(x - c₃*t))

    k₁ = 2/r₀      ; k₂ = 4/r₀    ; k₃ = 6/r₀

    ε₁ = 0.0075    ; ε₂ = 0.15    ; ε₃ = 0.3
    c₂ = 0.205U₀   ; c₃ = 0.461U₀ ; c₁ = c₃ + (√5-1)*(c₂-c₃)

    U₀ = 62.66e-6  ; L₀ = 1770e-3 ; r₀ = 6371e-3
end;
```
Now, `bickley` is a callable function with the standard `OrdinaryDiffEq`
signature `(u, p, t)` with state `u`, (unused) parameter `p` and time `t`.

## Geodesic vortices

Here we briefly demonstrate how to find material barriers to diffusive transport;
see [Geodesic elliptic material vortices](@ref) for references and details.
```
using Distributed
nprocs() == 1 && addprocs()

@everywhere begin
    using CoherentStructures, OrdinaryDiffEq, Tensors, StaticArrays
    const q = 81
    const tspan = range(0., stop=3456000., length=q)
    ny = 120
    nx = (22ny) ÷ 6
    xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0
    xspan = range(xmin, stop=xmax, length=nx)
    yspan = range(ymin, stop=ymax, length=ny)
    P = SVector{2}.(xspan', yspan)
    const δ = 1.e-6
    const DiffTensor = SymmetricTensor{2,2}([2., 0., 1/2])
    mCG_tensor = u -> av_weighted_CG_tensor(bickleyJet, u, tspan, δ;
              D=DiffTensor, tolerance=1e-6, solver=Tsit5())
end

C̅ = SymmetricTensorField((xspan, yspan), pmap(mCG_tensor, P; batch_size=ny))
p = LCSParameters(3*max(step(xspan), step(yspan)), 1.8, 60, 0.7, 1.5, 1e-4)
vortices, singularities = ellipticLCS(C̅, p)
```
The result is visualized as follows:
```
import Plots
Plots.clibrary(:misc) #hide
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, log10.(traceT);
                    aspect_ratio=1, color=:viridis, leg=true,
                    xlims=(0, 6.371π), ylims=(-3, 3),
                    title="DBS field and transport barriers")
foreach(vortices) do vortex
    Plots.plot!(vortex.curve, w=3, label="T = $(round(vortex.p, digits=2))")
end
scatter!(get_coords(singularities), color=:red)
Plots.plot(fig)
```

## FEM-based Methods

Assume we have setup the `bickley` function using the `@velo_from_stream` macro
as described above. We are working on a periodic domain in one direction:
```@example 2
LL = [0.0, -3.0]; UR = [6.371π, 3.0]
ctx, _ = regularP2TriangularGrid((50, 15), LL, UR, quadrature_order=2)
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 6.371π) < 1e-10
bdata = CoherentStructures.boundaryData(ctx, predicate, []);
```
Using a FEM-based method to compute coherent structures:
```@example 2
using Arpack
cgfun = (x -> mean_diff_tensor(bickley, x, range(0.0, stop=40*3600*24, length=81),
     1.e-8; tolerance=1.e-5))

K = assembleStiffnessMatrix(ctx, cgfun, bdata=bdata)
M = assembleMassMatrix(ctx, bdata=bdata)
λ, v = eigs(K, M, which=:SM, nev= 10)

import Plots
plot_real_spectrum(λ)
```
K-means clustering yields the coherent vortices.
```@example 2
using Clustering
ctx2, _ = regularTriangularGrid((200, 60), LL, UR)
v_upsampled = sample_to(v, ctx, ctx2, bdata=bdata)

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

n_partition = 8
res = iterated_kmeans(20, permutedims(v_upsampled[:,2:n_partition]), n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx2, u_combined, 400, 400;
    color=:rainbow, colorbar=:none, title="$n_partition-partition of Bickley jet")
```
