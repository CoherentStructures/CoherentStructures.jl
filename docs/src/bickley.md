# Bickley Jet

The Bickley jet flow is a kinematic idealized model of a meandering zonal jet
flanked above and below by counterrotating vortices. It was introduced by
[Rypina et al.](https://dx.doi.org/10.1175/JAS4036.1); cf. also [del‐Castillo‐Negrete and Morrison](https://doi.org/10.1063/1.858639).

The Bickley jet is described by a time-dependent velocity field arising from a
stream-function. The corresponding velocity field is provided by the package and
callable as `bickleyJet`.

Instead of using the `bickleyJet` function to get this velocity field, we could also use the `@velo_from_stream` macro:
```@example 2
using CoherentStructures
# after this, 'bickley' will reference a Dictionary of functions
# access it via the desired signature. e.g. F = bickley[:(dU, U, p, t)]
# for the right side of the equation of variation.
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


## Geodesic vortices

Here we briefly demonstrate how to find material barriers to diffusive transport;
see [Geodesic elliptic material vortices](@ref) for references and details.
```@example 1
using CoherentStructures
import Tensors, OrdinaryDiffEq

const q = 81
const tspan = collect(range(0.,stop=3456000.,length=q))
ny = 120
nx = div(ny*24,6)
N = nx*ny
xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0
xspan, yspan = range(xmin,stop=xmax,length=nx), range(ymin,stop=ymax,length=ny)
P = vcat.(xspan',yspan)
const δ = 1.e-6
const DiffTensor = Tensors.SymmetricTensor{2,2}([2., 0., 1/2])
mCG_tensor = u -> av_weighted_CG_tensor(bickleyJet,u,tspan,δ,
          D=DiffTensor,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())

C̅ = map(mCG_tensor,P)

LCSparams = (.1, 0.5, 0.04, 0.5, 1.8, 60)
vals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);
```
The result is visualized as follows:
```@example 1
import Plots
using Statistics
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))
begin
    fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂),aspect_ratio=1,color=:viridis,
            title="DBS-field and transport barriers",xlims=(0., 6.371π),ylims=(-3., 3.),leg=true)
    for i in eachindex(orbits)
        Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],digits=2))")
    end
end
Plots.plot(fig)
```



## FEM-based Methods

Assume we have setup the `bickley` function using the `@velo_from_stream` macro like described above.
As we are using a periodic domain in one direction:
```@example 2
LL = [0.0,-3.0]; UR=[6.371π,3.0]
ctx = regularP2TriangularGrid((50,15),LL,UR,quadrature_order=2)
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 6.371π) < 1e-10
bdata = CoherentStructures.boundaryData(ctx,predicate,[]);
```
Using a FEM-based method to compute coherent structures:
```@example 2
using Arpack,Statistics
cgfun = (x -> mean(pullback_diffusion_tensor(bickley, x,range(0.0,stop=40*3600*24,length=81),
     1.e-8,tolerance=1.e-5)))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,bdata=bdata)
λ, v = eigs(K,M,which=:SM, nev= 10)

import Plots
plot_real_spectrum(λ)
```
K-means clustering gives something we can plot:
```@example 2
using Clustering
n_partition = 8
res = kmeans(permutedims(v[:,2:n_partition]),n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx, u_combined,200,200,bdata=bdata,
    color=:rainbow,colorbar=:none,title="$n_partition-partition of Bickley Jet")
```
We ran `kmeans` only once. To get better results, `kmeans` should be run several times and only the run with the lowest objective function be kept. We also can upsample the eigenvectors to a finer grid, to obtain a better clustering:
```@example 2
ctx2 = regularTriangularGrid((200,60),LL,UR)
v_upsampled = sample_to(v, ctx,ctx2,bdata=bdata)

function iterated_kmeans(numiterations,args...)
    best = kmeans(args...)
    for i in 1:(numiterations-1)
        cur = kmeans(args...)
        if cur.totalcost < best.totalcost
            best = cur
        end
    end
    return best
end

res = iterated_kmeans(20, permutedims(v_upsampled[:,2:n_partition]),n_partition) 
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx2, u_combined,400,400,
    color=:rainbow,colorbar=:none,title="$n_partition-partition of Bickley Jet")



```


