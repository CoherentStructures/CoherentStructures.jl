# CoherentStructures.jl
*Tools for computing Lagrangian Coherent Structures in Julia*

## Introduction

CoherentStructures.jl is a toolbox for computing Lagrangian Coherent Structures
in aperiodic flows in Julia.
It has been developed in Oliver Junge's research group at TUM, Germany, by (in alphabetical order)
   * Alvaro de Diego ()
   * Daniel Karrasch (@dkarrasch)
   * Nathanael Schilling (@natschil)
Contributions from colleagues in the field are most welcome via raising issues or, even better, via pull requests.

## Installation

First install the [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) package.

Then, run the following in the Julia REPL:

    Pkg.clone("git@gitlab.lrz.de:ga24guz/CoherentStructures.git")

If you do not have a public key registered with gitlab, do:

    Pkg.clone("https://gitlab.lrz.de/ga24guz/CoherentStructures.git")

## Examples

As a quick hands-on introduction, we demonstrate the usage of the CoherentStructures
package on some classic flow problems. For references to the original works in
which the methods were developed see the respective help page.

### Rotating Double Gyre

### Bickley Jet
TODO: Cite something here

The Bickley Jet is described by a time-dependent velocity field arising from a stream-function. The `bickleyJet` velocity field is already built in for convenience:
```@example 1
using CoherentStructures
```
#### FEM-based methods

As we are using a periodic domain in one direction:
```@example 1
LL = [0.0,-3.0]; UR=[6.371π,3.0]
ctx = regularTriangularGrid((100,30),LL,UR,quadrature_order=1)
predicate = (x,y) -> (abs(x[2] - y[2]) < 1e-10) && (peuclidean(x[1],y[1],6.371π) < 1e-10)
bdata = CoherentStructures.boundaryData(ctx,predicate,[]);
```
Next, we define the tensor field to be used in the weak Laplace operator
construction, and assemble mass and stiffness matrices, to finally compute its
dominant spectrum and eigenfunctions:
```@example 1
cgfun = (x -> mean_diff_tensor(bickleyJet, x, linspace(0.0,40*3600*24,81),
     1.e-8,tolerance=1.e-5))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,bdata=bdata)
λ, v = eigs(K,M,which=:SM, nev= 10)
plot_real_spectrum(λ)
```
K-means clustering gives something we can plot:
```@example 1
using Clustering, Plots
n_partition = 8
res = kmeans(v[:,2:n_partition]',n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx, u_combined,200,200,bdata=bdata,
    color=:rainbow,colorbar=:none,title="$n_partition-partition of Bickley Jet")
```
We ran `kmeans` only once. To get better results, `kmeans` should be run several times and only the run with the lowest objective function be kept.

#### Geodesic vortices

Here we briefly demonstrate how to find material barriers to diffusive transport.
```@example 1
using CoherentStructures
import Tensors, OrdinaryDiffEq

const q = 81
const tspan = collect(linspace(0.,3456000.,q))
ny = 120
nx = div(ny*24,6)
N = nx*ny
xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0
xspan, yspan = linspace(xmin,xmax,nx), linspace(ymin,ymax,ny)
P = vcat.(xspan,yspan')
const δ = 1.e-6
const DiffTensor = Tensors.SymmetricTensor{2,2}([2., 0., 1/2])
mCG_tensor = u -> av_weighted_CG_tensor(bickleyJet,u,tspan,δ,D =  DiffTensor,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())

C̅ = map(mCG_tensor,P)

LCSparams = (.1, 0.5, 0.04, 0.5, 1.8, 60)
vals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);
```
The result is visualized as follows:
```@example 1
import Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))
begin
    fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂)',aspect_ratio=1,color=:viridis,
            title="DBS-field and transport barriers",xlims=(0., 6.371π),ylims=(-3., 3.),leg=true)
    for i in eachindex(orbits)
        Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],2))")
    end
end
Plots.plot(fig)
```

### Geostrophic Ocean Flow

### Standard Map

The "standard map" with parameter ``a`` is defined on a 2-dimensional doubly 2π-periodic domain by ``f(x,y) = (x+ y+ a \sin(x),y + a\cos(x))``.

For ``a = 0.971635``, the standard map is implemented in `standardMap`, its Jacobi-matrix in `DstandardMap`.

See also [Froyland & Junge (2015)](https://arxiv.org/abs/1505.05056), who calculate Coherent Structures for this map.

Below are some orbits of the standard map
```@example 1
using CoherentStructures, Plots
to_plot = []
for i in 1:50
    srand(i)
    x = rand(2)*2π
    for i in 1:500
        x = CoherentStructures.standardMap(x)
        push!(to_plot,x)
    end
end
Plots.scatter([x[1] for x in to_plot],[x[2] for x in to_plot],
    m=:pixel,ms=1,aspect_ratio=1,legend=:none)
```
#### FEM-based methods

Approximating the Dynamical Laplacian by FEM methods is straightforward:
```@example 1
using Tensors
ctx = regularTriangularGrid((100,100), [0.0,0.0],[2π,2π])
pred  = (x,y) -> (peuclidean(x[1],y[1],2π) < 1e-9) && (peuclidean(x[2],y[2],2π) < 1e-9)
bdata = boundaryData(ctx,pred) #Periodic boundary

const id2 = one(Tensors.Tensor{2,2}) # 2D identity tensor
cgfun = x -> 0.5*(id2 +  Tensors.dott(Tensors.inv(DstandardMap(x))))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,lumped=false,bdata=bdata)
@time λ, v = eigs(-1*K,M,which=:SM)
Plots.plot([
        plot_u(ctx,v[:,i],bdata=bdata,title=@sprintf("\\lambda = %.3f",λ[i]),
        clim=(-0.25,0.25),colorbar=:none) for i in 1:6]...
        )
```
