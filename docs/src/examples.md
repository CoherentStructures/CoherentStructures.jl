# Examples
As a quick hands-on introduction, we demonstrate the usage of the CoherentStructures
package on some classic flow problems. For references to the original works in
which the methods were developed see the respective help page.

## Rotating Double Gyre

### FEM-Based Methods

The following code-snippet shows how these methods can be used.
```@example 5
using CoherentStructures
LL = [0.0,0.0]; UR = [1.0,1.0];
ctx = regularTriangularGrid((50,50),LL,UR)

A = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-10,tolerance= 1.e-4)
K = assembleStiffnessMatrix(ctx,A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K,M,which=:SM);
```
This velocity field is given by the `rot_double_gyre` function. The second
argument to `mean_diff_tensor` are the times at which we average the pullback
diffusion tensors. The third parameter is the step size δ used for the
finite-difference scheme, `tolerance` is passed to the ODE solver from
[DifferentialEquations.jl](http://juliadiffeq.org/). In the above, `A(x)`
approximates the mean diffusion tensor given by

$A(x) = \sum_{t \in \mathcal T}(D\Phi^t(x))^{-1} (D\Phi^t x)^{-T}$

The eigenfunctions saved in `v` approximate those of $\Delta^{dyn}$
```@example 5
import Plots
res = [plot_u(ctx, v[:,i],colorbar=:none,clim=(-3,3)) for i in 1:6];
Plots.plot(res...,margin=-10Plots.px)
```
Looking at the spectrum, there appears a gap after the third eigenvalue:
```@example 5
Plots.scatter(range(1,6), real.(λ))
```
We can use the [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) package to compute coherent structures from the first two nontrivial eigenfunctions:
```@example 5
using Clustering
numclusters=2
res = kmeans(v[:,2:numclusters+1]',numclusters+1)
u = kmeansresult2LCS(res)
Plots.plot([plot_u(ctx,u[:,i],200,200,color=:viridis) for i in [1,2,3]]...)

```
### Geodesic vortices
The rotating double gyre model was introduced by
[Mosovsky & Meiss](https://dx.doi.org/10.1137/100794110). It can be derived from
the stream function
$ \psi(x,y,t)=(1−s(t))\psi_P +s(t)\psi_F$
$\psi_P (x, y) = \sin(2\pi x) \sin(\pi y)$
$\psi_F (x, y) = \sin(\pi x) \sin(2\pi y)$

where ``s`` is (usually taken to be) a cubic interpolating function satisfying
``s(0) = 0`` and ``s(1) = 1``. It therefore interpolates two double gyre flow
fields, from horizontally to vertically arranged counter-rotating gyres. The
corresponding velocity field is provided by the package and callable as
`rot_double_gyre`.

Here, we demonstrate how to calculate black-hole vortices, see
[Geodesic elliptic material vortices](@ref) for references and details.
```@example 1
using CoherentStructures
import Tensors, OrdinaryDiffEq, Plots

const q = 51
const tspan = collect(linspace(0.,1.,q))
ny = 101
nx = 101
N = nx*ny
xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
xspan, yspan = linspace(xmin,xmax,nx), linspace(ymin,ymax,ny)
P = vcat.(xspan',yspan)
const δ = 1.e-6
mCG_tensor = u -> CG_tensor(rot_double_gyre,u,tspan,δ,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())

C = map(mCG_tensor,P)

LCSparams = (.1, 0.5, 0.01, 0.2, 0.3, 60)
vals, signs, orbits = ellipticLCS(C,xspan,yspan,LCSparams);
```
The results are then visualized as follows.
```@example 1
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C)
# damp "outliers"
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))

fig = Plots.heatmap(xspan,yspan,log10.(l₂),aspect_ratio=1,color=:viridis,
            title="FTLE-field and transport barriers",xlims=(xmin, xmax),ylims=(ymin, ymax),leg=true)
for i in eachindex(orbits)
    Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],2))")
end
Plots.plot(fig)
```

## Bickley Jet

The Bickley jet flow is a kinematic idealized model of a meandering zonal jet
flanked above and below by counterrotating vortices. It was introduced by
[Rypina et al.](https://dx.doi.org/10.1175/JAS4036.1); cf. also [del‐Castillo‐Negrete and Morrison](https://doi.org/10.1063/1.858639).

The Bickley jet is described by a time-dependent velocity field arising from a
stream-function. The corresponding velocity field is provided by the package and
callable as `bickleyJet`.

### Geodesic vortices

Here we briefly demonstrate how to find material barriers to diffusive transport;
see [Geodesic elliptic material vortices](@ref) for references and details.
```@example 2
using CoherentStructures
import Tensors, OrdinaryDiffEq

const q = 81
const tspan = collect(linspace(0.,3456000.,q))
ny = 120
nx = div(ny*24,6)
N = nx*ny
xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0
xspan, yspan = linspace(xmin,xmax,nx), linspace(ymin,ymax,ny)
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
```@example 2
import Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))
begin
    fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂),aspect_ratio=1,color=:viridis,
            title="DBS-field and transport barriers",xlims=(0., 6.371π),ylims=(-3., 3.),leg=true)
    for i in eachindex(orbits)
        Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],2))")
    end
end
Plots.plot(fig)
```



### FEM-based Methods

The Bickley Jet is described by a time-dependent velocity field arising from a stream-function. Instead of using the `bickleyJet` function to get this velocity field, we can use the `@velo_from_stream` macro:
```@example 8
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
As we are using a periodic domain in one direction:
```@example 8
LL = [0.0,-3.0]; UR=[6.371π,3.0]
ctx = regularTriangularGrid((100,30),LL,UR,quadrature_order=1)
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 6.371π) < 1e-10
bdata = CoherentStructures.boundaryData(ctx,predicate,[]);
```
Using a FEM-based method to compute coherent structures:
```@example 8
cgfun = (x -> mean(pullback_diffusion_tensor(bickley, x,linspace(0.0,40*3600*24,81),
     1.e-8,tolerance=1.e-5)))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,bdata=bdata)
λ, v = eigs(K,M,which=:SM, nev= 10)
plot_real_spectrum(λ)
```
K-means clustering gives something we can plot:
```@example 8
using Clustering,Plots
n_partition = 8
res = kmeans(v[:,2:n_partition]',n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx, u_combined,200,200,bdata=bdata,
    color=:rainbow,colorbar=:none,title="$n_partition-partition of Bickley Jet")
```
We ran `kmeans` only once. To get better results, `kmeans` should be run several times and only the run with the lowest objective function be kept.

## Geostrophic Ocean Flow
For a more realistic application, we consider an unsteady ocean surface velocity
data set obtained from satellite altimetry measurements produced by SSALTO/DUACS
and distributed by AVISO. The particular space-time window has been used several
times in the literature.

### Geodesic vortices

Here, we demonstrate how to detect material barriers to diffusive transport.
```@example 3
using CoherentStructures
import JLD2, OrdinaryDiffEq, Plots
###################### load and interpolate velocity data sets #############
JLD2.@load("../../examples/Ocean_geostrophic_velocity.jld2")

UI, VI = interpolateVF(Lon,Lat,Time,UT,VT)
p = (UI,VI)

############################ integration set up ################################
q = 91
t_initial = minimum(Time)
t_final = t_initial + 90
const tspan = linspace(t_initial,t_final,q)
nx = 500
ny = Int(floor(0.6*nx))
N = nx*ny
xmin, xmax, ymin, ymax = -4.0, 6.0, -34.0, -28.0
xspan, yspan = linspace(xmin,xmax,nx), linspace(ymin,ymax,ny)
P = vcat.(xspan',yspan)
const δ = 1.e-5
mCG_tensor = u -> av_weighted_CG_tensor(interp_rhs,u,tspan,δ,p = p,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())

C̅ = map(mCG_tensor,P)
LCSparams = (.09, 0.5, 0.05, 0.5, 1.0, 60)
vals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);
```
The result is visualized as follows:
```@example 3
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))
fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂),aspect_ratio=1,color=:viridis,
            title="DBS-field and transport barriers",xlims=(xmin, xmax),ylims=(ymin, ymax),leg=true)
for i in eachindex(orbits)
    Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],2))")
end
Plots.plot(fig)
```

## Standard Map

The "standard map" with parameter ``a`` is defined on a 2-dimensional doubly 2π-periodic domain by ``f(x,y) = (x+ y+ a \sin(x),y + a\cos(x))``.

For ``a = 0.971635``, the standard map is implemented in `CoherentStructures.standardMap`, its Jacobi-matrix in `CoherentStructures.DstandardMap`.

See also [Froyland & Junge (2015)](https://arxiv.org/abs/1505.05056), who calculate Coherent Structures for this map.

Below are some orbits of the standard map
```@example 2
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
Approximating the Dynamical Laplacian by FEM methods is straightforward:
```@example 7
using CoherentStructures #hide
using Tensors, Plots
ctx = regularTriangularGrid((100,100), [0.0,0.0],[2π,2π])
pred  = (x,y) -> ((x[1] - y[1]) % 2π) < 1e-9 && ((x[2] - y[2]) % 2π) < 1e-9
bdata = boundaryData(ctx,pred) #Periodic boundary

id2 = one(Tensors.Tensor{2,2}) # 2D identity tensor
cgfun = x -> 0.5*(id2 +  Tensors.dott(Tensors.inv(CoherentStructures.DstandardMap(x))))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,lumped=false,bdata=bdata)
@time λ, v = eigs(-1*K,M,which=:SM)
Plots.plot(
    [plot_u(ctx,v[:,i],bdata=bdata,title=@sprintf("\\lambda = %.3f",λ[i]),
        clim=(-0.25,0.25),colorbar=:none)
         for i in 1:6]...)
```
