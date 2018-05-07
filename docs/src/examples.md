# Test Cases

## Rotating Double Gyre

## Bickley Jet
TODO: Cite something here

The Bickley Jet is described by a time-dependent velocity field arising from a stream-function. We can either use the `bickleyJet` (or `bickleyJet!`) functions to get this velocity field, or use the `@makefields` macro:
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
As we are using a periodic domain in one direction:
```@example 2
LL = [0.0,-3.0]; UR=[6.371π,3.0]
ctx = regularTriangularGrid((100,30),LL,UR,quadrature_order=1)
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && (abs((p1[1] - p2[1])%6.371π) < 1e-10)
bdata = CoherentStructures.boundaryData(ctx,predicate,[]);
```
Using a FEM-based method to compute coherent structures:
```@example 2
cgfun = (x -> mean(pullback_diffusion_tensor(bickley, x,linspace(0.0,40*3600*24,81),
     1.e-8,tolerance=1.e-5)))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,bdata=bdata)
λ, v = eigs(K,M,which=:SM, nev= 10)
plot_real_spectrum(λ)
```
K-means clustering gives something we can plot:
```@example 2
using Clustering,Plots
n_partition = 8
res = kmeans(v[:,2:n_partition]',n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx, u_combined,200,200,bdata=bdata,
    color=:rainbow,colorbar=:none,title="$n_partition-partition of Bickley Jet")
```
We only ran `kmeans` once, to get better results, `kmeans` should be run several times and only the run with the lowest objective function be kept.

## Geostrophic Ocean Flow

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
```@example 2
using Tensors
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
