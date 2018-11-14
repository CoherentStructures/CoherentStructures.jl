# Standard Map

The "standard map" with parameter ``a`` is defined on a 2-dimensional doubly 2π-periodic domain by ``f(x,y) = (x+ y+ a \sin(x),y + a\cos(x))``.

For ``a = 0.971635``, the standard map is implemented in `CoherentStructures.standardMap`, its Jacobi-matrix in `CoherentStructures.DstandardMap`.

See also [Froyland & Junge (2015)](https://arxiv.org/abs/1505.05056), who calculate Coherent Structures for this map.

Below are some orbits of the standard map
```@example 1
using CoherentStructures
using Random,Plots
to_plot = []
for i in 1:50
    Random.seed!(i)
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
```@example 1
using Tensors, Plots, Arpack, Printf
ctx, _ = regularTriangularGrid((100,100), [0.0,0.0],[2π,2π])
pred  = (x,y) -> peuclidean(x[1],y[1],2π) < 1e-9 && peuclidean(x[2],y[2],2π) < 1e-9
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
