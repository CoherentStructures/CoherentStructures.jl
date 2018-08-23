using CoherentStructures
using Arpack

abcctx = CoherentStructures.regularTriangularGrid3D((25,25,25),[0.0,0.0,0.0],
    [2π,2π,2π],quadrature_order=1 )
bdata_predicate = (x,y) -> (peuclidean(x[1],y[1],2π) < 1e-9 &&
                            peuclidean(x[2],y[2],2π) < 1e-9 &&
                            peuclidean(x[3],y[3],2π) < 1e-9)

bdata = boundaryData(abcctx,bdata_predicate)

cgfun = x-> mean_diff_tensor(abcFlow,x,[0.0,1.0], 1.e-10, p = (√3,√2,1), tolerance= 1.e-3)
@time M = assembleMassMatrix(abcctx,bdata=bdata);
@time K = assembleStiffnessMatrix(abcctx,cgfun,bdata=bdata)
@time λ, V = eigs(K,M,which=:SM,nev=10)

plot_real_spectrum(λ)

### Plotting in 2D

u  = undoBCS(abcctx,V[:,2],bdata)
u /= maximum(abs.(u))
for z in range(0,stop=2π,length=10)
    xs = range(0,stop=2π,length=50)
    ys = range(0,stop=2π,length=50)
    Plots.display(
        Plots.heatmap(xs,ys,
        (x,y) -> evaluate_function_from_dofvals(abcctx,u, [x,y,z]),
        title="z = $z",clim=(-1,1)
    ))
end

### Plotting in 3D
xs = range(0,stop=2π,length=25)
u = undoBCS(abcctx,V[:,3],bdata)
vals = [evaluate_function_from_dofvals(
    abcctx,u,[x,y,z]) for x in xs, y in xs, z in xs]
vals .-= minimum(vals)
vals ./= maximum(vals)


using Makie
scene = Scene()
volume(vals, algorithm = :iso,isovalue=0.2)
center!(scene)
