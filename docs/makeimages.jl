
addprocs()
mkdir("docs/buildimg/")
@everywhere using CoherentStructures
@everywhere function checkerboard(x)
    return ((-1)^(floor((x[1]*10)%10)))*((-1)^(floor((x[2]*10)%10)))
end

@macroexpand @animate for i in 1:10
    Plots.scatter([0.0],[1.0])
end

Plots.mp4(l)

plot_u(ctx2,u2,100,100)

inverse_flow_map_t = (t,u0) -> flow(rot_double_gyre,u0,[t,0.0])[end]
@everywhere ctx2 = regularTriangularGrid((200,200))
@everywhere u2 = nodal_interpolation(ctx2,checkerboard)
@everywhere uf(t) = u2
extra_kwargs_fun = t->  [(:title, @sprintf("Rotating Double Gyre, t=%.2f", t))]
res = CoherentStructures.eulerian_video(ctx2,uf,inverse_flow_map_t,
    0.0,1.0, 500,500,100, [0.0,0.0],[1.0,1.0],colorbar=false,
    extra_kwargs_fun=extra_kwargs_fun)
Plots.mp4(res ,"docs/buildimg/rotdoublegyre.mp4")

LL = [0.0,0.0]; UR = [1.0,1.0];
ctx = regularTriangularGrid((25,25),LL,UR)

A = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-10,tolerance= 1.e-3)
K = assembleStiffnessMatrix(ctx,A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K,M,which=:SM)
res = [plot_u(ctx, v[:,i],colorbar=:none,clim=(-3,3)) for i in 1:6]
res2 = Plots.plot(res...,margin=-10Plots.px,dpi=10)
Plots.savefig(res2,"docs/buildimg/rotdgev1.png")


ctx = regularQuadrilateralGrid((10,10))
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && (abs((p1[1] - p2[1])%1.0) < 1e-10)
bdata = boundaryData(ctx,predicate,[])
u = ones(nDofs(ctx,bdata))
u[20] = 2.0; u[38] = 3.0; u[56] = 4.0
plot_u(ctx,u,200,200,bdata=bdata)


using CoherentStructures,Tensors
ctx = regularTriangularGrid((100,100), [0.0,0.0],[2π,2π])
pred  = (x,y) -> ((x[1] - y[1]) % 2π) < 1e-9 && ((x[2] - y[2]) % 2π) < 1e-9
bdata = boundaryData(ctx,pred)

id2 = one(Tensors.Tensor{2,2}) # 2D identity tensor
cgfun = x -> 0.5*(id2 +  dott(inv(CoherentStructures.DstandardMap(x))))

K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
M = assembleMassMatrix(ctx,lumped=false,bdata=bdata)
@time λ, v = eigs(-1*K,M,which=:SM)
Plots.plot([plot_u(ctx,v[:,i],bdata=bdata,clim=(-0.25,0.25),colorbar=:none,title=@sprintf("\\lambda = %.3f",λ[i])) for i in 1:6]...)

res[3]
Plots.plot(res...)

orbits = []
for i in 1:50
    srand(i)
    x = rand(2)*2π
    for i in 1:500
        x = CoherentStructures.standardMap(x)
        push!(orbits,x)
    end
end
Plots.scatter([x[1] for x in orbits],[x[2] for x in orbits],
    m=:pixel,ms=1,aspect_ratio=1,legend=:none)
