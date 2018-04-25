addprocs()
@everywhere using CoherentStructures

#ctx = regularP2QuadrilateralGrid((25,25))
ctx = regularTriangularGrid((25,25))
#ctx = regularTriangularGrid((25,25))
#ctx = regularQuadrilateralGrid((10,10))
#ctx = regularP2TriangularGrid((30,30))

#With CG-Method

begin
    cgfun = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-10,tolerance= 1.e-3)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx,lumped=false)
    @time λ, v = eigs(-1*K,M,which=:SM)
end

function checkerboard(x)
    return ((-1)^(floor((x[1]*10)%10)))*((-1)^(floor((x[2]*10)%10)))
end

ctx2 = regularTriangularGrid((200,200))
u2 = nodal_interpolation(ctx2,checkerboard)
plot_u(ctx2,u2,100,100)
inverse_flow_map_t = (t,u0) -> flow(rot_double_gyre,u0,[t,0.0])[end]
inverse_flow_map_t(0.5,[0.5,0.5])
u(t) = u2
res = CoherentStructures.eulerian_video(ctx2,u,inverse_flow_map_t,
    0.0,1.0, 500,500,100, [0.0,0.0],[1.0,1.0],colorbar=false,title="Rotating Double Gyre")
Plots.mp4(res ,"/tmp/res.mp4")


#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    inverse_flow = u0 -> flow(rot_double_gyre!,u0,[1.0,0.0])[end]
    @time ALPHA = nonAdaptiveTO(ctx,inverse_flow)
    @time λ, v = eigs(-1*(S + ALPHA'*S*ALPHA),M,which=:SM)
end

#With adaptive TO method
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    forwards_flow = u0->flow(rot_double_gyre!, u0,[0.0,1.0],ctx_for_boundscheck=ctx,tolerance=1e-3)[end]
    @time S2= adaptiveTO(ctx,forwards_flow)
    @time λ, v = eigs(-1*(S + S2),M,which=:SM)
end
#With L2-Galerkin calculation of ALPHA
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx,lumped=false)

    flowMap = u0->flow(rot_double_gyre!,u0,[1.0,0.0],tolerance=1.e-4)[end]
    @time preALPHAS= CoherentStructures.L2GalerkinTOFromInverse(ctx,flowMap)
    Minv = inv(full(M))
    ALPHA = Minv*preALPHAS
    R = -1*(S+ALPHA'*S*ALPHA)
    R = 0.5(R + R')
    @time λ, v = eigs(R,M,which=:SM,nev=6)
end
index= 7
title = "\\\lambda = $(λ[index])"
plot_u(ctx,real.(v[:,index]),50,50,color=:rainbow,title=title)
