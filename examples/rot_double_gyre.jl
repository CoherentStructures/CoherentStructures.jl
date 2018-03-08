using juFEMDL

#ctx = regularP2QuadrilateralGrid((25,25))
ctx = regularTriangularGrid((30,30))
#ctx = regularTriangularGrid((25,25))
#ctx = regularQuadrilateralGrid((10,10))
#ctx = regularP2TriangularGrid((30,30))

#With CG-Method
begin
    cgfun = x -> invCGTensor(rot_double_gyre2!,x,[0.0,1.0], 1.e-10,tolerance= 1.e-3)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx,lumped=false)
    @time λ, v = eigs(-1*K,M,which=:SM)
end

#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    inverse_flow = u0 -> flow(rot_double_gyre2!,u0,[1.0,0.0])[end]
    @time ALPHA = nonAdaptiveTO(ctx,inverse_flow)
    @time λ, v = eigs(-1*(S + ALPHA'*S*ALPHA),M,which=:SM)
end

#With adaptive TO method
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    forwards_flow = u0->flow(rot_double_gyre2!, u0,[0.0,1.0],ctx_for_boundscheck=ctx,tolerance=1e-2)[end]
    @time S2= adaptiveTO(ctx,forwards_flow)
    @time λ, v = eigs(-1*(S + S2),M,which=:SM)
end
#With L2-Galerkin calculation of ALPHA
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx,lumped=false)

    flowMap = u0->flow(rot_double_gyre2!,u0,[1.0,0.0],tolerance=1.e-4)[end]
    @time preALPHAS= juFEMDL.L2GalerkinTOFromInverse(ctx,flowMap)
    Minv = inv(full(M))
    ALPHA = Minv*preALPHAS
    R = -1*(S+ALPHA'*S*ALPHA)
    R = 0.5(R + R')
    @time λ, v = eigs(R,M,which=:SM,nev=6)
end
index= 2
title = "\\\lambda = $(λ[index])"
plot_u(ctx,real.(v[:,index]),50,50,color=:rainbow,title=title)
