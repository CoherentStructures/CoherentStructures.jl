import GR
using juFEMDL


#ctx = regularP2QuadrilateralGrid((25,25))
ctx = regularDelaunayGrid((25,25))
#ctx = regularTriangularGrid((25,25))
#ctx = regularQuadrilateralGrid((10,10))
#ctx = regularP2TriangularGrid((30,30))

#With CG-Method
begin
    cgfun = x -> invCGTensor(rot_double_gyre2!,x,[0.0,1.0], 1.e-10,tolerance= 1.e-3)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx,lumped=false)
    @time λ, v = eigs(K,M,which=:SM)
end

#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time ALPHA = nonAdaptiveTO(ctx,u0->flow(rot_double_gyre2,u0,[0.0,-1.0])[end])
    @time λ, v = eigs(S + ALPHA'*S*ALPHA,M,which=:SM)
end

#With adaptive TO method
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time S2= adaptiveTO(ctx,u0->flow2D(rot_double_gyre2,u0,[0.0,1.0]))
    @time λ, v = eigs(S + S2,M,which=:SM)
end

#With L2-Galerkin calculation of ALPHA
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    DinvMap = u0 -> LinearizedFlowMap(rot_double_gyre2,u0,[0.0,-1.0],1.e-8)[end]
    flowMap = u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0],1.e-4)
    @time preALPHAS= L2GalerkinTO(ctx,flowMap,DinvMap)
    Minv = inv(full(M))
    ALPHA = Minv*preALPHAS
    @time λ, v = eigs(S + ALPHA'*S*ALPHA,M,which=:SM,nev=20)
end

#Plotting
index = sortperm(real.(λ))[end-1]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),100,100)
