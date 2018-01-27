#Based on static_Laplace_eigvs.jl
import GR
include("velocityFields.jl") #For rot_double_gyre2
include("TO.jl") #For nonAdaptiveTO
include("GridFunctions.jl") #For regularyDelaunayGrid
include("plotting.jl")#For plot_u
include("PullbackTensors.jl")#For invCGTensor
include("FEMassembly.jl")#For assembleMassMatrix & co

#ctx = regularP2QuadrilateralGrid((25,25))
ctx = regularDelaunayGrid((25,25))
#ctx = regularTriangularGrid((25,25))
#ctx = regularQuadrilateralGrid((10,10))
#ctx = regularP2TriangularGrid((30,30))

#With CG-Method
begin
    cgfun = (x -> invCGTensor(x,[0.0,1.0], 1.e-8,rot_double_gyre2,1.e-3))
    @time S = assembleStiffnessMatrix(ctx)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx,lumped=false)
    @time λ, v = eigs(K,M,which=:SM)
end

#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time ALPHA = nonAdaptiveTO(ctx,u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0]))
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
    DinvMap = u0 -> finDiffDFlowMap(u0,[0.0,-1.0],1.e-8,rot_double_gyre2)
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

plot_spectrum(λ)



##Functions below were used for testing Grids, please ignore
#TODO Cleanup
#ctx2 = regularQuadrilateralGrid((5,3))
#ctx2 = regularP2DelaunayGrid((5,3))
#ctx2 = regularP2TriangularGrid((5,3))
ctx2 = regularP2QuadrilateralGrid((5,3),Vec{2}([1.,1.]),Vec{2}([3.,3.]))
#locatePoint(ctx2,Vec{2}([0.9,0.9]))
#dof2U(ctx2,a)
GR.inline("mov")
for i in 1:10
    a = zeros(ctx2.n); a[i] = 1.0
    plot_u(ctx2,a,100,100,Vec{2}([1.,1.]),Vec{2}([3.,3.]))
end
GR.show()
GR.endprint()
GR.contourf([0.0,1.0,0.0,1.0,0.5],[0.0,0.0,1.0,1.0,0.5],[0.0,1.0,0.0,0.0,0.0])
#plot_spectrum(λ)
#savefig("output.png")



using GR

x = collect(0:0.01:2*pi)

beginprint("anim.mov")
for i = 1:200
    plot(x, sin(x + i / 10.0))
end
endprint()

m = 0
for cell in CellIterator(ctx2.dh)
    m += 1
end
m
for j in ctx2.loc.tess
    m += 1
end
m
