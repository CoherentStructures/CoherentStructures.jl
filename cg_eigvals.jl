#Based on static_Laplace_eigvs.jl
begin #begin/end block to evaluate all at once in atom
    import GR
    include("velocityFields.jl") #For rot_double_gyre2
    include("TO.jl") #For getAlphaMatrix
    include("GridFunctions.jl") #For regularyDelaunayGrid
    include("plotting.jl")#For plot_u
    include("PullbackTensors.jl")#For invCGTensor
    include("FEMassembly.jl")#For assembleMassMatrix & co
end

ctx = regularP2QuadrilateralGrid()
#ctx = regularDelaunayGrid()
#ctx = regularTriangularGrid((25,25))
#ctx = regularQuadrilateralGrid()


cgfun = (x -> invCGTensor(x,[0.0,1.0], 1.e-8,rot_double_gyre2,1.e-3))


#With CG-Method
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(S+K,M,which=:SM)
end

#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time ALPHA = getAlphaMatrix(ctx,u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0]))
    @time λ, v = eigs(S + ALPHA'*S*ALPHA,M,which=:SM)
end

#With adaptive TO method. Note that this gives very non-smooth results, so there is
#Probably a mistake in the code somewhere....
#Alternatively, it seems like the FEM paper uses more timesteps than just 2
#TODO: See if adding more timesteps fixes things
#Also, this doesn't work with P2-Lagrange Elements (or non-triangular elements) , as it's not clear
#how this would need to look like
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time S2= adaptiveTO(ctx,u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0]))
    @time λ, v = eigs(S + S2,M,which=:SM,nev=20)
end
#Plotting
index = sortperm(real.(λ))[end-2]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),30,30)

#ctx2 = regularQuadrilateralGrid((5,3))
#ctx2 = regularP2DelaunayGrid((5,3))
#ctx2 = regularP2TriangularGrid((5,3))
ctx2 = regularP2QuadrilateralGrid((5,3))
#locatePoint(ctx2,Vec{2}([0.9,0.9]))
#dof2U(ctx2,a)
i = 3
for i in 1:ctx2.n
    a = zeros(ctx2.n); a[i] = 1.0
    plot_u(ctx2,a,100,100)
    sleep(0.001)
end
GR.contourf([0.0,1.0,0.0,1.0,0.5],[0.0,0.0,1.0,1.0,0.5],[0.0,1.0,0.0,0.0,0.0])
#plot_spectrum(λ)
#savefig("output.png")

m = 0
for cell in CellIterator(ctx2.dh)
    m += 1
end
m
for j in ctx2.loc.tess
    m += 1
end
m
