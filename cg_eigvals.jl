#Based on static_Laplace_eigvs.jl
begin #begin/end block to evaluate all at once in atom
    import GR
    include("velocityFields.jl") #For rot_double_gyre2
    include("TO.jl") #For nonAdaptiveTO
    include("GridFunctions.jl") #For regularyDelaunayGrid
    include("plotting.jl")#For plot_u
    include("PullbackTensors.jl")#For invCGTensor
    include("FEMassembly.jl")#For assembleMassMatrix & co
end

#ctx = regularP2QuadrilateralGrid((10,30))
#ctx = regularDelaunayGrid((30,30))
#ctx = regularTriangularGrid((25,25))
#ctx = regularQuadrilateralGrid((10,10))
ctx = regularP2TriangularGrid((30,30))




#With CG-Method
begin
    cgfun = (x -> invCGTensor(x,[0.0,1.0], 1.e-8,rot_double_gyre2,1.e-3))
    @time S = assembleStiffnessMatrix(ctx)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(K,M,which=:SM)
end

#With CG-Method
include("field_from_hamiltonian.jl")
begin
    @makefields fields from H begin
        f(a,b) = a
        Psi = sin(π * f(x,t)) * sin(π * y)
        ## uncomment for a small strong

         bump(r) = (1 - r^2*(3-2r)) * heavyside(r) * heavyside(1-r)
         center_x = 2/3
         center_y = 0.5
         radius = 1/3
         strength = 5*sin(3*pi*t)
         r_sqr = (center_x - 2*x)^2 + (center_y - y)^2
         H =  Psi  + bump(sqrt(r_sqr / (radius^2))) * strength
    end
    field = fields[:(t,u,du)]
    cgfun = (x -> invCGTensor(x,[0.0,1.0], 1.e-8,rot_double_gyre2,1.e-3))
    @time S = assembleStiffnessMatrix(ctx)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(K,M,which=:SM, nev= 20)
end

#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time ALPHA = nonAdaptiveTO(ctx,u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0]))
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

#With L2-Galerkin calculation of ALPHA
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    DinvMap = u0 -> finDiffDFlowMap(u0,[0.0,-1.0],1.e-8,rot_double_gyre2)
    flowMap = u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0],1.e-4)
    @time preALPHAS= L2GalerkinTO(ctx,flowMap,DinvMap)
    Minv = inv(full(M))
    print(typeof(Minv))
    ALPHA = Minv*preALPHAS
    @time λ, v = eigs(S + ALPHA'*S*ALPHA,M,which=:SM,nev=20)
end

#Plotting
index = sortperm(real.(λ))[end-15]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),50,50)

plot_spectrum(λ)



##Functions below were used for testing Grids, please ignore
#ctx2 = regularQuadrilateralGrid((5,3))
#ctx2 = regularP2DelaunayGrid((5,3))
#ctx2 = regularP2TriangularGrid((5,3))
ctx2 = regularP2QuadrilateralGrid((5,3),Vec{2}([1.,1.]),Vec{2}([3.,3.]))
#locatePoint(ctx2,Vec{2}([0.9,0.9]))
#dof2U(ctx2,a)
i = 3
for i in 1:ctx2.n
    a = zeros(ctx2.n); a[i] = 1.0
    plot_u(ctx2,a,100,100,Vec{2}([1.,1.]),Vec{2}([3.,3.]))
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
