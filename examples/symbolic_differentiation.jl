# Based on static_Laplace_eigvs.jl
begin #begin/end block to evaluate all at once in atom
    import GR
    include("velocityFields.jl") #For rot_double_gyre2
    include("TO.jl") #For nonAdaptiveTO
    include("GridFunctions.jl") #For regularyDelaunayGrid
    include("plotting.jl")#For plot_u
    include("PullbackTensors.jl")#For invCGTensor
    include("FEMassembly.jl")#For assembleMassMatrix & co
    include("field_from_hamiltonian.jl")
end

ctx = regularP2TriangularGrid((30,30))

# define a field using its streamfunction
begin
    @makefields fields from H begin
        f(a,b) = a
        Psi = sin(π * f(x,t)) * sin(π * y)
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

#Plotting
index = sortperm(real.(λ))[end-15]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),50,50)

plot_spectrum(λ)
