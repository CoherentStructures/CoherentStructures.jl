# Based on static_Laplace_eigvs.jl
using CoherentStructures
using Tensors

LL=[0.0,0.0]
UR=[2.0,1.0]

ctx = regularP2TriangularGrid((25,25),LL,UR)

# define a field using its streamfunction
begin
    fields = @makefields from H begin
        f(a,b) = a
        Psi = sin(π * f(x,t)) * sin(π * y)
         bump(r) = (1 - r^2*(3-2r)) * heaviside(r) * heaviside(1-r)
         center_x = 2/3
         center_y = 0.5
         radius = 1/3
         strength = 5*sin(3*pi*t)
         r_sqr = (center_x - x)^2 + (center_y - y)^2
         H =  Psi  + bump(sqrt(r_sqr / (radius^2))) * strength
    end
    field = fields[:(du,u,p,t)]
    cgfun = (x -> mean_diff_tensor(field, x,[0.0,1.0], 1.e-8,tolerance=1.e-3))
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(K,M,which=:SM, nev= 20)
end

#Plotting
for i in 1:20
    title = "Eigenvector with eigenvalue $(λ[i])"
    plot_u(ctx,real.(v[:,i]),50,50,title=title)
    sleep(1)
end
