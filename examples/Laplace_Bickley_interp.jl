#Based on static_Laplace_eigvs.jl
addprocs()

using CoherentStructures
using Tensors
using PyPlot

##This example computes mean diffusion tensors at grid points, and then interpolates (linearly)
#to quadrature points

############################### set up ##################################
begin
    q = 81
    tspan = collect(range(0,stop=3456000,length=q))
    ny = 120; nx = div(ny*20,6);
    xmin = 0.0; xmax = 6.371π; ymin = -3.; ymax = 3.0
    L = 6.371π; # period in x-direction
    xspace = range(xmin,stop=xmax,length=nx); yspace = range(ymin,stop=ymax,length=ny)
    P = Array{Vec{2}{Float64}}(nx,ny)
    for i in CartesianRange(size(P))
        P[i] = Vec{2}([xspace[i[1]], yspace[i[2]]])
    end
    xi = [p[1] for p in P]'; yi = [p[2] for p in P]'
    p = (62.66e-6, 1770e-3, 9.058543015644972e-6, 1.28453e-5, 2.888626e-5,
         0.0075, 0.15, 0.3, 0.31392246115209543, 0.6278449223041909, 0.9417673834562862)
end

##################### tensor computation and interpolation ###################
@time Dt = map(u -> pullback_diffusion_tensor(bickleyJet!,u,tspan,1e-6,p=p),P)
D̅ = mean.(Dt)

using Interpolations
itp = Interpolations.interpolate(D̅,BSpline(Linear()),OnGrid())
Ditp = Interpolations.scale(itp,xspace,yspace)

################### Laplace discretization #####################
m = 100 # number of cell in one direction
n = 30

ctx = regularTriangularGrid( (m,n),Vec{2}((xmin,ymin)),Vec{2}((xmax,ymax)))
Afunc(x) = Ditp[x[1],x[2]]
@time K = assembleStiffnessMatrix(ctx, Afunc)
@time M = assembleMassMatrix(ctx)

############################# eigendecomposition ###############################

@time λ, v = eigs(K,M,nev=16,which=:SM)


############################### plotting ##################################
for i in 1:16
    plot_u(ctx,v[:,i])
    sleep(2)
end
