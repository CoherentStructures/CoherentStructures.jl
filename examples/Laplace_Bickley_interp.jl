#Based on static_Laplace_eigvs.jl
addprocs()

using juFEMDL
using Tensors
using PyPlot

##This example computes mean diffusion tensors at grid points, and then interpolates (linearly)
#to quadrature points

############################### set up ##################################
begin
    q = 81
    tspan = collect(linspace(0,3456000,q))
    ny = 120; nx = div(ny*20,6);
    xmin = 0.0; xmax = 6.371π; ymin = -3.; ymax = 3.0
    L = 6.371π; # period in x-direction
    xspace = linspace(xmin,xmax,nx); yspace = linspace(ymin,ymax,ny)
    P = Array{Vec{2}{Float64}}(nx,ny)
    for i in CartesianRange(size(P))
        P[i] = Vec{2}([xspace[i[1]], yspace[i[2]]])
    end
    xi = [p[1] for p in P]'; yi = [p[2] for p in P]'
    DiffTensor = SymmetricTensor{2,2}(eye(2,2))
end

##################### tensor computation and interpolation ###################
@time Dt = map(p -> pullback_diffusion_tensor(bickleyJet,p,tspan,1e-6,DiffTensor),P)
# @time Dt = pmap(p -> PullBackDiffTensor(bickleyJetEqVari,p,tspan,DiffTensor),P)
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
