#Based on static_Laplace_eigvs.jl
addprocs()

using CoherentStructures
using Tensors


############################### set up ##################################
begin
    q = 2
    tspan = collect(range(0.,stop=1.,length=q))
    ny = 100; nx = 100;
    xmin = 0.0; xmax = 1.0; ymin = 0.0; ymax = 1.0
    xspace = range(xmin,stop=xmax,length=nx); yspace = range(ymin,stop=ymax,length=ny)
    P = Array{Vec{2}}(nx,ny)
    for i in CartesianRange(size(P))
        P[i] = Vec{2}([xspace[i[1]], yspace[i[2]]])
    end
    xi = [p[1] for p in P]'; yi = [p[2] for p in P]'
    DiffTensor = SymmetricTensor{2,2}(eye(2,2))
end

##################### tensor computation and interpolation ###################
@time Dt = map(p -> pullback_diffusion_tensor(transientGyres,p,tspan,1.e-9,DiffTensor),P)
# @time Dt = map(p -> PullBackDiffTensor(transientGyresEqVari,p,tspan,0.,DiffTensor),P)
D̅ = mean.(Dt)

using Interpolations
@time itp = Interpolations.interpolate(D̅,BSpline(Linear()),OnGrid())
Ditp = Interpolations.scale(itp,xspace,yspace)

################### Laplace discretization #####################
m = 25 # number of cell in one direction
n = 25

ctx=regularTriangularGrid((m,n),[xmin,ymin],[xmax,ymax])
Afunc(x) = Ditp[x[1],x[2]]

@time K = assembleStiffnessMatrix(ctx,Afunc)
@time M = assembleMassMatrix(ctx)

############################# eigendecomposition ###############################

@time λ, v = eigs(K,M,nev=10,which=:SM)
############################### plotting ##################################
for i in 1:10
    plot_u(ctx,v[:,i])
    sleep(1)
end
