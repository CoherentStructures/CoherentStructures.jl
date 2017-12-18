#Based on static_Laplace_eigvs.jl
addprocs()

include("velocityFields.jl")
include("PullbackTensors.jl")
include("FEMassembly.jl")
using PyPlot

############################### set up ##################################
begin
    q = 2
    tspan = collect(linspace(0.,1.,q))
    ny = 100; nx = 100;
    xmin = 0.0; xmax = 1.0; ymin = 0.0; ymax = 1.0
    xspace = linspace(xmin,xmax,nx); yspace = linspace(ymin,ymax,ny)
    P = Array{Vector{Float64}}(nx,ny)
    for i in CartesianRange(size(P))
        P[i] = [xspace[i[1]], yspace[i[2]]]
    end
    xi = [p[1] for p in P]'; yi = [p[2] for p in P]'
    DiffTensor = SymmetricTensor{2,2}(eye(2,2))
end

##################### tensor computation and interpolation ###################
tic()
@time Dt = map(p -> PullBackDiffTensor(transientGyres,p,tspan,1.e-9,DiffTensor),P)
# @time Dt = map(p -> PullBackDiffTensor(transientGyresEqVari,p,tspan,0.,DiffTensor),P)
D̅ = mean.(Dt)

using Interpolations
@time itp = Interpolations.interpolate(D̅,BSpline(Linear()),OnGrid())
Ditp = Interpolations.scale(itp,xspace,yspace)

################### Laplace discretization #####################
m = 25 # number of cell in one direction
n = 25

grid = generate_grid(Triangle, (m,n),Vec{2}((xmin,ymin)),Vec{2}((xmax,ymax)))
# addnodeset!(grid, "boundary", x -> abs(x[2]) ≈ ymin ||  abs(x[2]) ≈ ymax)

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}()
quadrule = QuadratureRule{dim, RefTetrahedron}(5)

cv = CellScalarValues(quadrule, ip)
dh = DofHandler(grid)
push!(dh, :T, 1)
close!(dh)

@time K = assembleStiff(cv, dh, Ditp)
@time M = assembleMass(cv, dh)

############################# eigendecomposition ###############################

@time λ, v = eigs(K,M,nev=10,which=:SM)
toc()
############################### plotting ##################################
# nodesX = [node.x[1] for node in getnodes(grid)]
# nodesY = [node.x[2] for node in getnodes(grid)]
index = 2
figure(figsize=(6,5))
pcolormesh(kron(collect(linspace(xmin,xmax,m+1))',ones(n+1)),
    kron(ones(m+1)',collect(linspace(ymin,ymax,n+1))),reshape(real(v[:,index]),m+1,n+1)')
# scatter(nodesX,nodesY,5,v[:,index])
colorbar()
axes()[:set_aspect]("equal")

# import GR
# GR.contourf(reshape(real(v[:,index]),m+1,n+1),colormap=GR.COLORMAP_JET)
# GR.title("Eigenvector with eigenvalue $(λ[index])")
# savefig("output.png")

# GR.plot(λ,"x")
