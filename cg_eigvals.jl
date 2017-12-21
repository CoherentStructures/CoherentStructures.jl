#Based on static_Laplace_eigvs.jl
import GR
include("velocityFields.jl")
include("DelaunayGrid.jl")
include("util.jl")
include("TO.jl")
using JuAFEM

m = 50 # number of cell in one direction
node_list = Vec{2,Float64}[]
for x1 in linspace(0,1,m)
    for x0 in linspace(0,1,m)
        push!(node_list,Vec{2}([x0,x1]))
    end
end

print("Loaded necessary modules")

begin
    #grid = JuAFEM.generate_grid(Triangle, (m-1,m-1),Vec{2}((0.0,0.0)),Vec{2}((1.0,1.0)))
    grid,loc = generate_grid(Triangle,node_list)
    dim = 2
    ip = Lagrange{dim, RefTetrahedron, 1}()
    qr = QuadratureRule{dim, RefTetrahedron}(5)

    cv = CellScalarValues(qr, ip)
    dh = DofHandler(grid)
    push!(dh, :T, 1)
    close!(dh)
end

#addnodeset!(grid, "boundary", x -> x[1] ≈ 0.0 ||  abs(x[2]) ≈ 1)
#dbc = DirichletBoundaryConditions(dh)
#add!(dbc, :T, getnodeset(grid, "boundary"), (x,t) -> 0.0)
#close!(dbc)
#update!(dbc, 0.0)
#show(dbc)

  #fixU(dh,u)


include("PullbackTensors.jl")
include("FEMassembly.jl")
@time K,M = doassemble(cv,dh,rot_double_gyre2)

Id = one(Tensor{2,2})
cgfun = (x -> invCGTensor(x,[0.0,1.0], 1.e-8,rot_double_gyre2,1.e-3))
@time K = assembleStiffnessMatrix(cv, dh,cgfun)
@time M = assembleMassMatrix(cv,dh)

dht = nodeToDHTable(dh)
@time S = assembleStiffnessMatrix(cv, dh,x->Id)
@time M = assembleMassMatrix(cv,dh)
@time ALPHA = getAlphaMatrix(grid,loc,dht,u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0]),ip)
@time λ, v = eigs(S + ALPHA'*S*ALPHA,M,which=:SM,nev=30)

#@time apply!(K, dbc)
#apply!(M, dbc)
@time λ, v = eigs(S+K,M,which=:SM)

index = sortperm(real.(λ))[end-9]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(grid,loc,real(fixU(dh,v[:,index])),ip,100,100)
plot_spectrum(λ)
#GR.contourf(reshape(real(fixU(dh,v[:,index])),m,m),colormap=GR.COLORMAP_JET)
#savefig("output.png")
