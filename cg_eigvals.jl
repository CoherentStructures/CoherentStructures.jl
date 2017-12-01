#Based on static_Laplace_eigvs.jl
import GR
include("velocityFields.jl")
using JuAFEM

#The function below is taken from main_rot_gyre.jl
function rot_double_gyre2(t,x,dx)
  st = ((t>0)&(t<1))*t^2*(3-2*t) + (t>=1)*1
  dxΨP = 2π*cos.(2π*x[1]).*sin.(π*x[2])
  dyΨP = π*sin.(2π*x[1]).*cos.(π*x[2])
  dxΨF = π*cos.(π*x[1]).*sin.(2π*x[2])
  dyΨF = 2π*sin.(π*x[1]).*cos.(2π*x[2])
  dx[1] = - ((1-st)dyΨP + st*dyΨF)
  dx[2] = (1-st)dxΨP + st*dxΨF
end


print("Loaded necessary modules")
m = 25 # number of cell in one direction
grid = generate_grid(Triangle, (m,m),Vec{2}((0.0,0.0)),Vec{2}((1.0,1.0)))
#addnodeset!(grid, "boundary", x -> abs(x[1]) ≈ 1 ||  abs(x[2]) ≈ 1)

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}()
qr = QuadratureRule{dim, RefTetrahedron}(5)

cv = CellScalarValues(qr, ip)
dh = DofHandler(grid)
push!(dh, :T, 1)
close!(dh)

#dbc = DirichletBoundaryConditions(dh)
#add!(dbc, :T, getnodeset(grid, "boundary"), (x,t) -> 0.0)
#close!(dbc)
#update!(dbc, 0.0)


include("tensorComputations.jl")
function doassemble{dim}(cv::CellScalarValues{dim}, dh::DofHandler,velocityField)
    K = create_sparsity_pattern(dh)
    a_K = start_assemble(K)
    M = create_sparsity_pattern(dh)
    a_M = start_assemble(M)
    dofs = zeros(Int, ndofs_per_cell(dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n,n)
    Me = zeros(n,n)   # Local stiffness and mass matrix
    @inbounds for (cellcounto, cell) in enumerate(CellIterator(dh))
        fill!(Ke,0)
        fill!(Me,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
    	    q_coords::Vec{dim,Float64} = zero(Vec{dim})
    	    for j in 1:n
        		q_coords +=cell.coords[j] * cv.M[j,q]
    	    end
    	    const A = avDiffTensor(q_coords,[0.0,1.0], 1.e-9,velocityField)
                const dΩ = getdetJdV(cv,q)
                for i in 1:n
                    const φ = shape_value(cv,q,i)
                    const ∇φ = shape_gradient(cv,q,i)
                    for j in 1:(i-1)
                        const ψ = shape_value(cv,q,j)
                        const ∇ψ = shape_gradient(cv,q,j)
            		    Ke[i,j] += -1.0*(∇φ ⋅ (A⋅∇ψ)) * dΩ
            		    Ke[j,i] += -1.0*(∇φ ⋅ (A⋅∇ψ)) * dΩ
                        Me[i,j] += (φ ⋅ ψ) * dΩ
                        Me[j,i] += (φ ⋅ ψ) * dΩ
                    end
                     Ke[i,i] += -1*(∇φ ⋅ (A⋅∇φ)) * dΩ
                     Me[i,i] += (φ⋅φ) * dΩ
                end
            end
            celldofs!(dofs, cell)
            assemble!(a_K, dofs, Ke)
            assemble!(a_M, dofs, Me)
        end
        return K, M
end
@time K, M = doassemble(cv, dh,rot_double_gyre2)
#@time apply!(K, dbc)
#apply!(M, dbc)
@time λ, v = eigs(K,M,which=:SM)

index = sortperm(real.(λ))[end-1]
GR.title("Eigenvector with eigenvalue $(λ[index])")
GR.contourf(reshape(real(v[:,index]),m+1,m+1),colormap=GR.COLORMAP_JET)
#savefig("output.png")
