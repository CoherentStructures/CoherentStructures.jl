using JuAFEM
using Plots

m = 100 # number of cell in one direction
grid = generate_grid(Quadrilateral, (m,m))
addnodeset!(grid, "boundary", x -> abs(x[1]) ≈ 1 ||  abs(x[2]) ≈ 1)

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)

cv = CellScalarValues(qr, ip)
dh = DofHandler(grid)
push!(dh, :T, 1)
close!(dh)

dbc = DirichletBoundaryConditions(dh)
add!(dbc, :T, getnodeset(grid, "boundary"), (x,t) -> 0.0)
close!(dbc)
update!(dbc, 0.0)

function doassemble{dim}(cv::CellScalarValues{dim}, dh::DofHandler)
    K = create_sparsity_pattern(dh)
    a_K = start_assemble(K)
    M = create_sparsity_pattern(dh)
    a_M = start_assemble(M)
    dofs = zeros(Int, ndofs_per_cell(dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n,n)
    Me = zeros(n,n)   # Local stiffness and mass matrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke,0)
        fill!(Me,0)
        reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            dΩ = getdetJdV(cv,q)
            for i in 1:n
                φ = shape_value(cv,q,i)
                ∇φ = shape_gradient(cv,q,i)
                for j in 1:n
                    ψ = shape_value(cv,q,i)
                    ∇ψ = shape_gradient(cv,q,j)
                    Ke[i,j] += (∇φ ⋅ ∇ψ) * dΩ
                    Me[i,j] += (φ ⋅ ψ) * dΩ
                end
            end
        end
        celldofs!(dofs, cell)
        assemble!(a_K, dofs, Ke)
        assemble!(a_M, dofs, Me)
    end
    return K, M
end

@time K, M = doassemble(cv, dh)
@time apply!(K, dbc)
apply!(M, dbc)
@time λ, v = eigs(K,M,which=:SM)

plot(contour(reshape(real(v[:,2]),m+1,m+1),fill=true))