using JuAFEM

grid = generate_grid(Triangle, (20,20))
addnodeset!(grid, "boundary", x -> abs(x[1]) ≈ 1 ||  abs(x[2]) ≈ 1); #Why not where x[1] or x[2] == 0 ???? TODO: Find out what this does exactly

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}() # Change 1 here for higher order Lagrange methods
qr = QuadratureRule{dim, RefTetrahedron}(2) # Change here for lower order quadrature method
cellvalues = CellScalarValues(qr, ip);


dh = DofHandler(grid)
push!(dh, :density, 1) 
close!(dh)

Mass_matrix = create_sparsity_pattern(dh);
fill!(Mass_matrix.nzval, 0.0); 

function get_mass_matrix{dim}(cellvalues::CellScalarValues{dim}, Mass_matrix::SparseMatrixCSC, dh::DofHandler)
    b = 1.0
    f = zeros(ndofs(dh))
    assembler = start_assemble(Mass_matrix,f)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    Mm = zeros(n_basefuncs, n_basefuncs)#Local mass-matrix
    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
	fill!(Mm,0)
	reinit!(cellvalues,cell)
	for q_point in 1:getnquadpoints(cellvalues)
	    dΩ=getdetJdV(cellvalues, q_point)
	    for i in 1:n_basefuncs
		ϕi = shape_value(cellvalues, q_point,i)
		for j in 1:n_basefuncs
		    ϕj = shape_value(cellvalues,q_point,j)
		    Mm[i,j] += dΩ*ϕi*ϕj
		end
	    end
	end
	celldofs!(global_dofs,cell)
	assemble!(assembler, global_dofs, Mm)
    end
    return Mass_matrix,f
end

@time Mass_matrix = get_mass_matrix(cellvalues, Mass_matrix, dh)

