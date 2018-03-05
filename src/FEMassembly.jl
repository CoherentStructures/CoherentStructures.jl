using JuAFEM

Id = one(SymmetricTensor{2,2})
function tensorIdentity(x::Vec{2},i::Int,p)
    return Id
end

function assembleStiffnessMatrix{dim}(ctx::gridContext{dim},A::Function=tensorIdentity,p=nothing;dirichlet_boundary=false)
    cv::CellScalarValues{dim} = CellScalarValues(ctx.qr, ctx.ip)
    dh::DofHandler{dim} = ctx.dh
    K::SparseMatrixCSC{Float64,Int64} = create_sparsity_pattern(dh)
    a_K::JuAFEM.AssemblerSparsityPattern{Float64,Int64} = start_assemble(K)
    dofs::Vector{Int} = zeros(Int, ndofs_per_cell(dh))
    n::Int64 = getnbasefunctions(cv)         # number of basis functions
    Ke::Array{Float64,2} = zeros(n,n)
    index::Int = 1 #Counter to know the number of the current quadrature point
    A_type::Int = 0 #What type of function A is.
    if A == tensorIdentity
        A_type = 3
    elseif !isempty(methods(A,(Vec{dim},)))
        A_type = 0
    elseif !isempty(methods(A,(Vec{dim},Int,Any)))
        A_type = 1
    elseif !isempty(methods(A,(Vector{Float64},)))
        A_type = 2
    else
        fail("Function parameter A does not accept types supported by assembleStiffnessMatrix")
    end

    #Note: the Float64,3 part below is important! otherwise the method becomes 30x slower
    Aqcoords::SymmetricTensor{2,2,Float64,3} = zero(SymmetricTensor{2,2})
    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            if A_type == 0
                Aqcoords = A(ctx.quadrature_points[index])
            elseif A_type == 1
                Aqcoords = A(ctx.quadrature_points[index],index,p)
            elseif A_type == 2
                Aqcoords = A(Vector{Float64}(ctx.quadrature_points[index]))
            elseif A_type == 3
                Aqcoords = Id
            end
            const dΩ::Float64 = getdetJdV(cv,q) * ctx.mass_weights[index]
            for i in 1:n
                const ∇φ::Vec{2} = shape_gradient(cv,q,i)
                for j in 1:(i-1)
                    const ∇ψ::Vec{2} = shape_gradient(cv,q,j)
                    Ke[i,j] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ
                    Ke[j,i] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ

                end
                Ke[i,i] -= (∇φ ⋅(Aqcoords ⋅ ∇φ)) * dΩ
            end
            index += 1
        end
        celldofs!(dofs, cell)
        assemble!(a_K, dofs, Ke)
    end
    if dirichlet_boundary
        return applyHomDBCS(ctx,K)
    else
        return K
    end
end

function assembleMassMatrix{dim}(ctx::gridContext{dim};lumped=true,dirichlet_boundary=false)
    cv::CellScalarValues{dim} = CellScalarValues(ctx.qr, ctx.ip)
    dh::DofHandler{dim} = ctx.dh
    M::SparseMatrixCSC{Float64,Int64} = create_sparsity_pattern(dh)
    a_M::JuAFEM.AssemblerSparsityPattern{Float64,Int64} = start_assemble(M)
    dofs::Vector{Int} = zeros(Int, ndofs_per_cell(dh))
    n::Int64 = getnbasefunctions(cv)         # number of basis functions
    Me::Array{Float64,2} = zeros(n,n)   # Local stiffness and mass matrix
    index::Int = 1

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Me,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            const dΩ = getdetJdV(cv,q)*ctx.mass_weights[index]
            for i in 1:n
                const φ = shape_value(cv,q,i)
                for j in 1:(i-1)
                    const ψ = shape_value(cv,q,j)
                    const scalprod = (φ ⋅ ψ) * dΩ
                    Me[i,j] += scalprod
                    Me[j,i] += scalprod
                end
                Me[i,i] += (φ ⋅ φ) * dΩ
            end
            index += 1
        end
        celldofs!(dofs, cell)
        assemble!(a_M, dofs, Me)
    end

    if dirichlet_boundary
        M = applyHomDBCS(ctx,M)
    end

    if !lumped
        return M
    else
        Mlumped = speye(size(M)[1])
        for j = 1:n
            Mlumped[j,j] = sum(M[:,j])
        end

        return Mlumped
    end
end

function getQuadPoints{dim}(ctx::gridContext{dim})
    cv::CellScalarValues{dim} = CellScalarValues(ctx.qr, ctx.ip)
    dh::DofHandler{dim} = ctx.dh
    dofs::Vector{Int} = zeros(Int, ndofs_per_cell(dh))
    dofs = zeros(Int, ndofs_per_cell(dh))
    result = Vec{dim,Float64}[]

    const n::Int = getnbasefunctions(cv)         # number of basis functions
    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
    	    q_coords = zero(Vec{dim})
            for j in 1:n
                q_coords +=cell.coords[j] * cv.M[j,q]
            end
            push!(result,q_coords)
        end
        celldofs!(dofs, cell)
    end
    return result
end


#TODO: Make the following more efficient
function applyHomDBCS{dim}(ctx::gridContext{dim},K)
    dbcs = getHomDBCS(ctx)
    if !issorted(dbcs.prescribed_dofs)
        error("DBCS are not sorted")
    end
    k = length(dbcs.values)
    n = ctx.n
    Kres = spzeros(n-k,n-k)
    skip_cols = 0
    vals = nonzeros(K)
    rows = rowvals(K)
    for j = 1:n
            if j == dbcs.prescribed_dofs[skip_cols+1]
                    skip_cols += 1
                    continue
            end
            skip_rows = 1
            for i in nzrange(K,j)
                    row = rows[i]
                    while dbcs.prescribed_dofs[skip_rows+1] < row
                            skip_rows += 1
                    end
                    if dbcs.prescribed_dofs[skip_rows+1] == row
                            skip_rows += 1
                            continue
                    end
                    Kres[row - skip_rows ,j - skip_cols] = vals[i]
            end
    end
    return Kres
end
