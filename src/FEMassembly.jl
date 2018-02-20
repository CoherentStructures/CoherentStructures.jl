using JuAFEM

Id = one(Tensor{2,2})
function tensorIdentity(x::Vec{2},i::Int,p)
    return Id
end

function assembleStiffnessMatrix{dim}(ctx::gridContext{dim},A::Function,p=nothing)
    cv::CellScalarValues{dim} = CellScalarValues(ctx.qr, ctx.ip)
    dh::DofHandler{dim} = ctx.dh
    K = create_sparsity_pattern(dh)
    a_K = start_assemble(K)
    dofs = zeros(Int, ndofs_per_cell(dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n,n)
    index::Int = 1 #Counter to know the number of the current quadrature point

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points

            const Aqcoords::SymmetricTensor{2,2} = A(ctx.quadrature_points[index],index,p)
            const dΩ::Float64 = getdetJdV(cv,q) * ctx.mass_weights[index]
            for i in 1:n
                const ∇φ::Vec{2} = shape_gradient(cv,q,i)
                for j in 1:(i-1)
                    const ∇ψ::Vec{2} = shape_gradient(cv,q,j)
                    Ke[i,j] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ
                    Ke[j,i] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ
                end
                Ke[i,i] -= (∇φ ⋅ (Aqcoords⋅∇φ)) * dΩ
            end
            index += 1
        end
        celldofs!(dofs, cell)
        assemble!(a_K, dofs, Ke)
    end
    return K
end

function assembleMassMatrix{dim}(ctx::gridContext{dim};lumped=true)
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
    dh = ctx.dh
    cv = CellScalarValues(ctx.qr, ctx.ip)
    dofs = zeros(Int, ndofs_per_cell(dh))
    result = Vec{dim,Float64}[]

    n = getnbasefunctions(cv)         # number of basis functions
    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
    	    q_coords::Vec{dim} = zero(Vec{dim})
            for j in 1:n
                q_coords +=cell.coords[j] * cv.M[j,q]
            end
            push!(result,q_coords)
        end
        celldofs!(dofs, cell)
    end
    return result
end
