using JuAFEM
include("GridFunctions.jl")

function assembleStiff{dim}(cv::CellScalarValues,dh::DofHandler,Ditp)
    return assembleStiffnessMatrix2{dim}(cv,dh, x->Ditp[x...])
end

function assembleStiffnessMatrix{dim}(ctx::gridContext{dim}, A::Function)
    cv = CellScalarValues(ctx.qr, ctx.ip)
    return assembleStiffnessMatrix2(cv,ctx.dh,A)
end

function assembleStiffnessMatrix2{dim}(cv::CellScalarValues{dim},dh::DofHandler,A::Function)
    K = create_sparsity_pattern(ctx.dh)
    a_K = start_assemble(K)
    dofs = zeros(Int, ndofs_per_cell(ctx.dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n,n)

    @inbounds for (cellcount, cell) in enumerate(CellIterator(ctx.dh))
        fill!(Ke,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
    	    q_coords::Vec{dim} = zero(Vec{dim})
            for j in 1:n
                q_coords +=cell.coords[j] * cv.M[j,q]
            end

            const Aqcoords::Tensor{2,2} = A(q_coords)
            const dΩ::Float64 = getdetJdV(cv,q)
            for i in 1:n
                const ∇φ::Vec{2} = shape_gradient(cv,q,i)
                for j in 1:(i-1)
                    const ∇ψ::Vec{2} = shape_gradient(cv,q,j)
                    Ke[i,j] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ
                    Ke[j,i] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ
                end
                Ke[i,i] -= (∇φ ⋅ (Aqcoords⋅∇φ)) * dΩ
            end
        end
        celldofs!(dofs, cell)
        assemble!(a_K, dofs, Ke)
    end
    return K
end

function assembleMassMatrix{dim}(ctx::gridContext{dim})
    cv = CellScalarValues(ctx.qr, ctx.ip)
    return assembleMass(cv,ctx.dh)
end

function assembleMass{dim}(cv::CellScalarValues{dim},dh::DofHandler)
    M = create_sparsity_pattern(ctx.dh)
    a_M = start_assemble(M)
    dofs = zeros(Int, ndofs_per_cell(ctx.dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Me = zeros(n,n)   # Local stiffness and mass matrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(ctx.dh))
        fill!(Me,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            const dΩ = getdetJdV(cv,q)
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
        end
        celldofs!(dofs, cell)
        assemble!(a_M, dofs, Me)
    end
    return M
end
