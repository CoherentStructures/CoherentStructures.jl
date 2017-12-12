using JuAFEM

function assembleStiff{dim}(cv::CellScalarValues{dim}, dh::DofHandler, Ditp)
    K = create_sparsity_pattern(dh)
    a_K = start_assemble(K)
    dofs = zeros(Int, ndofs_per_cell(dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n,n)

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke,0)
        JuAFEM.reinit!(cv,cell)
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            q_coords = zero(Vec{dim})
            for j in 1:n
                q_coords +=cell.coords[j] * cv.M[j,q]
            end
            const D̅ = Ditp[q_coords...] #SymmetricTensor{2,2}(eye(2,2))
            # D̅ = mean(PullBackDiffTensor(Ditp,Array(q_coords),collect(linspace(0,3456000,401)),1.e-6,SymmetricTensor{2,2}([2., 0., 1/2]))) #SymmetricTensor{2,2}(eye(2,2))
            const dΩ = getdetJdV(cv,q)
            for i in 1:n
                const ∇φ = shape_gradient(cv,q,i)
                for j in 1:(i-1)
                    const ∇ψ = shape_gradient(cv,q,j)
                    const gradprod = -1.0*(∇φ ⋅ (D̅⋅∇ψ)) * dΩ
                    Ke[i,j] += gradprod
                    Ke[j,i] += gradprod
                end
                Ke[i,i] += -1.0*(∇φ ⋅ (D̅⋅∇φ)) * dΩ
            end
        end
        celldofs!(dofs, cell)
        assemble!(a_K, dofs, Ke)
    end
    return K
end

function assembleMass{dim}(cv::CellScalarValues{dim}, dh::DofHandler)
    M = create_sparsity_pattern(dh)
    a_M = start_assemble(M)
    dofs = zeros(Int, ndofs_per_cell(dh))
    n = getnbasefunctions(cv)         # number of basis functions
    Me = zeros(n,n)   # Local stiffness and mass matrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
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
