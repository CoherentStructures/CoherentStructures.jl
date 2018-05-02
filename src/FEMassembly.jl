JFM = JuAFEM

Id = one(Tensors.SymmetricTensor{2,2})
function tensorIdentity(x::Tensors.Vec{2},i::Int,p)
    return Id
end

doc"""
    assembleStiffnessMatrix(ctx,A,[p; bdata])

Assemble the stiffness-matrix for a symmetric bilinear form
```math
a(u,v) = \int \nabla u(x)\cdot A(x)\nabla v(x)f(x) dx
```
The integral is approximated using quadrature.
`A` is a function that returns a `Tensors.SymmetricTensor` and has one of the following forms:
   * `A(x::Vector{Float64})`
   * `A(x::Vec{dim})`
   * `A(x::Vec{dim}, index::Int, p)`. Here x is equal to `ctx.quadrature_points[index]`, and `p` is that which is passed to `assembleStiffnessMatrix`

The ordering of the result is in dof order, except that boundary conditions from `bdata` are applied. The default is natural boundary conditions.
"""
function assembleStiffnessMatrix{dim}(
        ctx::gridContext{dim},
        A::Function=tensorIdentity,
        p=nothing;
        bdata=boundaryData() #Default to natural BCs
        )
    cv::JFM.CellScalarValues{dim} = JFM.CellScalarValues(ctx.qr, ctx.ip)
    dh::JFM.DofHandler{dim} = ctx.dh
    K::SparseMatrixCSC{Float64,Int64} = JFM.create_sparsity_pattern(dh)
    a_K::JFM.AssemblerSparsityPattern{Float64,Int64} = JFM.start_assemble(K)
    dofs::Vector{Int} = zeros(Int, JFM.ndofs_per_cell(dh))
    n::Int64 = JFM.getnbasefunctions(cv)         # number of basis functions
    Ke::Array{Float64,2} = zeros(n,n)
    index::Int = 1 #Counter to know the number of the current quadrature point
    A_type::Int = 0 #What type of function A is.
    if A == tensorIdentity
        A_type = 3
    elseif !isempty(methods(A,(Tensors.Vec{dim},)))
        A_type = 0
    elseif !isempty(methods(A,(Tensors.Vec{dim},Int,Any)))
        A_type = 1
    elseif !isempty(methods(A,(Vector{Float64},)))
        A_type = 2
    else
        fail("Function parameter A does not accept types supported by assembleStiffnessMatrix")
    end

    #Note: the Float64,3 part below is important! otherwise the method becomes 30x slower
    Aqcoords::Tensors.SymmetricTensor{2,2,Float64,3} = zero(Tensors.SymmetricTensor{2,2})
    @inbounds for (cellcount, cell) in enumerate(JFM.CellIterator(dh))
        fill!(Ke,0)
        JFM.reinit!(cv,cell)
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
            const dΩ::Float64 = JFM.getdetJdV(cv,q) * ctx.mass_weights[index]
            for i in 1:n
                const ∇φ::Tensors.Vec{2} = JFM.shape_gradient(cv,q,i)
                for j in 1:(i-1)
                    const ∇ψ::Tensors.Vec{2} = JFM.shape_gradient(cv,q,j)
                    Ke[i,j] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ
                    Ke[j,i] -= (∇φ ⋅ (Aqcoords⋅∇ψ)) * dΩ

                end
                Ke[i,i] -= (∇φ ⋅(Aqcoords ⋅ ∇φ)) * dΩ
            end
            index += 1
        end
        JFM.celldofs!(dofs, cell)
        JFM.assemble!(a_K, dofs, Ke)
    end
    return applyBCS(ctx,K,bdata)
end


doc"""
    assembleMassMatrix(ctx;[bdata,lumped=false])

Assemble the mass matrix
```math
M_{i,j} = \int \varphi_j(x) \varphi_i(x) f(x)d\lambda^d
```
The integral is approximated using numerical quadrature.
The values of `f(x)` are taken from `ctx.mass_weights`, and should be ordered in the same way as `ctx.quadrature_points`

The result is ordered in a way so as to be usable with a stiffness matrix
with boundary data `bdata`.

Returns a lumped mass matrix if `lumped==true`.

# Example
```
ctx.mass_weights = map(f, ctx.quadrature_points)
M = assembleMassMatrix(ctx)
```
"""
function assembleMassMatrix{dim}(
        ctx::gridContext{dim};
        bdata=boundaryData(),
        lumped=false,
        )
    cv::JFM.CellScalarValues{dim} = JFM.CellScalarValues(ctx.qr, ctx.ip)
    dh::JFM.DofHandler{dim} = ctx.dh
    M::SparseMatrixCSC{Float64,Int64} = JFM.create_sparsity_pattern(dh)
    a_M::JuAFEM.AssemblerSparsityPattern{Float64,Int64} = JFM.start_assemble(M)
    dofs::Vector{Int} = zeros(Int, JFM.ndofs_per_cell(dh))
    n::Int64 = JFM.getnbasefunctions(cv)         # number of basis functions
    Me::Array{Float64,2} = zeros(n,n)   # Local stiffness and mass matrix
    index::Int = 1

    @inbounds for (cellcount, cell) in enumerate(JFM.CellIterator(dh))
        fill!(Me,0)
        JFM.reinit!(cv,cell)
        for q in 1:JFM.getnquadpoints(cv) # loop over quadrature points
            const dΩ::Float64 = JFM.getdetJdV(cv,q)*ctx.mass_weights[index]
            for i in 1:n
                const φ::Float64 = JFM.shape_value(cv,q,i)
                for j in 1:(i-1)
                    const ψ::Float64 = JFM.shape_value(cv,q,j)
                    const scalprod::Float64 = (φ ⋅ ψ) * dΩ
                    Me[i,j] += scalprod
                    Me[j,i] += scalprod
                end
                Me[i,i] += (φ ⋅ φ) * dΩ
            end
            index += 1
        end
        JFM.celldofs!(dofs, cell)
        JFM.assemble!(a_M, dofs, Me)
    end

    M = applyBCS(ctx,M,bdata)

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

"""
    getQuadPointsPoints(ctx)

Compute the coordinates of all quadrature points on a grid.
Helper function.
"""
function getQuadPoints{dim}(ctx::gridContext{dim})
    cv::JFM.CellScalarValues{dim} = JFM.CellScalarValues(ctx.qr, ctx.ip)
    dh::JFM.DofHandler{dim} = ctx.dh
    dofs::Vector{Int} = zeros(Int, JFM.ndofs_per_cell(dh))
    dofs = zeros(Int, JFM.ndofs_per_cell(dh))
    result = Tensors.Vec{dim,Float64}[]

    const n::Int = JFM.getnbasefunctions(cv)         # number of basis functions
    @inbounds for (cellcount, cell) in enumerate(JFM.CellIterator(dh))
        JuAFEM.reinit!(cv,cell)
        for q in 1:JFM.getnquadpoints(cv) # loop over quadrature points
    	    q_coords = zero(Tensors.Vec{dim})
            for j in 1:n
                q_coords +=cell.coords[j] * cv.M[j,q]
            end
            push!(result,q_coords)
        end
        JFM.celldofs!(dofs, cell)
    end
    return result
end
