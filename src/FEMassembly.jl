# Strongly inspired by an example provided on Ferrite's github page, modified and
# extended by Nathanael Schilling

struct Stiffness end
struct Mass end

#Works in n=2 and n=3
tensorIdentity(x::Vec{dim,T}, _, p) where {dim,T} = one(SymmetricTensor{2,dim,T})

"""
    assemble(Stiffness(), ctx, A=Id, p=nothing; bdata=BoundaryData())

Assemble the stiffness-matrix for a symmetric bilinear form
```math
a(u,v) = \\int \\nabla u(x)\\cdot A(x)\\nabla v(x)f(x) dx.
```
The integral is approximated using numerical quadrature. `A` is a function that returns a
`SymmetricTensor{2,dim}` object and must have one of the following signatures:

* `A(x::Vector{Float64})`;
* `A(x::Vec{dim})`;
* `A(x::Vec{dim}, index::Int, p)`. Here, `x` is equal to `ctx.quadrature_points[index]`,
  and `p` is some parameter, think of some precomputed object that is indexed via `index`.

The ordering of the result is in dof order, except that boundary conditions from `bdata` are
applied. The default is natural (homogeneous Neumann) boundary conditions.
"""
function assemble(::Stiffness, ctx::GridContext; A=tensorIdentity, p=nothing, bdata=BoundaryData())
    if A === tensorIdentity
        return _assembleStiffnessMatrix(ctx, A, p, bdata = bdata)
    elseif !isempty(methods(A, (Vec,)))
        return _assembleStiffnessMatrix(ctx, (qp, i, p) -> A(qp), p, bdata=bdata)
    elseif !isempty(methods(A, (Vec, Int, Any)))
        return _assembleStiffnessMatrix(ctx, (qp, i, p) -> A(qp, i, p), p, bdata=bdata)
    elseif !isempty(methods(A, (Vector{Float64},)))
        return _assembleStiffnessMatrix(ctx, (qp, i, p) -> A(convert(Vector{Float64}, qp)), p, bdata=bdata)
    end
    error("function argument `A` does not admit any of the accepted signatures")
end
@deprecate assembleStiffnessMatrix(ctx::GridContext, A=tensorIdentity, p=nothing; bdata=BoundaryData()) assemble(Stiffness(), ctx; A=A, p=p, bdata=bdata)

function _assembleStiffnessMatrix(ctx, A, p; bdata=BoundaryData())
    cv = FEM.CellScalarValues(ctx.qr, ctx.ip, ctx.ip_geom)
    dh = ctx.dh
    K = FEM.create_sparsity_pattern(dh)
    a_K = FEM.start_assemble(K)
    dofs = zeros(Int, FEM.ndofs_per_cell(dh))
    n = FEM.getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n, n)

    index = 1 # quadrature point counter

    @inbounds for (cellcount, cell) in enumerate(FEM.CellIterator(dh))
        fill!(Ke, 0)
        FEM.reinit!(cv, cell)
        for q in 1:FEM.getnquadpoints(cv) # loop over quadrature points
            Aq = A(ctx.quadrature_points[index], index, p)
            dΩ = FEM.getdetJdV(cv, q) * ctx.mass_weights[index]
            for i in 1:n
                ∇φ = FEM.shape_gradient(cv, q, i)
                for j in 1:(i-1)
                    ∇ψ = FEM.shape_gradient(cv, q, j)
                    Ke[i,j] -= (∇φ ⋅ (Aq ⋅ ∇ψ)) * dΩ
                    Ke[j,i] -= (∇φ ⋅ (Aq ⋅ ∇ψ)) * dΩ
                end
                Ke[i,i] -= (∇φ ⋅ (Aq ⋅ ∇φ)) * dΩ
            end
            index += 1
        end
        FEM.celldofs!(dofs, cell)
        FEM.assemble!(a_K, dofs, Ke)
    end
    return applyBCS(ctx, K, bdata)
end


"""
    assemble(Mass(), ctx; bdata=BoundaryData(), lumped=false)

Assemble the mass matrix
```math
M_{i,j} = \\int \\varphi_j(x) \\varphi_i(x) f(x)d\\lambda^d
```
The integral is approximated using numerical quadrature. The values of `f(x)` are taken from
`ctx.mass_weights`, and should be ordered in the same way as `ctx.quadrature_points`.

The result is ordered to be usable with a stiffness matrix with boundary data `bdata`.

Returns a lumped mass matrix if `lumped=true`.

# Example
```
ctx.mass_weights = map(f, ctx.quadrature_points)
M = assemble(Mass(), ctx)
```
"""
function assemble(::Mass, ctx::GridContext; bdata=BoundaryData(), lumped=false)
    cv = FEM.CellScalarValues(ctx.qr, ctx.ip, ctx.ip_geom)
    dh = ctx.dh
    M = FEM.create_sparsity_pattern(dh)
    a_M = FEM.start_assemble(M)
    dofs = zeros(Int, FEM.ndofs_per_cell(dh))
    n = FEM.getnbasefunctions(cv)         # number of basis functions
    Me = zeros(n, n)   # Local stiffness and mass matrix
    index = 1

    @inbounds for (cellcount, cell) in enumerate(FEM.CellIterator(dh))
        fill!(Me, 0)
        FEM.reinit!(cv, cell)
        for q in 1:FEM.getnquadpoints(cv) # loop over quadrature points
            dΩ = FEM.getdetJdV(cv, q) * ctx.mass_weights[index]
            for i in 1:n
                φ = FEM.shape_value(cv, q, i)
                for j in 1:(i-1)
                    ψ = FEM.shape_value(cv, q, j)
                    scalprod = (φ ⋅ ψ) * dΩ
                    Me[i,j] += scalprod
                    Me[j,i] += scalprod
                end
                Me[i,i] += (φ ⋅ φ) * dΩ
            end
            index += 1
        end
        FEM.celldofs!(dofs, cell)
        FEM.assemble!(a_M, dofs, Me)
    end

    M = applyBCS(ctx, M, bdata)
    return lumped ? spdiagm(0 => dropdims(reduce(+, M; dims=1); dims=1)) : M
end
@deprecate assembleMassMatrix(ctx::GridContext; bdata=BoundaryData(), lumped=false) assemble(Mass(), ctx; bdata=bdata, lumped=lumped)

"""
    getQuadPoints(ctx)

Compute the coordinates of all quadrature points on a grid. Helper function.
"""
function getQuadPoints(ctx::GridContext{dim}) where {dim}
    cv = FEM.CellScalarValues(ctx.qr, ctx.ip_geom)
    dh = ctx.dh
    dofs = zeros(Int, FEM.ndofs_per_cell(dh))
    result = Vec{dim,Float64}[]

    n = FEM.getnbasefunctions(cv)         # number of basis functions
    @inbounds for (cellcount, cell) in enumerate(FEM.CellIterator(dh))
        FEM.reinit!(cv, cell)
        for q in 1:FEM.getnquadpoints(cv) # loop over quadrature points
    	    q_coords = zero(Vec{dim,Float64})
            for j in 1:n
                q_coords += cell.coords[j] * cv.M[j,q]
            end
            push!(result, q_coords)
        end
        FEM.celldofs!(dofs, cell)
    end
    return result
end
