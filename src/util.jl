#(c) 2017 Nathanael Schilling
#Various utility functions

const SA = StaticArrays
"""
    arraymap!(du, u, p, t, odefun, N, dim)

Like `map`, but operates on 1d-datastructures.

# Arguments
Apply `odefun` consecutively to `N` subarrays of size `dim` of `u`, and store
the result in the corresponding slice of `du`.
This is so that a decoupled ODE system with several initial values can
be solved without having to call the ODE solver multiple times.
"""
@inline function arraymap!(du::Vector, u::Vector, p, t,
                            odefun, N::Int, dim::Int)
    @inbounds for i in 1:N
        @views odefun(du[1+(i - 1)*dim:i*dim], u[1+(i-1)*dim:i*dim], p, t)
    end
end

"""
    arraymap2(u, p, t, odefun) -> SVector{8}
This function is like `arraymap!(du, u, p, t, odefun, 4, 2)``,
but `du` is returned as a StaticVector.
"""
@inline function arraymap2(u::SA.SVector{8,T}, p, t, odefun)::SA.SVector{8,T} where T
    p1::SA.SVector{2,T} = odefun(SA.SVector{2,T}(u[1], u[2]), p, t)
    p2::SA.SVector{2,T} = odefun(SA.SVector{2,T}(u[3], u[4]), p, t)
    p3::SA.SVector{2,T} = odefun(SA.SVector{2,T}(u[5], u[6]), p, t)
    p4::SA.SVector{2,T} = odefun(SA.SVector{2,T}(u[7], u[8]), p, t)
    return SA.SVector{8,T}(p1[1],p1[2],p2[1],p2[2],p3[1],p3[2],p4[1],p4[2])
end

"""
    arraymap3(u, p, t, odefun) -> SVector{18}
This function is like `arraymap!(du, u, pt, odefun, 6, 3)
but `du` is returned as a StaticVector.
"""
@inline function arraymap3(u::SA.SVector{18,T}, p, t, odefun)::SA.SVector{18,T} where T
    p1::SA.SVector{3,T} = odefun(SA.SVector{3}(u[1], u[2], u[3]), p, t)
    p2::SA.SVector{3,T} = odefun(SA.SVector{3}(u[4], u[5], u[6]), p, t)
    p3::SA.SVector{3,T} = odefun(SA.SVector{3}(u[7], u[8], u[9]), p, t)
    p4::SA.SVector{3,T} = odefun(SA.SVector{3}(u[10], u[11], u[12]), p, t)
    p5::SA.SVector{3,T} = odefun(SA.SVector{3}(u[13], u[14], u[15]), p, t)
    p6::SA.SVector{3,T} = odefun(SA.SVector{3}(u[16], u[17], u[18]), p, t)
    return SA.SVector{18,T}(p1[1],p1[2],p1[3],p2[1],p2[2],p2[3],p3[1],p3[2],p3[3],p4[1],p4[2],p4[3],p5[1],p5[2],p5[3],p6[1],p6[2],p6[3])
end


"""
    tensor_invariants(T) -> λ₁, λ₂, ξ₁, ξ₂, traceT, detT

Returns pointwise invariants of the 2D symmetric tensor field `T`, i.e.,
smallest and largest eigenvalues, corresponding eigenvectors, trace and determinant.
# Example
```
T = [SymmetricTensor{2,2}(rand(3)) for i in 1:10, j in 1:20]
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(T)
```
All output variables have the same array arrangement as `T`; e.g., `λ₁` is a
10x20 array with scalar entries.
"""
function tensor_invariants(T::SymmetricTensor{2,2,S,3}) where S <: Real
    E = LinearAlgebra.eigen(T)
    λ₁ = LinearAlgebra.eigvals(E)[1]
    λ₂ = LinearAlgebra.eigvals(E)[2]
    ξ₁ = LinearAlgebra.eigvecs(E)[:,1]
    ξ₂ = LinearAlgebra.eigvecs(E)[:,2]
    traceT = LinearAlgebra.tr(T)
    detT = LinearAlgebra.det(T)
    return λ₁, λ₂, ξ₁, ξ₂, traceT, detT
end
function tensor_invariants(T::AbstractArray{SymmetricTensor{2,2,S,3}}) where S <: Real
    E = LinearAlgebra.eigen.(T)
    λ₁ = [ev[1] for ev in LinearAlgebra.eigvals.(E)]
    λ₂ = [ev[2] for ev in LinearAlgebra.eigvals.(E)]
    ξ₁ = [ev[:,1] for ev in LinearAlgebra.eigvecs.(E)]
    ξ₂ = [ev[:,2] for ev in LinearAlgebra.eigvecs.(E)]
    traceT = LinearAlgebra.tr.(T)
    detT = LinearAlgebra.det.(T)
    return λ₁, λ₂, ξ₁, ξ₂, traceT, detT
end

"""
    dof2node(ctx,u)

Interprets `u` as an array of coefficients ordered in dof order,
and reorders them to be in node order.
"""
function dof2node(ctx::abstractGridContext{dim}, u::Vector) where {dim}
   # n = ctx.n
   # res = fill(0.0, JuAFEM.getnnodes(ctx.grid))
   # for node in 1:n
   #         res[node] = u[ctx.node_to_dof[node]]
   #    end
   # TODO: isn't this just reordering? ctx.n == getnnodes(ctx.grid) by construction
  return u[ctx.node_to_dof[1:ctx.n]]
end

"""
    kmeansresult2LCS(kmeansresult)

Takes the result-object from kmeans(),
and returns a coefficient vector (in dof order)
corresponding to (interpolated) indicator functions.

# Example
```
v, λ = eigs(K,M)
numclusters = 5
res = kmeans(permutedims(v[:,1:numclusters]),numclusters+1)
u = kmeansresult2LCS(res)
plot_u(ctx,u)
```
"""
function kmeansresult2LCS(kmeansresult)
    n = length(kmeansresult.assignments)
    numclusters = size(kmeansresult.centers)[2]
    u = zeros(n,numclusters)
    for j in 1:n
        for i in 1:numclusters
            u[j,i] = kmeansresult.assignments[j] == i ? 1.0 : 0.0
        end
    end
    return u
end


#Unit Vectors in R^2
const e1 = Tensors.basevec(Vec{2}, 1)
const e2 = Tensors.basevec(Vec{2}, 2)

function rawInvCGTensor(args...; kwargs...)
    result = invCGTensor(args...; kwargs...)
    return result[1,1], result[1,2], result[2,2]
end

function AFromPrecomputedRaw(x, index, q)
    @views return SymmetricTensor{2,2}((q[1])[3*(index-1)+1 : 3*(index-1)+3])
end


#Returns true for all inputs. This is the default function for inbounds checking in plot_ftle
function always_true(x,y,p)
    return true
end


"""
    getH(ctx)

Return the mesh width of a regular grid.
"""
function getH(ctx::abstractGridContext)
    supportedRegularGridTypes = ["regular triangular grid",
                    "regular P2 triangular grid",
                    "regular Delaunay grid",
                    "regular P2 Delaunay grid",
                    "regular quadrilateral grid",
                    "regular P2 quadrilateral grid",
                    ]

    supported1DGridTypes = ["regular 1d grid",
                    "regular 1d P2 grid"]

    if ctx.gridType ∈ supportedRegularGridTypes
        hx = (ctx.spatialBounds[2][1] - ctx.spatialBounds[1][1])/(ctx.numberOfPointsInEachDirection[1] - 1)
        hy = (ctx.spatialBounds[2][2] - ctx.spatialBounds[1][2])/(ctx.numberOfPointsInEachDirection[1] - 1)

        return sqrt(hx^2 + hy^2)
    elseif ctx.gridType ∈ supported1DGridTypes
        hx = (ctx.spatialBounds[2][1] - ctx.spatialBounds[2][1])/(ctx.numberOfPointsInEachDirection[1] - 1)
        return hx
    else
        error("Mesh width for this grid type not yet implemented")
    end
end


#divrem that returns the first value as an Int
#TODO: maybe optimize this?
function gooddivrem(x, y)
        a, b = divrem(x, y)
        return Int(a), b
end

function goodmod(a,b)
    return Base.mod(a,b)
end


function gooddivrem(x::ForwardDiff.Dual, y)
        a, b = divrem(x, y)
        if !iszero(b)
            return Int(ForwardDiff.value(a)), b
        else
            aret = Int(ForwardDiff.value(a))
            return aret, x - aret * y
        end
end


function goodmod(x::ForwardDiff.Dual,y)
    a,b = gooddivrem(x,y)
    if b < 0
        return b+y
    else
        return b
    end
end

#TODO: Document this
 function unzip(A::Array{T}) where T
    res = map(x -> x[], T.parameters)
    res_len = length(res)
    for t in A
        for i in 1:res_len
            push!(res[i], t[i])
        end
    end
    res
end
