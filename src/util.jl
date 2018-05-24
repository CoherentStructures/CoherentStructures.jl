#(c) 2017 Nathanael Schilling
#Various utility functions


"""
    arraymap!(du,u,p,t,odefun,howmanytimes,basesize)

Like `map', but operates on 1d-datastructures.
# Arguments
Apply odefun(du,u,p,t) consecutively to `howmany` subarrays of size `basesize` of `u`, and stores
the result in the relevant slice of `du`.
This is so that a "diagonalized" ODE with several starting values can
be solved without having to call the ODE solver multiple times.
"""
@inline function arraymap!(du::Array{Float64},u::Array{Float64},p,t::Float64, odefun::Function,howmanytimes::Int64,basesize::Int64)
    @inbounds for i in 1:howmanytimes
        @views @inbounds  odefun(du[1 + (i - 1)*basesize: i*basesize],u[ 1 + (i-1)*basesize:  i*basesize],p,t)
    end
end

"""
    arraymap2(u,p,t,odefun) -> StaticArrays.SVector{8}
This function is like arraymap(u,p,t,odefun, 4,2),
but du is returned as a StaticVector
"""
@inline function arraymap2(u::StaticArrays.SVector{8,T},p,t::Float64, odefun::Function)::StaticArrays.SVector{8,T} where T
    p1::StaticArrays.SVector{2,T} = odefun(StaticArrays.SVector{2,T}(u[1], u[2]),p,t)
    p2::StaticArrays.SVector{2,T} = odefun(StaticArrays.SVector{2,T}(u[3], u[4]),p,t)
    p3::StaticArrays.SVector{2,T} = odefun(StaticArrays.SVector{2,T}(u[5], u[6]),p,t)
    p4::StaticArrays.SVector{2,T} = odefun(StaticArrays.SVector{2,T}(u[7], u[8]),p,t)
    return StaticArrays.SVector{8,T}(p1[1],p1[2],p2[1],p2[2],p3[1],p3[2],p4[1],p4[2])
end

"""
    arraymap3(u,p,t,odefun) -> StaticArrays.SVector{18}
This function is like arraymap(u,pt,odefun,6,3)
but du is returned as a StaticVector
"""
@inline function arraymap3(u::StaticArrays.SVector{18,T},p,t::Float64, odefun::Function)::StaticArrays.SVector{18,T} where T
    p1::StaticArrays.SVector{3,T} = odefun(StaticArrays.SVector{3,T}(u[1], u[2], u[3]),p,t)
    p2::StaticArrays.SVector{3,T} = odefun(StaticArrays.SVector{3,T}(u[4], u[5], u[6]),p,t)
    p3::StaticArrays.SVector{3,T} = odefun(StaticArrays.SVector{3,T}(u[7], u[8], u[9]),p,t)
    p4::StaticArrays.SVector{3,T} = odefun(StaticArrays.SVector{3,T}(u[10], u[11], u[12]),p,t)
    p5::StaticArrays.SVector{3,T} = odefun(StaticArrays.SVector{3,T}(u[13], u[14], u[15]),p,t)
    p6::StaticArrays.SVector{3,T} = odefun(StaticArrays.SVector{3,T}(u[16], u[17], u[18]),p,t)
    return StaticArrays.SVector{18,T}(p1[1],p1[2],p1[3],p2[1],p2[2],p2[3],p3[1],p3[2],p3[3],p4[1],p4[2],p4[3],p5[1],p5[2],p5[3],p6[1],p6[2],p6[3])
end


"""
    tensor_invariants(T::AbstractArray{Tensors.SymmetricTensor}) -> λ₁, λ₂, ξ₁, ξ₂, traceT, detT

Returns pointwise invariants of the 2D symmetric tensor field `T`, i.e.,
smallest and largest eigenvalues, corresponding eigenvectors, trace and determinant.
# Example
```
T = [Tensors.SymmetricTensor{2,2}(rand(3)) for i in 1:10, j in 1:20]
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(T)
```
All variables have the same array arrangement as `T`; e.g., `λ₁` is a
10x20 array with scalar entries.
"""

function tensor_invariants(T::AbstractArray{Tensors.SymmetricTensor{2,2,S,3}}) where S <: Real
    Efact = eigfact.(T)
    λ₁ = [ev[1] for ev in eigvals.(Efact)]
    λ₂ = [ev[2] for ev in eigvals.(Efact)]
    ξ₁ = [ev[:,1] for ev in eigvecs.(Efact)]
    ξ₂ = [ev[:,2] for ev in eigvecs.(Efact)]
    traceT = trace.(T)
    detT = det.(T)
    return λ₁, λ₂, ξ₁, ξ₂, traceT, detT
end


"""
    dof2node(ctx,u)

Interprets `u` as an array of coefficients ordered in dof order,
and reorders them to be in node order.
"""
function dof2node(ctx::abstractGridContext{dim} ,u::Vector) where {dim}
   n = ctx.n
   res = fill(0.0,JuAFEM.getnnodes(ctx.grid))
   for node in 1:n
           res[node] = u[ctx.node_to_dof[node]]
      end
  return res
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
res = kmeans(v[:,1:numclusters]',numclusters+1)
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
const e1 = Tensors.basevec(Tensors.Vec{2},1)
const e2 = Tensors.basevec(Tensors.Vec{2},2)




function rawInvCGTensor(args...;kwargs...)
    result = invCGTensor(args...;kwargs...)
    return result[1,1], result[1,2], result[2,2]
end


function AFromPrecomputedRaw(x,index,q)
    @views return Tensors.SymmetricTensor{2,2}((q[1])[3*(index-1)+1 : 3*(index-1)+3])
end


#The rhs for an ODE on interpolated vector fields
#The interpolant is passed via the p argument

#TODO: think of adding @inbounds here
function interp_rhs!(du::AbstractArray{T},u::AbstractArray{T},p,t::T) where {T <: Real}
    du[1] = p[1][u[1],u[2],t]
    du[2] = p[2][u[1],u[2],t]
end

"""
    interp_rhs(u,p,t) -> StaticArrays.SVector{2}

Defines a 2D vector field that is readily usable for trajectory integration from
vector field interpolants of the x- and y-direction, resp. It assumes that the
interpolants are provided as a 2-tuple `(UI, VI)` via the parameter `p`. Here,
`UI` and `VI` are the interpolants for the x- and y-components of the velocity
field.
"""

function interp_rhs(u,p,t)
    du1 = p[1][u[1],u[2],t]
    du2 = p[2][u[1],u[2],t]
    return StaticArrays.SVector{2}(du1, du2)
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
                    "regular P2 quadrilateral grid"]

    if ctx.gridType ∉ supportedRegularGridTypes
        error("Mesh width for this grid type not yet implemented")
    end

    hx = (ctx.spatialBounds[2][1] - ctx.spatialBounds[1][1])/(ctx.numberOfPointsInEachDirection[1] - 1)
    hy = (ctx.spatialBounds[2][2] - ctx.spatialBounds[1][2])/(ctx.numberOfPointsInEachDirection[1] - 1)

    return sqrt(hx^2 + hy^2)
end

"""
    distmod(a,b,c)

return the distance from a to b, where the distace is taken to be modulo
"""
function distmod(a,b,c)
    diff = mod( (a -b),c)
    return min(diff, c- diff)
end
