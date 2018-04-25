#(c) 2017 Nathanael Schilling
#Various utility functions


"""
The following function is like `map', but operates on 1d-datastructures.
#Arguments
t::Float64 - time (passed to odefun)
u::Float64[] -  must have howmanytimes*basesize elements
odefun is a function that takes arguments (du,u,p,t)
     where t::Float64, x is an Array{Float64} of size basesize,
       and du::Array{Float64} is of size basesize
       odefun is assumed to return the result into the result array passed to it
This function applies myfun consecutively to slices of u, and stores
the result in the relevant slice of result.
This is so that a "diagonalized" ODE with several starting values can
be solved without having to call the ODE solver multiple times.
"""
@inline function arraymap!(du::Array{Float64},u::Array{Float64},p,t::Float64, odefun::Function,howmanytimes::Int64,basesize::Int64)
    @inbounds for i in 1:howmanytimes
        @views @inbounds  odefun(du[1 + (i - 1)*basesize: i*basesize],u[ 1 + (i-1)*basesize:  i*basesize],p,t)
    end
end

# TODO: this is plainly assuming 2D-systems, generalize to ND-systems
@inline @inbounds function arraymap(u::StaticVector{8,Float64},p,t::Float64, odefun::Function)::SVector{8,Float64}
    p1::SVector{2,Float64} = odefun((@SVector Float64[u[1], u[2]]),p,t)
    p2::SVector{2,Float64} = odefun((@SVector Float64[u[3], u[4]]),p,t)
    p3::SVector{2,Float64} = odefun((@SVector Float64[u[5], u[6]]),p,t)
    p4::SVector{2,Float64} = odefun((@SVector Float64[u[7], u[8]]),p,t)
    @SVector [p1[1],p1[2],p2[1],p2[2],p3[1],p3[2],p4[1],p4[2]]
end

"""
`tensor_invariants(T::AbstractArray{Tensors.SymmetricTensor})`
computes pointwise invariants of the 2D tensor field `T`, i.e.,
smallest and largest eigenvalues, corresponding eigenvectors, trace and determinant.
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


#Reorders an array of values corresponding to dofs from a DofHandler
#To the order which the nodes of the grid would be
function dof2U(ctx::abstractGridContext{dim} ,u::Vector) where {dim}
   n = ctx.n
   res = fill(0.0,getnnodes(ctx.grid))
   for node in 1:n
           res[node] = u[ctx.node_to_dof[node]]
      end
  return res
end

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
e1 = basevec(Vec{2},1)
e2 = basevec(Vec{2},2)




function rawInvCGTensor(args...;kwargs...)
    result = invCGTensor(args...;kwargs...)
    return result[1,1], result[1,2],result[2,2]
end


function AFromPrecomputedRaw(x,index,q)
    @views return SymmetricTensor{2,2}((q[1])[3*(index-1)+1 : 3*(index-1)+3])
end


#The rhs for an ODE on interpolated vector fields
#The interpolant is passed via the p argument

#TODO: think of adding @inbounds here
function interp_rhs!(du::AbstractArray{T},u::AbstractArray{T},p,t::T) where {T <: Real}
    du[1] = p[1][u[1],u[2],t]
    du[2] = p[2][u[1],u[2],t]
end

function interp_rhs(u,p,t)
    du1 = p[1][u[1],u[2],t]
    du2 = p[2][u[1],u[2],t]
    return SVector{2}(du1, du2)
end

#Returns true for all inputs. This is the default function for inbounds checking in plot_ftle
function always_true(x,y,p)
    return true
end
