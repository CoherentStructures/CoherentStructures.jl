#(c) 2017 Nathanael Schilling
#Various utility functions


#The following function is like `map', but operates on 1d-datastructures.
#@param t::Float64 is just some number
#@param u::Float64 must have howmanytimes*basesize elements
#@param odefun is a function that takes arguments (du,u,p,t)
#     where t::Float64, x is an Array{Float64} of size basesize,
#       and du::Array{Float64} is of size basesize
#       odefun is assumed to return the result into the result array passed to it
#This function applies myfun consecutively to slices of u, and stores
#the result in the relevant slice of result.
#This is so that a "diagonalized" ODE with several starting values can
#be solved without having to call the ODE solver multiple times.
@inline function arraymap(du::Array{Float64},u::Array{Float64},p,t::Float64, odefun::Function,howmanytimes::Int64,basesize::Int64)
    @inbounds for i in 1:howmanytimes
        @views @inbounds  odefun(du[1 + (i - 1)*basesize: i*basesize],u[ 1 + (i-1)*basesize:  i*basesize],p,t)
    end
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


#Unit Vectors in R^2
e1 = basevec(Vec{2},1)
e2 = basevec(Vec{2},2)
