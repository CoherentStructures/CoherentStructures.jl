#(c) 2017 Nathanael Schilling
#Various utility functions

#The following function is like `map', but operates on 1d-datastructures.
#@param t::Float64 is just some number
#@param x::Float64 must have howmanytimes*basesize elements
#@param myfun is a function that takes arguments (t, x, result)
#     where t::Float64, x is an Array{Float64} of size basesize,
#       and result::Array{Float64} is of size basesize
#       myfun is assumed to return the result into the result array passed to it
#This function applies myfun consecutively to slices of x, and stores
#the result in the relevant slice of result.
#This is so that a "diagonalized" ODE with several starting values can
#be solved without having to call the ODE multiple times.
@everywhere @inline function arraymap(myfun,howmanytimes::Int64,basesize::Int64,t::Float64,x::Array{Float64},result::Array{Float64})
    @inbounds for i in 1:howmanytimes
        @views @inbounds  myfun(t,x[ 1 + (i-1)*basesize:  i*basesize],result[1 + (i - 1)*basesize: i*basesize])
    end
end

#Based on JuAFEM's WriteVTK.vtk_point_data
using JuAFEM
function fixU(dh::DofHandler, u::Vector)
    res = fill(0.0,getnnodes(dh.grid))
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        counter = 1
        offset = JuAFEM.field_offset(dh, dh.field_names[1])
       for node in getnodes(cell)
               res[node] = u[_celldofs[counter + offset]]
               counter += 1
          end
      end
      return res
  end
