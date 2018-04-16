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

@inline function arraymap(u,p,t,odefun) # TODO: this is plainly assuming 2D-systems, generalize to ND-systems
    du1 = odefun(u[1:2],p,t)
    du2 = odefun(u[3:4],p,t)
    du3 = odefun(u[5:6],p,t)
    du4 = odefun(u[7:8],p,t)
    return StaticArrays.SVector{8}(du1[1], du1[2], du2[1],du2[2], du3[1], du3[2], du4[1], du4[2])
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

#Returns true for all inputs. This is the default for plot_ftle
function always_true(x,y,p)
    return true
end

 function fast_trilinear_earth_interpolate(du::AbstractArray{T},u,p,tin) where {T<:Real}
    Us = p[1]
    Vs = p[2]
    nx::Int64 = size(Us)[1]
    ny::Int64 = size(Us)[2]
    nt::Int64 = size(Us)[3]
    #Get the spatial bounds from p
    ll1::Float64,ur1::Float64  = p[3]
    ll2::Float64,ur2::Float64 = p[4]
    t0::Float64,tf::Float64 = p[5]
    t::Float64 = tin
    if t > tf
        t = tf
    end
    if t < t0
        t = t0
    end
    #Just divrem, but casts the first result to Int
    function gooddivrem(x,y)
        a,b = divrem(x,y)
        return Int(a), b
    end
    function gooddivrem(x::ForwardDiff.Dual, y)
        a,b = divrem(x,y)
        return Int(ForwardDiff.value(a)), b
    end
    xindex::Int64, xcoord::T = gooddivrem((mod((u[1] - ll1), 360)*nx)/360.0,1)
    yindex::Int64, ycoord::T = gooddivrem((mod((u[2] - ll2), 180)*ny)/180.0,1)
    #
    tindex::Int64, tcoord::Float64 = gooddivrem((nt-1)*(t-t0)/(tf-t0),1)
    tindex += 1
    #Make sure we don't go out of bounds
    tpp = tindex + 1
    if tpp > nt
        tpp = nt
    end
    #Actual interpolation for u
    r1u::T =  Us[xindex+1,yindex + 1,tindex ]*(1 - xcoord) +            Us[ (xindex + 1) % nx + 1,yindex + 1, tindex ]*xcoord
    r2u::T =  Us[xindex+1,(yindex + 1) % ny + 1,tindex ]*(1 - xcoord) + Us[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tindex ]*xcoord
    r3u::T =  Us[xindex+1,yindex + 1,tpp ]*(1 - xcoord) +               Us[ (xindex + 1) % nx + 1,yindex + 1, tpp ]*xcoord
    r4u::T =  Us[xindex+1,(yindex + 1) % ny + 1,tpp ]*(1 - xcoord) +    Us[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tpp ]*xcoord
    du[1] =  (
        (1-tcoord)*((1-ycoord)*r1u + ycoord*r2u)
         + tcoord*((1-ycoord)*r3u + ycoord*r4u))
    #For v
    r1v::T =  Vs[xindex+1,yindex + 1,tindex ]*(1 - xcoord) +            Vs[ (xindex + 1) % nx + 1,yindex + 1, tindex ]*xcoord
    r2v::T =  Vs[xindex+1,(yindex + 1) % ny + 1,tindex ]*(1 - xcoord) + Vs[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tindex ]*xcoord
    r3v::T =  Vs[xindex+1,yindex + 1,tpp ]*(1 - xcoord) +               Vs[ (xindex + 1) % nx + 1,yindex + 1, tpp ]*xcoord
    r4v::T =  Vs[xindex+1, (yindex + 1) % ny + 1,tpp ]*(1 - xcoord) +   Vs[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tpp ]*xcoord
    du[2] =  (
        (1-tcoord)*((1-ycoord)*r1v + ycoord*r2v)
         + tcoord*((1-ycoord)*r3v + ycoord*r4v))
    return
end


function fast_trilinear_ssh_gradient_flipped(du::AbstractVector{Float64},u::AbstractVector{Float64},p,tin)
    sshs = p[7]
    nx::Int64 = size(sshs)[1]
    ny::Int64 = size(sshs)[2]
    nt::Int64 = size(sshs)[3]
    #Get the spatial bounds from p
    ll1::Float64,ur1::Float64  = p[3]
    ll2::Float64,ur2::Float64 = p[4]
    t0::Float64,tf::Float64 = p[5]
    t::Float64 = tin
    if t > tf
        t = tf
    end
    if t < t0
        t = t0
    end
    #Just divrem, but casts the first result to Int
    function gooddivrem(x,y)
        a,b = divrem(x,y)
        return Int(a), b
    end

    xindex::Int64, xcoord::Float64 = gooddivrem((mod((u[1] - ll1), 360)*nx)/360.0,1)
    yindex::Int64, ycoord::Float64 = gooddivrem((mod((u[2] - ll2), 180)*ny)/180.0,1)
    #
    tindex::Int64, tcoord::Float64 = gooddivrem((nt-1)*(t-t0)/(tf-t0),1)
    tindex += 1
    #Make sure we don't go out of bounds
    tpp = tindex + 1
    if tpp > nt
        tpp = nt
    end
    #Actual interpolation for u
    r1::Float64 =  sshs[xindex+1,yindex + 1,tindex ]*(1 - xcoord) +            sshs[ (xindex + 1) % nx + 1,yindex + 1, tindex ]*xcoord
    dr1dx::Float64 =  -sshs[xindex+1,yindex + 1,tindex ] +            sshs[ (xindex + 1) % nx + 1,yindex + 1, tindex ]
    r2::Float64 =  sshs[xindex+1,(yindex + 1) % ny + 1,tindex ]*(1 - xcoord) + sshs[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tindex ]*xcoord
    dr2dx::Float64 =  -sshs[xindex+1,(yindex + 1) % ny + 1,tindex ] + sshs[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tindex ]
    r3::Float64 =  sshs[xindex+1,yindex + 1,tpp ]*(1 - xcoord) +               sshs[ (xindex + 1) % nx + 1,yindex + 1, tpp ]*xcoord
    dr3dx::Float64 =  -sshs[xindex+1,yindex + 1,tpp ] +               sshs[ (xindex + 1) % nx + 1,yindex + 1, tpp ]
    r4::Float64 =  sshs[xindex+1,(yindex + 1) % ny + 1,tpp ]*(1 - xcoord) +    sshs[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tpp ]*xcoord
    dr4dx::Float64 =  -sshs[xindex+1,(yindex + 1) % ny + 1,tpp ] +    sshs[ (xindex + 1) % nx + 1,(yindex + 1)%ny + 1, tpp ]

    du[2] = ((1-tcoord)*((1-ycoord)*dr1dx + ycoord*dr2dx)
             + tcoord*((1-ycoord)*dr3dx + ycoord*dr4dx))*nx/360.0

    du[1] = ((1-tcoord)*(-r1+ r2)
             + tcoord*(-r3+ r4))*ny/180.0
     return
end
