#(c) 2017 Nathanael Schilling
#Implementation of (non-adaptive) TO method from Froyland & Junge's paper

using JuAFEM
include("GridFunctions.jl")
include("util.jl")
zero2D = zero(Vec{2})
one2D = e1 + e2

function getAlphaMatrix(ctx::gridContext{2},inverse_flow_map::Function)
    n = ctx.n
    result = spzeros(n,n) #TODO: Use sparse matrix here possibly
    for j in 1:n
        current_point = ctx.grid.nodes[j].x
        jdof = (ctx.dhtable)[j]
        try
            #TODO: Make the following domain-invariant more or less...
            pointPullback = Vec{2}(min.((1-1e-6)*one2D, max.(1e-6*one2D, inverse_flow_map(current_point))))
            local_coords, nodelist = locatePoint(ctx,pointPullback)
            for  (i,nodeid) in enumerate(nodelist)
                result[jdof,ctx.dhtable[nodeid]] = JuAFEM.value(ctx.ip,i,local_coords)
            end
            catch y
                if !isa(y, DomainError)
                    throw(y)
                end
                print("Inverse flow map gave result outside of domain!")
                #pointPullback = Vec{2}(min.((1+1e-6)*one2D, max.(1e-6*one2D, inverse_flow_map(current_point))))
                pointPullback = inverse_flow_map(current_point)
                print(pointPullback)
                #TODO: What do we do if the point is outside of the triangulation?
            end
    end
    return result
end
