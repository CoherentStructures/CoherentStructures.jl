#(c) 2017 Nathanael Schilling
#Implementation of (non-adaptive) TO method from Froyland & Junge's paper

using JuAFEM
include("DelaunayGrid.jl")
include("util.jl")

zero2D = zero(Vec{2})
one2D = e1 + e2
function getAlphaMatrix(grid::JuAFEM.Grid,loc::cellLocator,inverse_flow_map::Function,ip::JuAFEM.Interpolation)
    n = length(grid.nodes) #TODO: Maybe do this better
    result = spzeros(n,n) #TODO: Use sparse matrix here possibly
    for j in 1:n
        current_point = grid.nodes[j].x
        try
            #TODO: Make the following domain-invariant more or less...
            pointPullback = Vec{2}(min.((1-1e-6)*one2D, max.(1e-6*one2D, inverse_flow_map(current_point))))
            local_coords, nodelist = locatePoint(loc,grid,pointPullback)
            for  (i,nodeid) in enumerate(nodelist)
                result[j,nodeid] += JuAFEM.value(ip,i,local_coords)
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
