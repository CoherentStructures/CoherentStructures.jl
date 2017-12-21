#(c) 2017 Nathanael Schilling
#Implementation of TO methods from Froyland & Junge's FEM paper

using JuAFEM
include("GridFunctions.jl")
include("util.jl")
zero2D = zero(Vec{2})
one2D = e1 + e2

#Currently only works on a rectangular grid that must be specified in advance
function getAlphaMatrix(ctx::gridContext{2},inverse_flow_map::Function,LL=zero2D, UR=one2D )
    n = ctx.n
    result = spzeros(n,n)
    for j in 1:n
        current_point = ctx.grid.nodes[j].x
        jdof = (ctx.node_to_dof)[j]
        try
            #TODO: Is using the Vec{2} type here slower than using Arrays?
            pointPullback = Vec{2}(min.((1-1e-6)*UR, max.(1e-6*LL, inverse_flow_map(current_point))))
            #TODO: Don't doo this pointwise, but pass whole vector to locatePoint
            local_coords, nodelist = locatePoint(ctx,pointPullback)
            for  (i,nodeid) in enumerate(nodelist)
                result[jdof,ctx.node_to_dof[nodeid]] = JuAFEM.value(ctx.ip,i,local_coords)
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

#TODO Rename functions name in this file to something more consistent
#Note that it seems like this function is broken somehow TODO: Fix this.
function adaptiveTO(ctx::gridContext{2},flow_map::Function,quadrature_order=default_quadrature_order)
    n = ctx.n
    new_nodes_in_dof_order = [ flow_map(ctx.grid.nodes[ctx.dof_to_node[j]].x) for j in 1:n ]
    new_ctx = gridContext{2}(Triangle, new_nodes_in_dof_order, quadrature_order)
    #Now we just need to reorder K2 to have the ordering of the dofs of the original ctx
    I,J,V = findnz(assembleStiffnessMatrix(new_ctx))
    l = length(I)
    result = spzeros(n,n)
    for i in 1:l
        result[new_ctx.dof_to_node[I[i]],new_ctx.dof_to_node[J[i]]] = V[i]
    end
    return result
end
