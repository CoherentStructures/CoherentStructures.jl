#(c) 2017 Nathanael Schilling
#Implementation of TO methods from Froyland & Junge's FEM paper
#And L2-Galerkin approximation of TO

using JuAFEM
zero2D = zero(Vec{2})
one2D = e1 + e2

#Currently only works on a rectangular grid that must be specified in advance
function nonAdaptiveTO(ctx::gridContext{2},inverse_flow_map::Function)
    LL = ctx.spatialBounds[1]
    UR = ctx.spatialBounds[2]
    n = ctx.n
    result = spzeros(n,n)
    for j in 1:n
        current_point = ctx.grid.nodes[j].x
        jdof = (ctx.node_to_dof)[j]
        try
            #TODO: Is using the Vec{2} type here slower than using Arrays?
            pointPullback = Vec{2}(min.(UR - 1e-10one2D, max.(LL + 1e-10*one2D, inverse_flow_map(current_point))))
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
    #TODO:Remove code commented out below
    #xs = [i[1] for i in new_nodes_in_dof_order]
    #ys = [i[2] for i in new_nodes_in_dof_order]
    #GR.plot(xs,ys,".")
    new_ctx = gridContext{2}(Triangle, new_nodes_in_dof_order, quadrature_order=quadrature_order)
    #Now we just need to reorder K2 to have the ordering of the dofs of the original ctx
    I,J,V = findnz(assembleStiffnessMatrix(new_ctx))
    l = length(I)
    result = spzeros(n,n)
    for i in 1:l
        #The node-ordering of the grid in new_ctx is the dof-ordering of the grid in ctx
        result[new_ctx.dof_to_node[I[i]],new_ctx.dof_to_node[J[i]]] = V[i]
    end
    return result
end


#L2-Galerkin approximation of Transfer Operator
#TODO: Can this be written without any dependence on the dimension?
#TODO: Implement this for multiple timesteps
#TODO: Implement this for missing data
function L2GalerkinTOFromInverse(ctx::gridContext{2},flow_map::Function)
    DL2 = spzeros(ctx.n,ctx.n)
    cv::CellScalarValues{2} = CellScalarValues(ctx.qr, ctx.ip)
    nshapefuncs = getnbasefunctions(cv)         # number of basis functions
    dofs::Vector{Int} = zeros(nshapefuncs)
    index::Int = 1 #Counter to know the number of the current quadrature point

    #TODO: allow for using a different ctx here, e.g. like in the adaptiveTO settings
    @inbounds for (cellnumber, cell) in enumerate(CellIterator(ctx.dh))
        JuAFEM.reinit!(cv,cell)
        celldofs!(dofs,ctx.dh,cellnumber)
        #Iterate over all quadrature points in the cell
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            const dΩ::Float64 = getdetJdV(cv,q)
            TQ::Vec{2,Float64} = flow_map(ctx.quadrature_points[index])
            try
                local_coords::Vec{2,Float64}, nodes::Vector{Int} = locatePoint(ctx,TQ)
                for (shape_fun_num,j) in enumerate(nodes)
                    for i in 1:nshapefuncs
                        φ::Float64 = shape_value(cv,q,i)
                        ψ::Float64 = JuAFEM.value(ctx.ip,shape_fun_num,local_coords)
                        DL2[dofs[i],ctx.node_to_dof[j]] += dΩ*φ*ψ
                    end
                end
            catch y
                if !isa(y,DomainError)
                    throw(y)
                end
                print("Flow map gave result $TQ outside of domain!")
            end
            index += 1
        end
    end
    return DL2
end

function L2GalerkinTO(ctx::gridContext{2},flow_map::Function)
    DL2 = spzeros(ctx.n,ctx.n)
    cv::CellScalarValues{2} = CellScalarValues(ctx.qr, ctx.ip)
    nshapefuncs = getnbasefunctions(cv)         # number of basis functions
    dofs::Vector{Int} = zeros(nshapefuncs)
    index::Int = 1 #Counter to know the number of the current quadrature point

    #TODO: allow for using a different ctx here, e.g. like in the adaptiveTO settings
    @inbounds for (cellnumber, cell) in enumerate(CellIterator(ctx.dh))
        JuAFEM.reinit!(cv,cell)
        celldofs!(dofs,ctx.dh,cellnumber)
        #Iterate over all quadrature points in the cell
        for q in 1:getnquadpoints(cv) # loop over quadrature points
            const dΩ::Float64 = getdetJdV(cv,q)
            TQ::Vec{2,Float64} = flow_map(ctx.quadrature_points[index])
            try
                local_coords::Vec{2,Float64}, nodes::Vector{Int} = locatePoint(ctx,TQ)
                for (shape_fun_num,i) in enumerate(nodes)
                    for j in 1:nshapefuncs
                        φ::Float64 = shape_value(cv,q,j)
                        ψ::Float64 = JuAFEM.value(ctx.ip,shape_fun_num,local_coords)
                        DL2[ctx.node_to_dof[i],dofs[j]] += dΩ*φ*ψ
                    end
                end
            catch y
                if !isa(y,DomainError)
                    throw(y)
                end
                print("Flow map gave result $TQ outside of domain!")
            end
            index += 1
        end
    end
    return DL2
end
