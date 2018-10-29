#(c) 2017 Nathanael Schilling
#Implementation of TO methods from Froyland & Junge's FEM paper
#And L2-Galerkin approximation of TO

# zero2D = zero(Vec{2}) # seems unused currently
const one2D = e1 + e2

function nonAdaptiveTO(
        ctx_domain::gridContext{dim},ctx_codomain::gridContext{dim},
        inverse_flow_map::Function;
        project_in=false,
        outside_value=NaN
    ) where dim
    n = ctx_codomain.n
    m = ctx_domain.n
    result = spzeros(n,m)
    speyem = sparse(I,m,m)*1.0

    for j in 1:n
        current_point = ctx_codomain.grid.nodes[j].x
        jdof = (ctx_codomain.node_to_dof)[j]
        shape_function_values = try
            pointPullBack::Tensors.Vec{dim,Float64} = Tensors.Vec{dim}(inverse_flow_map(current_point))
            evaluate_function_from_node_or_cellvals_multiple(ctx_domain,speyem,[pointPullBack],
                outside_value=outside_value,project_in=project_in,is_diag=true)
            catch y
                if !isa(y, DomainError)
                    rethrow(y)
                end
                print("Inverse flow map gave result outside of domain!")
                #pointPullback = Vec{2}(min.((1+1e-6)*one2D, max.(1e-6*one2D, inverse_flow_map(current_point))))
                pointPullback = inverse_flow_map(current_point)
                print(pointPullback)
                #TODO: What do we do if the point is outside of the triangulation?
            end
            for valindex in shape_function_values.colptr[1]:(shape_function_values.colptr[2]-1)
                    row = shape_function_values.rowval[valindex]
                    result[jdof,ctx_domain.node_to_dof[row]] = shape_function_values.nzval[valindex]
            end
    end
    return result
end

#TODO Rename functions name in this file to something more consistent
function adaptiveTO(
        ctx::gridContext{2},
        flow_map::Function,
        quadrature_order=default_quadrature_order;
        on_torus::Bool=false,
        LL::AbstractVector{Float64}=[0.0,0.0],
        UR::AbstractVector{Float64}=[1.0,1.0],
        bdata::boundaryData=boundaryData,
        volume_preserving=true
        )
    new_ctx,new_bdata = adaptiveTOFutureGrid(ctx,flow_map; on_torus=on_torus,LL=LL,UR=UR,bdata=bdata)
    if !volume_preserving
            vols_new = sum(assembleMassMatrix(new_ctx,bdata=new_bdata),dim=1)
            vols_old = sum(assembleMassMatrix(ctx,bdata),dim=1)
    else
            vols_new = ones(new_ctx.n - length(new_bdata.periodic_dofs_from) )
            vols_old = ones(ctx.n - length(bdata.periodic_dofs_from) )
    end

    if !on_torus
        new_density_nodevals = vols_new ./ vols_old

        new_ctx.mass_weights = [evaluate_function_from_node_or_cellvals(ctx_new,new_density_nodevals,q)
                   for q in new_ctx.quadrature_points ]
        #Now we just need to reorder K2 to have the ordering of the dofs of the original ctx
        n = ctx.n
        I, J, V = findnz(assembleStiffnessMatrix(new_ctx))

        I .= new_ctx.dof_to_node[I]
        J .= new_ctx.dof_to_node[J]
       return sparse(I,J,V,n,n)
    else
        new_density_pdofvals = vols_new ./ vols_old
        new_density_dofvals = undoBCS(new_ctx,new_density_pdofvals, new_bdata)
        new_density_nodevals = new_density_dofvals[new_ctx.node_to_dof]

        new_ctx.mass_weights = [evaluate_function_from_node_or_cellvals(new_ctx,new_density_nodevals,q)
                   for q in new_ctx.quadrature_points ]
        K = assembleStiffnessMatrix(new_ctx,bdata=new_bdata)
        return  periodizedNewToOldDofOrder(ctx,new_ctx,new_bdata,K)
    end
end


#Reordering in the periodic case is slightly more tricky
function periodizedNewToOldDofOrder(
        old_ctx::gridContext{dim},new_ctx::gridContext{dim},new_bdata::boundaryData,K
        ) where dim

        I, J ,V = findnz(K)

        bcdof_to_node = pdof_to_node(new_ctx,new_bdata)
        I .= old_ctx.node_to_dof[bcdof_to_node[I]]
        J .= old_ctx.node_to_dof[bcdof_to_node[J]]

        return sparse(I,J,V,old_ctx.n,old_ctx.n)
end

function node_to_pdof(ctx::gridContext{dim},bdata::boundaryData) where dim
        n_nodes = ctx.n - length(bdata.periodic_dofs_from)
        bdata_table = BCTable(ctx,bdata)
        return bdata_table[ctx.node_to_dof[1:n_nodes]]
end

function pdof_to_node(ctx::gridContext{dim},bdata::boundaryData) where dim
        return sortperm(node_to_pdof(ctx,bdata))
end


function adaptiveTOFutureGrid(ctx::gridContext{dim},flow_map;
        on_torus=false,bdata=nothing,LL=[0.0,0.0], UR=[1.0,1.0]
        ) where dim


    if !on_torus
        n = ctx.n
        new_nodes_in_dof_order = [ Tensors.Vec{2}(flow_map(ctx.grid.nodes[ctx.dof_to_node[j]].x)) for j in 1:n ]
        new_ctx = gridContext{2}(JuAFEM.Triangle, new_nodes_in_dof_order, quadrature_order=quadrature_order)
        return new_ctx,boundaryData()

    else
        if bdata == nothing
            throw(AssertionError("Require bdata parameter if on_torus==true"))
        end

        #Push forward "original" nodes
        # (There are additional nodes from the periodic dofs, but
        #by construction of the periodic delaunay triangulation, they
        #are at the end)

        n_nodes = ctx.n - length(bdata.periodic_dofs_from)
        new_nodes_in_node_order = [
                    Tensors.Vec{2}(flow_map(ctx.grid.nodes[j].x))
                    for j in 1:n_nodes ]

        new_ctx, new_bdata = periodicDelaunayGrid(new_nodes_in_node_order,LL,UR)
        return new_ctx,new_bdata
    end
end


function adaptiveTransferOperator(ctx::gridContext{dim}, flow_map::Function;
     on_torus::Bool=false,  bdata=nothing, LL=[0.0,0.0],UR=[1.0,1.0]
     ) where dim

    if on_torus
            if bdata == nothing
                    throw(AssertionError("bdata == nothing"))
            end
        npoints = ctx.n - length(bdata.periodic_dofs_from)
        ctx_new, bdata_new = CoherentStructures.adaptiveTOFutureGrid(ctx,flow_map,
             on_torus=true, LL=LL, UR=UR,bdata=bdata)
        ALPHA = nonAdaptiveTO(ctx_new,ctx,x->x)
        ALPHA_bc = applyBCS(ctx,ALPHA,bdata,ctx_col=ctx_new,bdata_col=bdata_new,add_vals=false)
        L = sparse(I,npoints,npoints)[node_to_pdof(ctx,bdata)[pdof_to_node(ctx_new,bdata_new)],:]
        return ALPHA_bc*L
    else
            throw(AssertionError("Not yet implemented"))
    end
end


#L2-Galerkin approximation of Transfer Operator
#TODO: Can this be written without any dependence on the dimension?
#TODO: Implement this for multiple timesteps
#TODO: Implement this for missing data
function L2GalerkinTOFromInverse(ctx::gridContext{2},inverse_flow_map::Function,ϵ::Float64=0.0;periodic_directions::Tuple{Bool,Bool}=(false,false),n_stencil_points::Int=10)

    #See http://blog.marmakoide.org/?p=1
    stencil::Vector{Tensors.Vec{2,Float64}} = Tensors.Vec{2,Float64}[]
    stencil_density::Float64 = 0.0
    φ = π*(3 - √5)
    if ϵ ≠ 0.0
        stencil_density = 1. /n_stencil_points
        print(stencil_density)
        for i in 0:(n_stencil_points-1)
            θ::Float64 = i * φ
            r::Float64 = ϵ*(√i / √(n_stencil_points-1))
            push!(stencil,Tensors.Vec{2}((r*cos(θ), r*sin(θ))))
        end
    else
        stencil_density = 1.0
        push!(stencil, zero(Tensors.Vec{2}))
    end

    LL::Tensors.Vec{2,Float64} = Tensors.Vec{2}(ctx.spatialBounds[1])
    UR::Tensors.Vec{2,Float64} = Tensors.Vec{2}(ctx.spatialBounds[2])

    cv::JuAFEM.CellScalarValues{2} = JuAFEM.CellScalarValues(ctx.qr, ctx.ip,ctx.ip_geom)
    nshapefuncs::Int = JuAFEM.getnbasefunctions(cv)         # number of basis functions
    dofs::Vector{Int} = zeros(nshapefuncs)
    index::Int = 1 #Counter to know the number of the current quadrature point
    DL2I = Vector{Int}()
    DL2J = Vector{Int}()
    DL2V = Vector{Float64}()

    #TODO: allow for using a different ctx here, e.g. like in the adaptiveTO settings
    @inbounds for (cellnumber, cell) in enumerate(JuAFEM.CellIterator(ctx.dh))
        JuAFEM.reinit!(cv,cell)
        JuAFEM.celldofs!(dofs,ctx.dh,cellnumber)
        #Iterate over all quadrature points in the cell
        for q in 1:JuAFEM.getnquadpoints(cv) # loop over quadrature points
            dΩ::Float64 = JuAFEM.getdetJdV(cv,q)
            TQ::Tensors.Vec{2,Float64} = Tensors.Vec{2}(inverse_flow_map(ctx.quadrature_points[index]))
            for s in stencil
                current_point::Tensors.Vec{2,Float64} = TQ + s
                current_point = Tensors.Vec{2,Float64}((
                    periodic_directions[1] ? LL[1] + (mod(current_point[1] - LL[1],UR[1] - LL[1])) : current_point[1],
                    periodic_directions[2] ? LL[2] + (mod(current_point[2] - LL[2],UR[2] - LL[2])) : current_point[2],
                    ))
                try
                    local_coords::Tensors.Vec{2,Float64}, nodes::Vector{Int},TQinvCellNumber = locatePoint(ctx,current_point)
                    if isa(ctx.ip, JuAFEM.Lagrange)
                        for (shape_fun_num,j) in enumerate(nodes)
                                for i in 1:nshapefuncs
                                    φ::Float64 = JuAFEM.shape_value(cv,q,i)
                                    ψ::Float64 = JuAFEM.value(ctx.ip,shape_fun_num,local_coords)
                                    push!(DL2I, dofs[i])
                                    push!(DL2J,ctx.node_to_dof[j])
                                    push!(DL2V, dΩ*φ*ψ*stencil_density)
                                end
                         end
                    else
                            push!(DL2J, ctx.cell_to_dof[cellnumber])
                            push!(DL2I,ctx.cell_to_dof[TQinvCellNumber])
                            push!(DL2V, dΩ*stencil_density)
                    end
                catch y
                    if !isa(y,DomainError)
                        rethrow(y)
                    end
                    print("Got result $current_point outside of domain!")
                end
            end
            index += 1
        end
    end

    DL2::SparseMatrixCSC{Float64,Int64} = sparse(DL2I,DL2J, DL2V,ctx.n,ctx.n)
    return DL2
end

function L2GalerkinTO(ctx::gridContext{2},flow_map::Function)
    DL2 = spzeros(ctx.n,ctx.n)
    cv::JuAFEM.CellScalarValues{2} = JuAFEM.CellScalarValues(ctx.qr,ctx.ip, ctx.ip_geom)
    nbasefuncs = JuAFEM.getnbasefunctions(cv)         # number of basis functions
    dofs::Vector{Int} = zeros(nbasefuncs)
    index::Int = 1 #Counter to know the number of the current quadrature point

    #TODO: allow for using a different ctx here, e.g. like in the adaptiveTO settings
    @inbounds for (cellnumber, cell) in enumerate(JuAFEM.CellIterator(ctx.dh))
        JuAFEM.reinit!(cv,cell)
        JuAFEM.celldofs!(dofs,ctx.dh,cellnumber)
        #Iterate over all quadrature points in the cell
        for q in 1:JuAFEM.getnquadpoints(cv) # loop over quadrature points
            dΩ::Float64 = JuAFEM.getdetJdV(cv,q)
            TQ::Tensors.Vec{2,Float64} = Tensors.Vec{2}(flow_map(ctx.quadrature_points[index]))
            try
                local_coordsTQ::Tensors.Vec{2,Float64}, nodesTQ::Vector{Int},cellIndexTQ = locatePoint(ctx,TQ)
                if isa(ctx.ip,JuAFEM.Lagrange)
                        for (shape_fun_num,i) in enumerate(nodesTQ)
                            ψ::Float64 = JuAFEM.value(ctx.ip,shape_fun_num,local_coordsTQ)
                            for j in 1:nbasefuncs
                                φ::Float64 = JuAFEM.shape_value(cv,q,j)
                                DL2[ctx.node_to_dof[i],dofs[j]] += dΩ*φ*ψ
                            end
                        end
                else
                        indexi = ctx.cell_to_dof[cellIndexTQ]
                        indexj = ctx.cell_to_dof[cellnumber]
                        DL2[indexi,indexj] +=  dΩ
                end
            catch y
                if !isa(y,DomainError)
                    rethrow(y)
                end
                print("Flow map gave result $TQ outside of domain!")
            end
            index += 1
        end
    end
    return DL2
end
