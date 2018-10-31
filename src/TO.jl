#(c) 2017 Nathanael Schilling
#Implementation of TO methods from Froyland & Junge's FEM paper
#And L2-Galerkin approximation of TO

const one2D = e1 + e2

"""
    nonAdaptiveTOCollocation(ctx_domain,ctx_codomain, inverse_flow_map; [bdata_domain,bdata_codomain,project_in,outside_value,volume_preserving=true])

Compute a represenation matrix for $ I_h T$ where $T$ is the Transfer-Operator for the inverse of `inverse_flow_map`,
and $I_h$ is (nodal) interpolation onto `ctx_codomain`.

For (periodic) boundary conditions on the domain or codomain, set `bdata_domain` and `bdata_codomain` appropriately.

The parameters `project_in` and `outside_value` determine what to do if inverse images of points
fall outside the domain. See also `evaluate_function_from_dofvals`.

The parameter `volume_preserving` determines whether to attempt to correct for non-volume-preserving maps.
"""
function nonAdaptiveTOCollocation(
        ctx_domain::gridContext{dim},ctx_codomain::gridContext{dim},
        inverse_flow_map::Function;
        bdata_domain=boundaryData(),
        bdata_codomain=boundaryData(),
        project_in=false,
        outside_value=NaN,
        volume_preserving=true
    ) where dim

    n_codomain = ctx_codomain.n
    n_domain = ctx_domain.n
    result = spzeros(n_codomain,n_domain)

    speye_n_domain = sparse(1.0*I,n_domain,n_domain)

    pointsInvImages = Vec{dim}[]

    ### Calculate inverse images of points
    for j in 1:n_codomain
        current_point = ctx_codomain.grid.nodes[j].x
        pointPullBack::Tensors.Vec{dim,Float64} = Tensors.Vec{dim}(inverse_flow_map(current_point))
        push!(pointsInvImages,pointPullBack)
    end

    #Calculate the integral of shape function in the domain (dof order)
    shape_function_weights_domain_original = undoBCS(ctx_domain,
            vec(sum(assembleMassMatrix(ctx_domain,bdata=bdata_domain),dims=1)),
            bdata_domain)
    #Calculate the integral of shape functions in the codomain (dof order)
    #Do not include boundary conditions here, as we end up summing over this later
    shape_function_weights_codomain = vec(sum(assembleMassMatrix(ctx_codomain),dims=1))

    #This variable will hold approximate pullback (in the measure sense)
    #of shape_function_weights_codomain to domain via finv
    pullback_shape_function_weights = zeros(n_domain)

    #Enumerate over all nodes
    for i in 1:n_codomain
        #Get the corresponding dof
        idof = (ctx_codomain.node_to_dof)[i]
        #Calculate the values of all shape functions at the corresponding pulled back point
        shape_function_values = evaluate_function_from_node_or_cellvals_multiple(ctx_domain,speye_n_domain,[pointsInvImages[i]];
                    outside_value=outside_value,project_in=project_in,is_diag=true
                )
            for valindex in shape_function_values.colptr[1]:(shape_function_values.colptr[2]-1)
                    rowdof = ctx_domain.node_to_dof[shape_function_values.rowval[valindex]]
                    val = shape_function_values.nzval[valindex]
                    result[idof,rowdof] = val
                    pullback_shape_function_weights[rowdof] += shape_function_weights_codomain[idof]*val
            end
    end
    if !volume_preserving
        for jdof in 1:n_domain
                if pullback_shape_function_weights[jdof] != 0.0
                        for ptr in result.colptr[jdof]:(result.colptr[jdof+1]-1)
                                idof = result.rowval[ptr]
                                pullbackW = pullback_shape_function_weights[jdof]
                                originalW = shape_function_weights_domain_original[jdof]
                                finalW = shape_function_weights_codomain[idof]
                                result.nzval[ptr] *= originalW/pullbackW
                        end
                end
        end
    end

    return applyBCS(ctx_domain,result,bdata_domain,add_vals=false)
end

"""
    adaptiveToCollocationStiffnessMatrix(ctx,flow_map; [quadrature_order, on_torus, LL,UR, bdata, volume_preserving=true] )

Calculate the matrix-representation of the bilinear form $a(u,v) = a_1(I_hTu,I_hTv)$ where
$I_h$ is pointwise interpolation of the grid obtained by doing Delaunay triangulation on images of grid points from ctx
and $T$ is the Transfer-operator for `flow_map` and $a_1$ is the weak form of the Laplacian on the codomain.

If `on_torus` is true, the Delaunay Triangulation is done on the torus. Here we require `bdata` for boundary information
on the original domain as well as `LL` and `UR` as lower-left and upper-right corners of the image.

If `volume_preserving == false`, add a volume_correction term to $a_1$ (See paper by Froyland & Junge).
"""
function adaptiveTOCollocationStiffnessMatrix(
        ctx::gridContext{2},
        flow_map::Function;
        quadrature_order=default_quadrature_order,
        on_torus::Bool=false,
        LL::AbstractVector{Float64}=[0.0,0.0],
        UR::AbstractVector{Float64}=[1.0,1.0],
        bdata::boundaryData=boundaryData,
        volume_preserving=true
        )
    if !on_torus  && length(bdata.periodic_dofs_from) != 0
        @warn "This function probably doesn't work for this case"
    end
    #Push forward the points, perform the triangulation
    new_ctx,new_bdata = adaptiveTOFutureGrid(ctx,flow_map;
                            on_torus=on_torus,LL=LL,UR=UR,bdata=bdata
                            )

    #Do volume corrections
    #All values are in node order, which is the same for ctx and new_ctx
    if !volume_preserving
            vols_new = sum(assembleMassMatrix(new_ctx,bdata=new_bdata),dims=1)[node_to_pdof(new_ctx,new_bdata)]
            vols_old = sum(assembleMassMatrix(ctx,bdata=bdata),dims=1)[node_to_pdof(ctx,bdata)]
    else
            vols_new = ones(new_ctx.n - length(new_bdata.periodic_dofs_from) )
            vols_old = ones(ctx.n - length(bdata.periodic_dofs_from) )
    end

    #Assemble stiffness matrix
    if !on_torus
        new_density_nodevals = vols_new ./ vols_old

        new_ctx.mass_weights = [
                evaluate_function_from_node_or_cellvals(ctx_new,new_density_nodevals,q)
                for q in new_ctx.quadrature_points
                ]
        #Now we just need to reorder K2 to have the ordering of the dofs of the original ctx
        n = ctx.n
        I, J, V = findnz(assembleStiffnessMatrix(new_ctx))

        I .= new_ctx.dof_to_node[I]
        J .= new_ctx.dof_to_node[J]
       return applyBCS(ctx,sparse(I,J,V,n,n),bdata)
    else
        new_density_nodevals = zeros(new_ctx.n)
        new_density_nodevals[1:length(vols_new)] = (vols_new ./ vols_old)

        new_ctx.mass_weights = [evaluate_function_from_node_or_cellvals(new_ctx,new_density_nodevals,q)
                   for q in new_ctx.quadrature_points ]
        K = assembleStiffnessMatrix(new_ctx,bdata=new_bdata)
        return  periodizedNewToOldDofOrder(ctx,bdata,new_ctx,new_bdata,K)
    end
end


#Reordering in the periodic case is slightly more tricky
function periodizedNewToOldpdofOrder(
        old_ctx::gridContext{dim},bdata::boundaryData,
        new_ctx::gridContext{dim},new_bdata::boundaryData,K
        ) where dim

        I, J ,V = findnz(K)

        #Here I,J are in pdof order for new_ctx

        bcdof_to_node_new = pdof_to_node(new_ctx,new_bdata)
        node_to_bcdof_old = node_to_pdof(new_ctx,new_bdata)
        I .= node_to_bcdof_old[bcdof_to_node_new[I]]
        J .= node_to_bcdof_old[bcdof_to_node_new[J]]

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

        new_ctx, new_bdata = irregularDelaunayGrid(new_nodes_in_node_order;on_torus=true,LL=LL,UR=UR)
        return new_ctx,new_bdata
    end
end

"""
    adaptiveTOCollocation(ctx,flow_map; [on_torus=false, bdata,LL,UR, volume_preserving=true])

Calculate the represenation matrix for $J_h I_h T$ where $T$ is the Transfer-Operator
for `flow_map` and $J_h$ is the nodal interpolation operator for `ctx` and $I_h$ is the
nodal interpolation operator onto a grid which has nodes at future time given by images (under `flow_map`)
of nodal basis points of $f$.

If `on_torus==true`, then everything is done with periodic boundary conditions.
If `volume_preserving==false`, a correction is made for non-volume-preserving maps.
"""
function adaptiveTOCollocation(ctx::gridContext{dim}, flow_map::Function;
     on_torus::Bool=false,  bdata=nothing, LL=[0.0,0.0],UR=[1.0,1.0],
     volume_preserving=true
     ) where dim

    if on_torus
            if bdata == nothing
                    throw(AssertionError("bdata == nothing"))
            end
        npoints = ctx.n - length(bdata.periodic_dofs_from)
        ctx_new, bdata_new = CoherentStructures.adaptiveTOFutureGrid(ctx,flow_map,
             on_torus=on_torus, LL=LL, UR=UR,bdata=bdata)
        ALPHA = nonAdaptiveTOCollocation(ctx_new,ctx,x->x,volume_preserving=volume_preserving)
        ALPHA_bc = applyBCS(ctx,ALPHA,bdata,ctx_col=ctx_new,bdata_col=bdata_new,add_vals=false)
        if volume_preserving
                L = sparse(I,npoints,npoints)[node_to_pdof(ctx,bdata)[pdof_to_node(ctx_new,bdata_new)],:]
        else
                vols_start = vec(sum(assembleMassMatrix(ctx,bdata=bdata),dims=1))[node_to_pdof(ctx,bdata)]
                vols_end = vec(sum(assembleMassMatrix(ctx_new,bdata=bdata_new),dims=1))[node_to_pdof(ctx_new,bdata_new)]
                inv_volume_distortion = vols_end ./ vols_start

                L = spdiagm(0 => inv_volume_distortion)[node_to_pdof(ctx,bdata)[pdof_to_node(ctx_new,bdata_new)],:]
        end
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
                            push!(DL2I, ctx.cell_to_dof[cellnumber])
                            push!(DL2J,ctx.cell_to_dof[TQinvCellNumber])
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
