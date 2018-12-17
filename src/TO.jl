#(c) 2017 Nathanael Schilling
#Implementation of TO methods from Froyland & Junge's FEM paper
#And L2-Galerkin approximation of TO

const one2D = e1 + e2

"""
    nonAdaptiveTOCollocation(ctx_domain, inverse_flow_map, [ctx_codomain; bdata_domain, bdata_codomain, project_in, outside_value, volume_preserving=true])

Compute a represenation matrix for ``I_h T`` where ``T`` is the transfer operator
for the inverse of `inverse_flow_map`, and ``I_h`` is (nodal) interpolation onto
`ctx_codomain`.

For (periodic) boundary conditions on the domain or codomain, set `bdata_domain`
and `bdata_codomain` appropriately.

The parameters `project_in` and `outside_value` determine what to do if inverse
images of points fall outside the domain. See also `evaluate_function_from_dofvals`.

The parameter `volume_preserving` determines whether to attempt to correct for
non-volume-preserving maps.
"""
function nonAdaptiveTOCollocation(
        ctx_domain::gridContext{dim},
        inverse_flow_map::Function,
        ctx_codomain::gridContext{dim}=ctx_domain;
        bdata_domain=boundaryData(),
        bdata_codomain=bdata_domain,
        project_in=false,
        outside_value=NaN,
        volume_preserving=true
    ) where dim

    @assert isa(ctx_codomain.ip, JuAFEM.Lagrange)

    n_codomain = ctx_codomain.n
    n_domain = ctx_domain.n
    result = spzeros(n_codomain,n_domain)

    speye_n_domain = sparse(1.0*I,n_domain,n_domain)

    pointsInvImages = Vec{dim}[]

    ### Calculate inverse images of points
    for j in 1:n_codomain
        current_point = ctx_codomain.grid.nodes[j].x
        pointPullBack::Vec{dim,Float64} = Vec{dim}(inverse_flow_map(current_point))
        push!(pointsInvImages,pointPullBack)
    end

    #Calculate the integral of shape function in the domain (dof order)
    shape_function_weights_domain_original = undoBCS(ctx_domain,
            vec(sum(assembleMassMatrix(ctx_domain, bdata=bdata_domain), dims=1)),
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

    result = applyBCS(ctx_codomain, result, nothing;
                    ctx_col=ctx_domain, bdata_col=bdata_domain,
                    add_vals=true)

    return applyBCS(ctx_codomain, result, bdata_codomain;
                    ctx_col=ctx_domain, bdata_col=nothing,
                    add_vals=false), vec(applyBCS(
                        ctx_domain,
                        (pullback_shape_function_weights./shape_function_weights_domain_original)[:,:],bdata_domain; bdata_col=nothing,add_vals=true)
                        )
end

"""
    adaptiveTOCollocationStiffnessMatrix(ctx,flow_maps,times=nothing; [quadrature_order, on_torus, LL,UR, bdata, volume_preserving=true,flow_map_mode=0] )

Calculate the matrix-representation of the bilinear form ``a(u,v) = 1/N \\sum_n^N a_1(I_hT_nu,I_hT_nv)`` where
``I_h`` is pointwise interpolation of the grid obtained by doing Delaunay triangulation on images of grid points from ctx
and ``T_n`` is the Transfer-operator for ``x \\mapsto flow_maps(x,times)[n]`` and ``a_1`` is the weak form of the Laplacian on the codomain. Moreover,
``N`` in the equation above is equal to `length(times)` and ``t_n`` ranges over the elements of `times`.

If `times==nothing`, take ``N=1`` above and use the map ``x \\mapsto flow_maps(x)` instead of the version with `t_n`.

If `on_torus` is true, the Delaunay Triangulation is done on the torus. Here we require `bdata` for boundary information
on the original domain as well as `LL` and `UR` as lower-left and upper-right corners of the image.

If `volume_preserving == false`, add a volume_correction term to ``a_1`` (See paper by Froyland & Junge).

If `flow_map_mode==0`, apply flow map to nodal basis function coordinates. Else apply it to index number.
"""
function adaptiveTOCollocationStiffnessMatrix(
        ctx::gridContext{2},
        flow_maps::Function,
        times=nothing
        ;
        quadrature_order=default_quadrature_order,
        on_torus::Bool=false,
        LL_future::AbstractVector{Float64}=ctx.spatialBounds[1],
        UR_future::AbstractVector{Float64}=ctx.spatialBounds[2],
        bdata::boundaryData=boundaryData(),
        volume_preserving=true,
        flow_map_mode=0
        )
    if !on_torus  && length(bdata.periodic_dofs_from) != 0
        @warn "This function probably doesn't work for this case"
    end


    if times==nothing
        N = 1
    else
        N = length(times)
    end
    flow_map_images = zeros(Vec{2},(N,ctx.n))
    if times == nothing
        for i in 1:ctx.n
            if flow_map_mode == 0
                flow_map_images[1,i] = Vec{2}(flow_maps(ctx.grid.nodes[i].x))
            else
                flow_map_images[1,i] = Vec{2}(flow_maps(i))
            end
        end
    else
        for i in 1:ctx.n
            if flow_map_mode == 0
                flow_map_images[:,i] = Vec{2}.(flow_maps(ctx.grid.nodes[i].x,times))
            else
                flow_map_images[:,i] = Vec{2}.(flow_maps(i,times))
            end
        end
    end

    #Push forward the points, perform the triangulation
    As = SparseMatrixCSC{Float64,Int64}[]
    for n in 1:N
        flow_map_t = j -> flow_map_images[n,j]
        new_ctx,new_bdata,new_density_bcdofvals = adaptiveTOFutureGrid(ctx,flow_map_t;
                                on_torus=on_torus,LL_future=LL_future,UR_future=UR_future,bdata=bdata,
                                flow_map_mode=1
                                )


        translation_table = bcdof_to_node(new_ctx,new_bdata)
        #Assemble stiffness matrix
        if !volume_preserving
            #This is allowed because new_ctx is a (possibly periodic) delaunay grid,
            #with the node ordering of the first n nodes being the bcdof ordering of
            #the old grid
            new_density_nodevals = new_density_bcdofvals
            while length(new_density_nodevals) != new_ctx.n
                push!(new_density_nodevals,0.0)
            end

            new_ctx.mass_weights = [evaluate_function_from_node_or_cellvals(new_ctx,new_density_nodevals,q)
                       for q in new_ctx.quadrature_points ]
        end
        I, J, V = findnz(assembleStiffnessMatrix(new_ctx,bdata=new_bdata))
        I .= translation_table[I]
        J .= translation_table[J]
        n = ctx.n - length(bdata.periodic_dofs_from)
        push!(As,sparse(I,J,V,n,n))
    end
    return mean(As)
end


#Reordering in the periodic case is slightly more tricky
function bcdofNewToBcdofOld(
        old_ctx::gridContext{dim},bdata::boundaryData,
        new_ctx::gridContext{dim},new_bdata::boundaryData,K
        ) where dim

        I, J ,V = findnz(K)

        #Here I,J are in pdof order for new_ctx

        bcdof_to_node_new = bcdof_to_node(new_ctx,new_bdata)
        node_to_bcdof_old = node_to_bcdof(old_ctx,bdata)
        I .= node_to_bcdof_old[bcdof_to_node_new[I]]
        J .= node_to_bcdof_old[bcdof_to_node_new[J]]

        old_pdof_n = old_ctx.n - length(bdata.periodic_dofs_from)

        return sparse(I,J,V,old_pdof_n,old_pdof_n)
end

function node_to_bcdof(ctx::gridContext{dim},bdata::boundaryData) where dim
        n_nodes = ctx.n - length(bdata.periodic_dofs_from)
        bdata_table = BCTable(ctx,bdata)
        return bdata_table[ctx.node_to_dof]
end

function bcdof_to_node(ctx::gridContext{dim},bdata::boundaryData) where dim
        n_nodes = ctx.n - length(bdata.periodic_dofs_from)
        bdata_table = BCTable(ctx,bdata)
        result = zeros(Int,n_nodes)
        for i in 1:ctx.n
            if result[bdata_table[ctx.node_to_dof[i]]] == 0
                result[bdata_table[ctx.node_to_dof[i]]] = i
            end
        end
        return result
end


"""
    adaptiveTOFutureGrid(ctx, flow_map;[on_torus=false,bdata=nothing,LL,UR,quadrature_order,flow_map_mode=0])

Takes the (non-periodic) grid points from `ctx` (in bcdof order), applies `flow_map` to them,
and (periodic) delaunay triangulates them and returns a P1-Delaunay `gridContext,bdata, volchange` tuple for that,
where `volchange` is a vector of relative volumes of the nodal basis functions.
If `flow_map_mode==0` (default), apply `flow_map` to node coordinates. If `flow_map_mode==1`,
apply `flow_map` to node index
"""
function adaptiveTOFutureGrid(ctx::gridContext{dim},flow_map;
        on_torus=false,bdata=boundaryData(),LL_future=ctx.spatialBounds[1], UR_future=ctx.spatialBounds[2],
        quadrature_order=default_quadrature_order,
        flow_map_mode=0
        ) where dim

    if isEmptyBC(bdata) && on_torus==true
        throw(AssertionError("Require bdata parameter if on_torus==true"))
    end

    #Push forward "original" nodes

    if flow_map_mode == 0
        new_nodes_in_bcdof_order = [
            Vec{2}(flow_map(ctx.grid.nodes[j].x))
            for j in bcdof_to_node(ctx,bdata)
                ]
    else
        new_nodes_in_bcdof_order = [
            Vec{2}(flow_map(j))
            for j in bcdof_to_node(ctx,bdata)
                ]
    end
    new_ctx, new_bdata = irregularDelaunayGrid(
                            new_nodes_in_bcdof_order;
                            on_torus=on_torus,LL=LL_future,UR=UR_future,
                            quadrature_order=quadrature_order
                            )

    #Do volume corrections
    #All values are in node order for new_ctx, which is the same as bcdof order for ctx2
    n_nodes = length(new_nodes_in_bcdof_order)
    vols_new = sum(assembleMassMatrix(new_ctx,bdata=new_bdata),dims=1)[1,node_to_bcdof(new_ctx,new_bdata)[1:n_nodes]]
    vols_old = sum(assembleMassMatrix(ctx,bdata=bdata),dims=1)[1,1:n_nodes]

    return new_ctx,new_bdata, vols_new ./ vols_old

end


"""
    adaptiveTOCollocation(ctx,flow_map; [on_torus=false, bdata,LL,UR, volume_preserving=true])

Calculate the represenation matrix for ``J_h I_h T`` where ``T`` is the Transfer-Operator
for `flow_map` and ``J_h`` is the nodal interpolation operator for `ctx` and ``I_h`` is the
nodal interpolation operator onto a grid which has nodes at future time given by images (under `flow_map`)
of nodal basis points of ``f``.

If `on_torus==true`, then everything is done with periodic boundary conditions.
If `volume_preserving==false`, a correction is made for non-volume-preserving maps.
"""
function adaptiveTOCollocation(ctx::gridContext{dim}, flow_map::Function;
     on_torus::Bool=false,  bdata=nothing, LL=[0.0,0.0],UR=[1.0,1.0],
     volume_preserving=true,projection_method=:naTO
     ) where dim

    if on_torus
        if bdata === nothing
            throw(AssertionError("bdata == nothing"))
        end
        npoints = ctx.n - length(bdata.periodic_dofs_from)
        ctx_new, bdata_new, volume_change = adaptiveTOFutureGrid(ctx, flow_map;
             on_torus=on_torus, LL=LL, UR=UR,bdata=bdata)
        if projection_method == :naTO
            ALPHA_bc, _ = nonAdaptiveTOCollocation(ctx_new,x->x,ctx;
                volume_preserving=true,bdata_domain=bdata_new,bdata_codomain=bdata)
        elseif projection_method == :L2Ginv
            ALPHA_bc  = L2GalerkinTOFromInverse(ctx_new, x->x, ctx;
                bdata_domain=bdata_new, bdata_codomain=bdata)
        elseif projection_method == :L2G
            ALPHA_bc, _ = L2GalerkinTO(ctx_new, x->x, ctx;
                volume_preserving=volume_preserving, bdata_domain=bdata_new, bdata_codomain=bdata)
        else
            throw(AssertionError("Invalid projection_method"))
        end
        if volume_preserving
            L = sparse(I, npoints, npoints)[node_to_bcdof(ctx,bdata)[bcdof_to_node(ctx_new,bdata_new)],:]
            result = ALPHA_bc*L
        else
            volume_change_pdof = volume_change[bcdof_to_node(ctx,bdata)]
            K = assembleStiffnessMatrix(ctx,bdata=bdata)
            M = assembleMassMatrix(ctx,bdata=bdata)
            volume_change_pdof = (M - 1e-2K)\(M*volume_change_pdof)
            volume_change = volume_change_pdof[node_to_bcdof(ctx,bdata)]

            L = sparse(1.0*I,npoints,npoints)[node_to_bcdof(ctx,bdata)[bcdof_to_node(ctx_new,bdata_new)],:]
            for j in 1:(size(L)[2])
                L[:,j] *= volume_change_pdof[j]
            end
            result = ALPHA_bc*L
        end
        return result, volume_change
    else # not on torus
        throw(AssertionError("Not yet implemented"))
    end
end

"""
    L2GalerkinTOFromInverse(ctx_domain,inverse_flow_map,ctx_codomain=ctx_domain;
        [ϵ=0,periodic_directions=(false,false), n_stencil_points,bdata_domain,bdata_codomain]
        )

Approximate the transfer operator for volume preserving maps (``f \\mapsto f ∘ F^{-1}``)
using an L²-Galerkin method. For this, we need to calculate the following integrals:
```math
T_{i,j} = \\int φ_i φ_j \\circ F^{-1} dx
```
We approximate the integral by doing quadrature. If ``ϵ>0`` (works only if `dim==2`), we approximate some additional diffusion
by replacing each image of a quadrature point by ``F^{-1}`` by `n_stencil_points` points distributed on a
ball of radius ``ϵ`` around it. If these additional points should be placed in a periodic way,
then set the corresponding entry in `periodic_directions` to true.

It should be noted that (perhaps counter-intuitively) `inverse_flow_map` should go from the
grid described by `ctx_codomain` to that described by `ctx_domain`
"""
function L2GalerkinTOFromInverse(
                ctx_domain::gridContext{dim},
                inverse_flow_map::Function,
                ctx_codomain::gridContext{dim}=ctx_domain;
                ϵ::Float64=0.0, n_stencil_points::Int=10,
                periodic_directions=(false,false),
                bdata_domain=boundaryData(),
                bdata_codomain=bdata_domain,
                ) where dim

    #See http://blog.marmakoide.org/?p=1
    stencil::Vector{Vec{2,Float64}} = Vec{dim,Float64}[]
    stencil_density::Float64 = 0.0
    φ = π*(3 - √5)
    if ϵ ≠ 0.0
        @assert dim == 2
        stencil_density = 1. /n_stencil_points
        print(stencil_density)
        for i in 0:(n_stencil_points-1)
            θ::Float64 = i * φ
            r::Float64 = ϵ*(√i / √(n_stencil_points-1))
            push!(stencil,Vec{2}((r*cos(θ), r*sin(θ))))
        end
    else
        stencil_density = 1.0
        push!(stencil, zero(Vec{dim}))
    end

    LL_domain::Vec{dim,Float64} = Vec{dim}(ctx_domain.spatialBounds[1])
    UR_domain::Vec{dim,Float64} = Vec{dim}(ctx_domain.spatialBounds[2])

    cv::JuAFEM.CellScalarValues{dim} = JuAFEM.CellScalarValues(ctx_codomain.qr, ctx_codomain.ip,ctx_codomain.ip_geom)
    nshapefuncs::Int = JuAFEM.getnbasefunctions(cv)         # number of basis functions
    dofs::Vector{Int} = zeros(nshapefuncs)
    index::Int = 1 #Counter to know the number of the current quadrature point
    DL2I = Vector{Int}()
    DL2J = Vector{Int}()
    DL2V = Vector{Float64}()

    @inbounds for (cellnumber, cell) in enumerate(JuAFEM.CellIterator(ctx_codomain.dh))
        JuAFEM.reinit!(cv,cell)
        JuAFEM.celldofs!(dofs,ctx_codomain.dh,cellnumber)
        #Iterate over all quadrature points in the cell
        for q in 1:JuAFEM.getnquadpoints(cv) # loop over quadrature points
            dΩ::Float64 = JuAFEM.getdetJdV(cv,q)
            TQ::Vec{dim,Float64} = Vec{dim}(inverse_flow_map(ctx_codomain.quadrature_points[index]))
            for s in stencil
                current_point::Vec{dim,Float64} = TQ + s
                current_point = Vec{dim,Float64}( i ->
                    periodic_directions[i] ? LL_domain[i] + (mod(current_point[i] - LL_domain[i],UR_domain[i] - LL_domain[i])) : current_point[i]
                    )
                try
                    local_coords::Vec{dim,Float64}, nodes::Vector{Int},TQinvCellNumber = locatePoint(ctx_domain,current_point)
                    if isa(ctx_domain.ip, JuAFEM.Lagrange)
                        for (shape_fun_num,j) in enumerate(nodes)
                                for i in 1:nshapefuncs
                                    φ::Float64 = JuAFEM.shape_value(cv,q,i)
                                    ψ::Float64 = JuAFEM.value(ctx_domain.ip,shape_fun_num,local_coords)
                                    push!(DL2I, dofs[i])
                                    push!(DL2J,ctx_domain.node_to_dof[j])
                                    push!(DL2V, dΩ*φ*ψ*stencil_density)
                                end
                         end
                    else
                            push!(DL2I, ctx_codomain.cell_to_dof[cellnumber])
                            push!(DL2J,ctx_domain.cell_to_dof[TQinvCellNumber])
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

    DL2::SparseMatrixCSC{Float64,Int64} = sparse(DL2I,DL2J, DL2V,ctx_codomain.n,ctx_domain.n)
    return applyBCS(ctx_codomain,DL2,bdata_codomain;
                    ctx_col=ctx_domain,bdata_col=bdata_domain,
                    add_vals=true)
end

"""
    L2GalerkinTO(ctx,flow_map; [bdata])

Approximate the Transfer-Operator (for volume preserving maps) by calculating
```math
T_{i,j} = \\int φ_i ∘ F φ_j dx
```
"""
function L2GalerkinTO(
    ctx_domain::gridContext{dim},
    flow_map::Function,
    ctx_codomain::gridContext{dim}=ctx_domain;
    bdata_domain=boundaryData(),
    bdata_codomain=bdata_domain,
    verbosity=1
    ) where dim
    DL2 = spzeros(ctx_codomain.n,ctx_domain.n)
    cv::JuAFEM.CellScalarValues{dim} = JuAFEM.CellScalarValues(ctx_domain.qr,ctx_domain.ip, ctx_domain.ip_geom)
    nbasefuncs = JuAFEM.getnbasefunctions(cv)         # number of basis functions
    dofs::Vector{Int} = zeros(nbasefuncs)
    index::Int = 1 #Counter to know the number of the current quadrature point
    badcounter = 0

    #TODO: Make this more efficient
    @inbounds for (cellnumber, cell) in enumerate(JuAFEM.CellIterator(ctx_domain.dh))
        JuAFEM.reinit!(cv,cell)
        JuAFEM.celldofs!(dofs,ctx_domain.dh,cellnumber)
        #Iterate over all quadrature points in the cell
        for q in 1:JuAFEM.getnquadpoints(cv) # loop over quadrature points
            dΩ::Float64 = JuAFEM.getdetJdV(cv,q)
            TQ::Vec{2,Float64} = Vec{2}(flow_map(ctx_domain.quadrature_points[index]))
            local_coordsTQ::Vec{2,Float64}, nodesTQ::Vector{Int},cellIndexTQ = try
                    locatePoint(ctx_codomain,TQ)
            catch y
                if !isa(y,DomainError)
                    rethrow(y)
                end
                (verbosity >= 2) && @warn("Flow map gave result $TQ outside of domain!")
                badcounter += 1
                continue
            end
            if isa(ctx_codomain.ip,JuAFEM.Lagrange)
                    for (shape_fun_num,i) in enumerate(nodesTQ)
                        ψ::Float64 = JuAFEM.value(ctx_codomain.ip,shape_fun_num,local_coordsTQ)
                        indexi = ctx_domain.node_to_dof[i]
                        for j in 1:nbasefuncs
                            indexj::Int64 = 0
                            if isa(ctx_domain.ip,JuAFEM.Lagrange)
                                indexj=dofs[j]
                                φ::Float64 = JuAFEM.shape_value(cv,q,j)
                            else
                                indexj=ctx_domain.cell_to_dof[cellnumber]
                                φ = 1.0
                            end
                            DL2[indexi,indexj] += dΩ*φ*ψ
                        end
                    end
            else
                    indexi = ctx_codomain.cell_to_dof[cellIndexTQ]
                    ψ = 1.0
                    for j in 1:nbasefuncs
                        indexj::Int64 = 0
                        if isa(ctx_domain.ip,JuAFEM.Lagrange)
                            indexj=dofs[j]
                            φ::Float64 = JuAFEM.shape_value(cv,q,j)
                        else
                            indexj=ctx_domain.cell_to_dof[cellnumber]
                            φ = 1.0
                        end
                        DL2[indexi, indexj] += dΩ*φ*ψ
                    end
            end
        index += 1
        end
    end
    (verbosity >= 1 && badcounter > 0) && @warn "$badcounter results outside of domain"
    return applyBCS(ctx_codomain,DL2,bdata_codomain; ctx_col=ctx_domain,bdata_col=bdata_domain)
end
