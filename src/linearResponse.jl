# functions to be used in the context of the methods described in https://arxiv.org/abs/1907.10852


"""
    linear_response_tensor(parametrized_flow_map, u, p)

Calculate the tensor dot_A from the linear response paper, when given a flow map T(x,p)
"""
function linear_response_tensor(parametrized_flow_map, u, p)
	return linear_response_tensor(
					x->parametrized_flow_map(x,p),
					x->flow_map_parameter_derivative(parametrized_flow_map, x, p),
					u,
					p)
end

"""
    linear_response_tensor(flow_map, parameter_derivative, u, p)

Calculate the tensor dot_A from the linear response paper, when given a flow map T(x) and its
parameter derivative dotT(x)
"""
function linear_response_tensor(flow_map, parameter_derivative, u, p)
	DT = linearized_flow_autodiff(flow_map, u)
	DTdot = linearized_flow_autodiff(parameter_derivative, u)
	DTinv = inv(DT)
	return -Tensors.symmetric(DTinv ⋅ DTdot ⋅ Tensors.dott(DTinv))
end

"""
    linearized_flow_autodiff(flow_map, x)

Calculate derivative of flow map by automatic differentiation

Return the time-resolved base trajectory and its associated linearized flow maps.
"""
linearized_flow_autodiff(flow_map, x) = Tensor{2,2}(ForwardDiff.jacobian(flow_map, x))


flow_map_parameter_derivative(flow_map, u, p) = ForwardDiff.jacobian(x -> flow_map(x[1:end - 1], x[end]), vcat(u, p))[:,end]

"""
    adaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, parametrized_flow_map, p;
                                        quadrature_order=default_quadrature_order,
                                        on_torus::Bool=false,
                                        on_cylinder::Bool=false,
                                        LL_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[1],
                                        UR_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[2],
                                        bdata::BoundaryData=BoundaryData(),
                                        flow_map_mode=0
										)

Return the linear response matrix L computed by the adaptive TO method, using a parametrized flow map T(x,p)

Also note that, as with all functions of the TO method, BoundaryData makes sense only for periodic
boundary data. If we have Dirichlet boundary conditions, it is best to simply ignore them here
and apply them to L afterwards.
"""
function adaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, parametrized_flow_map, p;
                                        quadrature_order=default_quadrature_order,
                                        on_torus::Bool=false,
                                        on_cylinder::Bool=false,
                                        LL_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[1],
                                        UR_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[2],
                                        bdata::BoundaryData=BoundaryData(),
                                        flow_map_mode=0
										)
	return adaptiveTOCollocationLinearResponseMatrix(
		ctx,
		x->parametrized_flow_map(x,p),
		x->flow_map_parameter_derivative(parametrized_flow_map, x, p),
		p,
		quadrature_order=quadrature_order,on_torus=on_torus,on_cylinder=on_cylinder,LL_future=LL_future,
		UR_future=UR_future,bdata=bdata,flow_map_mode=flow_map_mode)
end

"""
    adaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, flow_map, parameter_derivative, p;
                                        quadrature_order=default_quadrature_order,
                                        on_torus::Bool=false,
                                        on_cylinder::Bool=false,
                                        LL_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[1],
                                        UR_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[2],
                                        bdata::BoundaryData=BoundaryData(),
                                        flow_map_mode=0
                                        )

Return the linear response matrix L computed by the adaptive TO method, using a flow map T(x) and its
parameter derivative dotT(x)

Also note that, as with all functions of the TO method, BoundaryData makes sense only for periodic
boundary data. If we have Dirichlet boundary conditions, it is best to simply ignore them here
and apply them to L afterwards.
"""
function adaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, flow_map, parameter_derivative, p;
                                        quadrature_order=default_quadrature_order,
                                        on_torus::Bool=false,
                                        on_cylinder::Bool=false,
                                        LL_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[1],
                                        UR_future::Tuple{<:Real,<:Real}=ctx.spatialBounds[2],
                                        bdata::BoundaryData=BoundaryData(),
                                        flow_map_mode=0
                                        )

	# adapted from adaptiveTOCollocationStiffnessMatrix
	# functionality is slightly reduced from there, as we assume only the volume preserving case
	# and a single time in the future

    flow_map_images = zeros(Vec{2}, ctx.n)
	for i in 1:ctx.n
		if flow_map_mode == 0
			flow_map_images[i] = Vec{2}(flow_map(ctx.grid.nodes[i].x))
		else
			flow_map_images[i] = Vec{2}(flow_map(i))
		end
	end

    flow_map_t(j) = flow_map_images[j]

	# we only support volume_preserving here for now
	new_ctx,new_bdata,_ = adaptiveTOFutureGrid(ctx, flow_map_t;
                                                    on_torus=on_torus,
                                                    on_cylinder=on_cylinder,
                                                    LL_future=LL_future,
                                                    UR_future=UR_future,
                                                    bdata=bdata,
													quadrature_order=quadrature_order,
                                                    flow_map_mode=1)

	# find the derivatives on the nodes of the old grid, then reorder this so it is in dof order for the new grid
	derivative_on_old_grid = zeros(2,ctx.n)
	for i in 1:ctx.n
		derivative_on_old_grid[:,i] = parameter_derivative(ctx.grid.nodes[i].x)
	end
	# keep in mind: node order for new_ctx is bcdof order for ctx, which allows us to link the two grids
	# unfortunately we need some intermediate steps. new_ctx potentially has more nodes than ctx (we need
	# to duplicate some if we are e.g. on a torus), so new_ctx.dof_to_node might fail
	translation_table_bcdof_new_to_node_old = bcdof_to_node(ctx,bdata)[bcdof_to_node(new_ctx,new_bdata)]
	derivative_in_bcdof_order_new_grid = derivative_on_old_grid[:,translation_table_bcdof_new_to_node_old]
	# we need the derivatives in dof order instead of in bcdof order
	W = mapslices(x->undoBCS(new_ctx, x, new_bdata),derivative_in_bcdof_order_new_grid,dims=2)

    L = assembleLinearResponseMatrix(new_ctx, W, bdata=new_bdata)

	# change L from bcdof order for ctx_new to node order for ctx_new. This is by design the same
	# as bcdof order for ctx, which is the order we need it in (this can be slightly confusing)
    translation_table_bcdof_new_to_bcdof_old = bcdof_to_node(new_ctx,new_bdata)
    I, J, V = Main.CoherentStructures.findnz(L)
    I .= translation_table_bcdof_new_to_bcdof_old[I]
    J .= translation_table_bcdof_new_to_bcdof_old[J]
	n = ctx.n - length(bdata.periodic_dofs_from)
    L = sparse(I,J,V,n,n)

    return 0.5(L+L')
end



"""
function  nonadaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, parametrized_flow_map, A, p;
    													bdata::BoundaryData=BoundaryData())

Return the linear response matrix L computed by the nonadaptive TO method, using a parametrized flow
map T(x,p) as well as a representation matrix A computed by nonAdaptiveTOCollocation

Also note that, as with all functions of the TO method, BoundaryData makes sense only for periodic
boundary data. If we have Dirichlet boundary conditions, it is best to simply ignore them here
and apply them to L afterwards.
"""
function  nonadaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, parametrized_flow_map, A, p;
    													bdata::BoundaryData=BoundaryData())
	return nonadaptiveTOCollocationLinearResponseMatrix(
			ctx,
			x->parametrized_flow_map(x,p),
			x->flow_map_parameter_derivative(parametrized_flow_map, x, p),
			A,
			p;
			bdata=bdata)
end


"""
function  nonadaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, parametrized_flow_map, A, p;
    													bdata::BoundaryData=BoundaryData())

Return the linear response matrix L computed by the nonadaptive TO method, using a flow map T(x) and
its parameter derivative dotT(x) as well as a representation matrix A computed by nonAdaptiveTOCollocation

Also note that, as with all functions of the TO method, BoundaryData makes sense only for periodic
boundary data. If we have Dirichlet boundary conditions, it is best to simply ignore them here
and apply them to L afterwards.
"""
function nonadaptiveTOCollocationLinearResponseMatrix(ctx::GridContext{2}, flow_map, parameter_derivative, A, p;
    													bdata::BoundaryData=BoundaryData())
	W = zeros(2,ctx.n)
	for i in 1:ctx.n
		W[:,i] = parameter_derivative(ctx.grid.nodes[i].x)
	end
	# from node order to bcdof order
	W = W[:,bcdof_to_node(ctx, bdata)];
	WA = W*A'
	# and now back to dof order
	WA = mapslices(x->undoBCS(ctx, x, bdata),WA,dims=2)

	L = assembleLinearResponseMatrix(ctx, WA, bdata=bdata)

	return 0.5A'*(L+L')*A
end


# adapted from _assembleStiffnessMatrix
# WA is a matrix containing all the derivatives at the dofs, possibly multiplied by a representation
# matrix A in the nonadaptive case
function assembleLinearResponseMatrix(ctx, WA; bdata::BoundaryData=BoundaryData())
    cv = FEM.CellScalarValues(ctx.qr, ctx.ip, ctx.ip_geom)
    dh = ctx.dh
    K = FEM.create_sparsity_pattern(dh)
    a_K = FEM.start_assemble(K)
    dofs = zeros(Int, FEM.ndofs_per_cell(dh))
    n = FEM.getnbasefunctions(cv)         # number of basis functions
    Ke = zeros(n, n)

    index = 1 # quadrature point counter

    @inbounds for cell in FEM.CellIterator(dh)
        fill!(Ke, 0)
        FEM.reinit!(cv, cell)
        for q in 1:FEM.getnquadpoints(cv)
            dΩ = FEM.getdetJdV(cv, q) * ctx.mass_weights[index]
            for i in 1:n
                ∇φᵢ = FEM.shape_gradient(cv, q, i)
                for j in 1:n
                    ∇φⱼ = FEM.shape_gradient(cv, q, j)
                    for k in 1:n
                        ∇φₖ = FEM.shape_gradient(cv, q, k)
                        Ke[i,j] += (∇φᵢ ⋅ WA[:,FEM.celldofs(cell)[k]])  * (∇φₖ ⋅ ∇φⱼ) * dΩ
                    end
                end
            end
            index += 1
        end
        FEM.celldofs!(dofs, cell)
        FEM.assemble!(a_K, dofs, Ke)
    end
    return applyBCS(ctx, K, bdata)
end

"""
    get_linear_response(u₀,λ₀,M,K,L)

Solve the linear system defined in the linear response paper, given an eigenpair (u₀,λ₀), mass matrix
M, stiffness matrix K and linear response matrix L
"""
function get_linear_response(u₀,λ₀,M,K,L)
	lhs = [K-λ₀*M   -M*u₀
			u₀'*M      0]
	rhs = [-L*u₀    ;  0]

	sol = lhs \ rhs

	u_dot = sol[1:end-1]
	λ_dot = sol[end]

	return u_dot, λ_dot
end
