#TODO: Can this be made more efficient?
function plot_u(ctx::gridContext,dof_vals::Vector{Float64},nx=50,ny=50;plotit=true,bdata=nothing,kwargs...)
    id = x -> x
    plot_u_eulerian(ctx,dof_vals,ctx.spatialBounds[1],ctx.spatialBounds[2],id,nx,ny,plotit=plotit,bdata=bdata;kwargs...)
end


function plot_ftle(
		   odefun, p,tspan,
		   LL, UR, nx=50,ny=50;δ=1e-9,
		   tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),
		   inplace=true,existing_plot=nothing,flip_y=false, check_inbounds=always_true,
		   kwargs...)
    x1 =  collect(linspace(LL[1] + 1e-8, UR[1] -1.e-8,nx))
    x2 =  collect(linspace(LL[2] + 1.e-8,UR[2]-1.e-8,ny))
    if flip_y
	x2 = reverse(x2)
    end
    #Initialize FTLE-field with NaNs
    FTLE = SharedArray{Float64,2}(ny,nx)
    for j in 1:ny
	for i in 1:nx
	    FTLE[j,i] = NaN
	end
    end
    nancounter, nonancounter = @sync @parallel ((x,y)->x.+y) for i in 1:nx
        nancounter_local = 0
        nonancounter_local = 0
	for j in 1:ny
	    if check_inbounds(x1[i],x2[j],p)
		try
		    FTLE[j,i] =
		    1./(2*(tspan[end]-tspan[1]))*
		    log(maximum(abs.(eigvals(eigfact(dott(linearized_flow(odefun,Vec{2}([x1[i],x2[j]]),tspan,δ,tolerance=tolerance,p=p,solver=solver )[end]))))))
		    nonancounter_local += 1
		catch e
		    nancounter_local += 1
		end
	    end
	end
	(nancounter_local,nonancounter_local)
    end

    print("plot_ftle Ignored $nancounter NaN values ($nonancounter were good)")
    if flip_y == true
	x2 = reverse(x2)
	x2 *= -1.0
    end
    if inplace
	return Plots.heatmap!(existing_plot,x1,x2,FTLE; kwargs...)
    else
	return Plots.heatmap(x1,x2,FTLE; kwargs...)
    end
end

function plot_u_eulerian(
                    ctx::gridContext,
                    dof_vals::Vector{Float64},
                    LL::AbstractVector{Float64},
                    UR::AbstractVector{Float64},
                    inverse_flow_map::Function,
                    nx=50,
                    ny=60;
                    plotit=true,euler_to_lagrange_points=nothing,
                    only_get_lagrange_points=false,postprocessor=nothing,
                    return_scalar_field=false,
                    bdata=nothing,
                    kwargs...)
    if (bdata==nothing) && (ctx.n != length(dof_vals))
        dbcs = getHomDBCS(ctx)
        if length(dbcs.dbc_dofs) + length(dof_vals) != ctx.n
            error("Input u has wrong length")
        end
        dof_values = undoBCS(ctx,dof_vals,dbcs)
    elseif (bdata != nothing)
        dof_values = undoBCS(ctx,dof_vals,bdata)
    else
        dof_values = dof_vals
    end
    #
    x1 = Float64[]
    x2 = Float64[]
    values = Float64[]
    const u_values =  dof2U(ctx,dof_values)
    x1 =  linspace(LL[1] , UR[1] ,nx)
    x2 =  linspace(LL[2] ,UR[2] ,ny)
    if euler_to_lagrange_points == nothing
	euler_to_lagrange_points_raw = SharedArray{Float64}(ny,nx,2)
        @sync @parallel for i in 1:nx
            for j in 1:ny
                point = Vec{2}((x1[i],x2[j]))
                try
		    back = inverse_flow_map(point)
		    euler_to_lagrange_points_raw[j,i,1] = back[1]
		    euler_to_lagrange_points_raw[j,i,2] = back[2]
                catch e
                    if isa(e,InexactError)
			euler_to_lagrange_points_raw[j,i,1] = NaN
			euler_to_lagrange_points_raw[j,i,2] = NaN
                    else
                        throw(e)
                    end
                end
            end
        end
	euler_to_lagrange_points = [zero(Vec{2}) for y in x2, x in x1]
	for i in 1:nx
	    for j in 1:ny
		euler_to_lagrange_points[j,i] = Vec{2}(euler_to_lagrange_points_raw[j,i,1:2])
	    end
	end
    end
    if only_get_lagrange_points
        return euler_to_lagrange_points
    end
    z = zeros(ny,nx)
    for i in 1:nx
        for j in 1:ny
            if isnan((euler_to_lagrange_points[j,i])[1])
                z[j,i] = NaN
            else
                z[j,i] = evaluate_function_from_nodevals(ctx,euler_to_lagrange_points[j,i],u_values,NaN)
            end
        end
    end
    if postprocessor != nothing
       z =  postprocessor(z)
    end
    result =  Plots.heatmap(x1,x2,z,fill=true,aspect_ratio=1,xlim=(LL[1],UR[1]),ylim=(LL[2],UR[2]);kwargs...)#,colormap=GR.COLORMAP_JET)
    if plotit
        Plots.plot(result)
    end
    if return_scalar_field
        return result, z
    end
    return result
end

function plot_spectrum(λ)
    Plots.scatter(real.(λ),imag.(λ))
end

function plot_real_spectrum(λ)
    Plots.scatter(1:length(λ),real.(λ))
end

function eulerian_video(ctx, u::Function, LL, UR,nx,ny,t0,tf,nt,inverse_flow_map_t;kwargs...)
    return @animate for (index,t) in enumerate(linspace(t0,tf,nt))
        print("Processing frame $index")
        current_inv_flow_map = (x) -> inverse_flow_map_t(t,x)
        current_u = u(nt)
        plot_u_eulerian(ctx, current_u, LL, UR, current_inv_flow_map, nx,ny;kwargs...)
    end
end

function eulerian_video_fast(ctx, u::Function,
    nx, ny, t0,tf,nt, forward_flow_map, LL_big,UR_big;bdata=nothing,display_inplace=true,kwargs...)
    LL = ctx.spatialBounds[1]
    UR = ctx.spatialBounds[2]
    function corrected_u(t)
        dof_vals = u(t)
        if (bdata==nothing) && (ctx.n != length(dof_vals))
            dbcs = getHomDBCS(ctx)
            if length(dbcs.dbc_dofs) + length(dof_vals) != ctx.n
                error("Input u has wrong length")
            end
            dof_values = undoBCS(ctx,dof_vals,dbcs)
        elseif (bdata != nothing)
            dof_values = undoBCS(ctx,dof_vals,bdata)
        else
            dof_values = dof_vals
        end
        return dof_values
    end
    x1 = linspace(LL[1],UR[1],nx)
    x2 = linspace(LL[2],UR[2],ny)
    allpoints = [
        Vec{2}((x,y)) for y in x2, x in x1
    ]
    allpoints_initial = copy(allpoints)
    #
    times = linspace(t0,tf,nt)
    x1p = [p[1] for p in allpoints]
    x2p = [p[2] for p in allpoints]
    ut = dof2U(ctx,corrected_u(t0))
    val = [evaluate_function_from_nodevals(ctx,[p[1],p[2]],ut) for p in allpoints]
    #
    res = [scatter(x1p[1:end],x2p[1:end],zcolor=val[1:end],
        xlim=(LL_big[1],UR_big[1]),ylim=(LL_big[2],UR_big[2]),legend=false,
        marker=:square,markersize=300./nx,markerstrokewidth=0;kwargs...)]
    if display_inplace
        Plots.display(res[end])
    end
    #
    for t in 1:(nt-1)
        allpoints = [forward_flow_map(times[t], times[t+1],p) for p in allpoints]
        x1p = [p[1] for p in allpoints]
        x2p = [p[2] for p in allpoints]
        ut = dof2U(ctx,corrected_u(times[t+1]))
        val = [evaluate_function_from_nodevals(ctx,p,ut) for p in allpoints_initial]
        push!(res,
            Plots.scatter(x1p[1:end],x2p[1:end],zcolor=val[1:end],
                xlim=(LL_big[1],UR_big[1]),ylim=(LL_big[2],UR_big[2]),legend=false,
                marker=:square,markersize=70./nx,markerstrokewidth=0;kwargs...)
                )
		#
	if display_inplace
	    Plots.display(res[end])
	end
    end
    return @animate for p in res
        p
    end
end
