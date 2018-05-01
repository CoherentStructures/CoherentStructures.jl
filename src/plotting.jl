#TODO: Can this be made more efficient?
function plot_u(ctx::gridContext,dof_vals::Vector{Float64},nx=50,ny=50;bdata=nothing,kwargs...)
    id = x -> x
    plot_u_eulerian(ctx,dof_vals,ctx.spatialBounds[1],ctx.spatialBounds[2],id,nx,ny,bdata=bdata;kwargs...)
end


function plot_ftle(
		   odefun, p,tspan,
		   LL, UR, nx=50,ny=50;δ=1e-9,
		   tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),
		   inplace=true,existing_plot=nothing,flip_y=false, check_inbounds=always_true,
		   kwargs...)
    x1 = collect(linspace(LL[1] + 1.e-8, UR[1] - 1.e-8,nx))
    x2 = collect(linspace(LL[2] + 1.e-8, UR[2] - 1.e-8,ny))
    if flip_y
        x2 = reverse(x2)
    end
    #Initialize FTLE-field with NaNs
    FTLE = SharedArray{Float64,2}(ny,nx)
    for I in CartesianRange(size(FTLE))
	    FTLE[I] = NaN
    end
    nancounter, nonancounter = @sync @parallel ((x,y)->x.+y) for i in eachindex(x1)
        nancounter_local = 0
        nonancounter_local = 0
        for j in eachindex(x2)
            if check_inbounds(x1[i],x2[j],p)
                try
                    FTLE[j,i] = 1.0/(2*(tspan[end]-tspan[1]))*
		                          log(eigvals(eigfact(dott(linearized_flow(odefun,[x1[i],x2[j]],tspan,δ,tolerance=tolerance,p=p,solver=solver)[end])))[1])
                    nonancounter_local += 1
                catch e
                    nancounter_local += 1
                end
            end
        end
        (nancounter_local,nonancounter_local)
        end

    print("plot_ftle ignored $nancounter NaN values ($nonancounter were good)")
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
                    euler_to_lagrange_points=nothing,
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
    # x1 = Float64[]
    # x2 = Float64[]
    # values = Float64[]
    const u_values =  dof2node(ctx,dof_values)
    x1 = linspace(LL[1],UR[1],nx)
    x2 = linspace(LL[2],UR[2],ny)
    if euler_to_lagrange_points == nothing
        # euler_to_lagrange_points_raw = SharedArray{Float64}(ny,nx,2)
        # @sync @parallel for i in eachindex(x1)
        #     for j in eachindex(x2)
        #         point = StaticArrays.SVector{2}(x1[i],x2[j])
        #         try
        #             back = inverse_flow_map(point)
        #             euler_to_lagrange_points_raw[j,i,1] = back[1]
        #             euler_to_lagrange_points_raw[j,i,2] = back[2]
        #         catch e
        #             if isa(e,InexactError)
        #                 euler_to_lagrange_points_raw[j,i,1] = NaN
        #                 euler_to_lagrange_points_raw[j,i,2] = NaN
        #             else
        #                 throw(e)
        #             end
        #         end
        #     end
        # end
        euler_to_lagrange_points_raw = compute_euler_to_lagrange_points_raw(inverse_flow_map,x1,x2)
        euler_to_lagrange_points = [zero(Tensors.Vec{2}) for y in x2, x in x1]
        for i in 1:nx, j in 1:ny
            euler_to_lagrange_points[j,i] = Tensors.Vec{2}([euler_to_lagrange_points_raw[j,i,1],euler_to_lagrange_points_raw[j,i,2]])
	    end
	end

    if only_get_lagrange_points
        return euler_to_lagrange_points
    end
    z = zeros(size(euler_to_lagrange_points))
    for I in eachindex(euler_to_lagrange_points)
        if isnan((euler_to_lagrange_points[I])[1])
            z[I] = NaN
        else
            z[I] = evaluate_function_from_nodevals(ctx,u_values,euler_to_lagrange_points[I],NaN)
        end
    end

    if postprocessor != nothing
       z =  postprocessor(z)
    end
    result =  Plots.heatmap(x1,x2,z,fill=true,aspect_ratio=1,xlim=(LL[1],UR[1]),ylim=(LL[2],UR[2]);kwargs...)#,colormap=GR.COLORMAP_JET)

    if return_scalar_field
        return result, z
    end
    return result
end

function compute_euler_to_lagrange_points_raw(inv_flow_map,x1,x2)
    euler_to_lagrange_points_raw = SharedArray{Float64}(length(x2),length(x1),2)
    @sync @parallel for i in eachindex(x1)
        for j in eachindex(x2)
            point = StaticArrays.SVector{2}(x1[i],x2[j])
            try
                back = inv_flow_map(point)
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
    return euler_to_lagrange_points_raw
end

function plot_spectrum(λ)
    Plots.scatter(real.(λ),imag.(λ))
end

function plot_real_spectrum(λ)
    Plots.scatter(1:length(λ),real.(λ))
end


function eulerian_videos(ctx, us::Function,inverse_flow_map_t,t0,tf, nx, ny,nt, LL, UR,num_videos=1;extra_kwargs_fun=nothing,kwargs...)
    allvideos = [Plots.Animation() for i in 1:num_videos]

    for (index,t) in enumerate(linspace(t0,tf,nt))
    	print("Processing frame $index")
        if t != t0
        	current_inv_flow_map = x -> inverse_flow_map_t(t,x)
        else
            current_inv_flow_map = x->x
        end
    	euler_to_lagrange_points = plot_u_eulerian(ctx,zeros(ctx.n), LL,UR,current_inv_flow_map,nx,ny; only_get_lagrange_points=true,kwargs...)
        function plotsingleframe(i)
    	    current_u = us(i,t)
    	    if extra_kwargs_fun != nothing
        		curframe = plot_u_eulerian(ctx, current_u, LL, UR, current_inv_flow_map, nx,ny;euler_to_lagrange_points=euler_to_lagrange_points,extra_kwargs_fun(i,t)...,kwargs...);
    	    else
        		curframe = plot_u_eulerian(ctx, current_u, LL, UR, current_inv_flow_map, nx,ny;euler_to_lagrange_points=euler_to_lagrange_points,kwargs...);
    	    end
            return curframe
        end
        tmpres = pmap(plotsingleframe, 1:num_videos)
        for i in 1:num_videos
    	    Plots.frame(allvideos[i],tmpres[i])
    	end
    end;
    return allvideos
end

function eulerian_video(ctx, u::Function, inverse_flow_map_t,t0,tf, nx, ny, nt, LL, UR;extra_kwargs_fun=nothing,kwargs...)
    usfun = (index,t) -> u(t)
    if (extra_kwargs_fun!= nothing)
        extra_kwargs_fun_out = (i,t) -> extra_kwargs_fun(t)
    else
        extra_kwargs_fun_out = nothing
    end
    return eulerian_videos(ctx,usfun,inverse_flow_map_t, t0,tf, nx,ny,nt, LL,UR,1;extra_kwargs_fun=extra_kwargs_fun_out,kwargs...)[1]
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
    ut = dof2node(ctx,corrected_u(t0))
    val = [evaluate_function_from_nodevals(ctx,ut,[p[1],p[2]]) for p in allpoints]
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
        ut = dof2node(ctx,corrected_u(times[t+1]))
        val = [evaluate_function_from_nodevals(ctx,ut,p) for p in allpoints_initial]
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
    return Plots.@animate for p in res
        p
    end
end
