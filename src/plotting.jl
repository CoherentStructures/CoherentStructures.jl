
"""
    plot_u(ctx, dof_vals, nx, ny; bdata=nothing, kwargs...)

Plot the function with coefficients (in dof order, possible boundary conditions in `bdata`) given by `dof_vals` on the grid `ctx`.
The domain to be plotted on is given by `ctx.spatialBounds`.
The function is evaluated on a regular `nx` by `ny` grid, the resulting plot is a heatmap.
Keyword arguments are passed down to `plot_u_eulerian`, which this function calls internally.
"""
function plot_u(ctx::gridContext, dof_vals::Vector{Float64}, nx=50, ny=50; bdata=nothing, kwargs...)
    id = x -> x
    plot_u_eulerian(ctx, dof_vals, id, ctx.spatialBounds[1], ctx.spatialBounds[2], nx, ny, bdata=bdata; kwargs...)
end


function get_full_nodevals(ctx,dof_vals; bdata=nothing)
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
    return dof2node(ctx,dof_values)
end

@userplot Plot_U_Eulerian
@recipe function f(
        as::Plot_U_Eulerian;
        bdata=nothing,
        euler_to_lagrange_points=nothing,
        z=nothing,
        postprocessor=nothing,
        return_scalar_field=false
    )
    ctx::gridContext = as.args[1]
    dof_vals::Vector{Float64} = as.args[2]
    inverse_flow_map::Function = as.args[3]
    LL::AbstractVector{Float64} = as.args[4]
    UR::AbstractVector{Float64} = as.args[5]
    if length(as.args) >= 6
        nx=as.args[6]
    else
        nx = 100
    end
    if length(as.args) >= 7
        ny = as.args[7]
    else
        ny = 100
    end

    #
    # x1 = Float64[]
    # x2 = Float64[]
    # values = Float64[]
    u_values = get_full_nodevals(ctx,dof_vals;bdata=bdata)
    x1 = range(LL[1],stop=UR[1],length=nx)
    x2 = range(LL[2],stop=UR[2],length=ny)
    if euler_to_lagrange_points == nothing
        euler_to_lagrange_points_raw = compute_euler_to_lagrange_points_raw(inverse_flow_map,x1,x2)
        euler_to_lagrange_points = [zero(Tensors.Vec{2}) for y in x2, x in x1]
        for i in 1:nx, j in 1:ny
            euler_to_lagrange_points[j,i] = Tensors.Vec{2}([euler_to_lagrange_points_raw[j,i,1],euler_to_lagrange_points_raw[j,i,2]])
	    end
	end

    if z == nothing
        z = zeros(size(euler_to_lagrange_points))
        for I in eachindex(euler_to_lagrange_points)
            if isnan((euler_to_lagrange_points[I])[1])
                z[I] = NaN
            else
                z[I] = evaluate_function_from_nodevals(ctx,u_values,euler_to_lagrange_points[I],NaN)
            end
        end
    end

    if postprocessor != nothing
       z =  postprocessor(z)
    end

    seriestype --> :heatmap
    fill --> true
    aspect_ratio --> 1
    xlim --> (LL[1],UR[1])
    ylim --> (LL[2],UR[2])
    x1,x2,z
end


"""
    plot_u_eulerian(ctx,dof_vals,inverse_flow_map,
        LL,UR,nx,ny,
        euler_to_lagrange_points=nothing, only_get_lagrange_points=false,
        z=nothing,
        postprocessor=nothing,
        bdata=nothing, ....)
Plot a heatmap of a function in eulerian coordinates, i.e. the pushforward of \$f\$. This is given by
\$f \\circ \\Phi^{-1}\$, \$f\$ is a function defined on the grid `ctx`, represented by coefficients given by `dof_vals` (with possible boundary conditions given in `bdata`)

The argument `inverse_flow_map` is \$\\Phi^{-1}\$.

The resulting plot is on a regular `nx` by `ny` grid on the grid with lower left corner `LL` and upper right corner `UR`.

Points that fall outside of the domain represented by `ctx` are plotted as `NaN`, which results in transparency.

One can pass values to be plotted directly by providing them in an array in the argument `z`. `postprocessor` can modify the values being plotted, `return_scalar_field` results in these values being returned.
 See the source code for further details.  Additional arguments are passed to `Plots.heatmap`

Inverse flow maps are computed in parallel if there are multiple workers.
"""
plot_u_eulerian


function compute_euler_to_lagrange_points_raw(inv_flow_map,x1,x2)
    euler_to_lagrange_points_raw = SharedArray{Float64}(length(x2),length(x1),2)
    @sync @distributed for i in eachindex(x1)
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


@userplot Plot_Spectrum
@recipe function  f(as::Plot_Spectrum)
    λ = as.args[1]
    seriestype --> :scatter
    (real.(λ),imag.(λ))
end


@userplot Plot_Real_Spectrum
@recipe function  f(as::Plot_Real_Spectrum)
    λ = as.args[1]
    seriestype --> :scatter
    (1:length(λ),real.(λ))
end

struct frameCollection
    frames::Vector{Any}
end


function Base.iterate(col::frameCollection, state=0 )
    if state == length(col.frames)
        return nothing
    end
    return col.frames[state+1] , state+1
end


function eulerian_videos(ctx,
                         us::Function,
                         inverse_flow_map_t,
                         t0,tf,
                         nx, ny,nt,
                         LL, UR,
                         num_videos=1;
                         bdata=nothing,
                         extra_kwargs_fun=nothing,kwargs...)
    allvideos = [frameCollection([]) for i in 1:num_videos]

    for (index,t) in enumerate(range(t0,stop=tf,length=nt))
    	print("Processing frame $index")
        if t != t0
        	current_inv_flow_map = x -> inverse_flow_map_t(t,x)
        else
            current_inv_flow_map = x->x
        end
        x1 = range(LL[1],stop=UR[1],length=nx)
        x2 = range(LL[2],stop=UR[2],length=ny)
        euler_to_lagrange_points_raw = compute_euler_to_lagrange_points_raw(current_inv_flow_map,x1,x2)
        euler_to_lagrange_points = [zero(Tensors.Vec{2}) for y in x2, x in x1]
        for i in 1:nx, j in 1:ny
            euler_to_lagrange_points[j,i] = Tensors.Vec{2}([euler_to_lagrange_points_raw[j,i,1],euler_to_lagrange_points_raw[j,i,2]])
	    end

        zs_all = SharedArray{Float64}(ny,nx,num_videos)
        current_us = SharedArray{Float64}(ctx.n,num_videos)
        for i in 1:num_videos
            current_us[:,i] = get_full_nodevals(ctx,us(i,t); bdata = bdata)
        end
        @sync @distributed for xindex in 1:nx
            #@distributed for current_index in eachindex(z_all)
            for yindex in 1:ny
                for i in 1:num_videos
            	    current_u = current_us[:,i]
                    zs_all[yindex,xindex, i]= evaluate_function_from_nodevals(
                        ctx, current_u,
                        euler_to_lagrange_points[yindex,xindex])
                end
            end
        end
        for i in 1:num_videos
    	    if extra_kwargs_fun != nothing
        		curframe = plot_u_eulerian(ctx, zeros(ctx.n) , current_inv_flow_map,LL,UR, nx,ny;zs=zs_all[:,:,i],euler_to_lagrange_points=euler_to_lagrange_points,extra_kwargs_fun(i,t)...,kwargs...);
    	    else
        		curframe = plot_u_eulerian(ctx, zeros(ctx.n), current_inv_flow_map,LL,UR, nx,ny;zs=zs_all[:,:,i],euler_to_lagrange_points=euler_to_lagrange_points,kwargs...);
    	    end
            push!(allvideos[i].frames, curframe)
        end
    end;
    return allvideos
end




"""
     eulerian_videos(ctx, us, inverse_flow_map_t, t0,tf, nx,ny,nt, LL,UR, num_videos=1;
        extra_kwargs_fun=nothing, ...)

Create `num_videos::Int` videos in eulerian coordinates, i.e. where the time \$t\$ is varied, plot \$f_i \\circ \\Phi_t^0\$ for \$f_1, \\dots\$.

`us(i,t)` is a vector of dofs to be plotted at time `t` for the `i`th video.

`inverse_flow_map_t(t,x)` is \$\\Phi_t^0(x)\$

`t0, tf`  are initial and final time. Spatial bounds are given by `LL,UR`

`nx,ny,nt` give the number of points in each direction.

`extra_kwargs_fun(i,t)` can be used to provide additional keyword arguments to Plots.heatmap()

Additional kwargs are passed down to `plot_eulerian_video`

As much as possible is done in parallel.

Returns a Vector of iterables `result`. Call `Plots.animate(result[i])` to get an animation.
"""
eulerian_videos


"""
    eulerian_video(ctx, u, inverse_flow_map_t,t0,tf, nx, ny, nt, LL, UR;extra_kwargs_fun=nothing,...)

Like `eulerian_videos`, but `u(t)` is a vector of dofs, and `extra_kwargs_fun(t)` gives extra keyword arguments.
Returns only one result, on which `Plots.animate()` can be applied.
"""
function eulerian_video(ctx, u::Function, inverse_flow_map_t,t0,tf, nx, ny, nt, LL, UR;extra_kwargs_fun=nothing,kwargs...)
    usfun = (index,t) -> u(t)
    if (extra_kwargs_fun!= nothing)
        extra_kwargs_fun_out = (i,t) -> extra_kwargs_fun(t)
    else
        extra_kwargs_fun_out = nothing
    end
    return eulerian_videos(ctx,usfun,inverse_flow_map_t, t0,tf, nx,ny,nt, LL,UR,1;extra_kwargs_fun=extra_kwargs_fun_out,kwargs...)[1]
end

#=
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
    x1 = range(LL[1],stop=UR[1],length=nx)
    x2 = range(LL[2],stop=UR[2],length=ny)
    allpoints = [
        Vec{2}((x,y)) for y in x2, x in x1
    ]
    allpoints_initial = copy(allpoints)
    #
    times = range(t0,stop=tf,length=nt)
    x1p = [p[1] for p in allpoints]
    x2p = [p[2] for p in allpoints]
    ut = dof2node(ctx,corrected_u(t0))
    val = [evaluate_function_from_nodevals(ctx,ut,[p[1],p[2]]) for p in allpoints]
    #
    res = [scatter(x1p[1:end],x2p[1:end],zcolor=val[1:end],
        xlim=(LL_big[1],UR_big[1]),ylim=(LL_big[2],UR_big[2]),legend=false,
        marker=:square,markersize=300.0 /nx,markerstrokewidth=0;kwargs...)]
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
                marker=:square,markersize=70. /nx,markerstrokewidth=0;kwargs...)
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
=#


@userplot Plot_FTLE
@recipe function f(
            as::Plot_FTLE;
    	   tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),
		   #existing_plot=nothing, TODO 1.0
           flip_y=false, check_inbounds=always_true,
           vari=false
           )
   odefun=as.args[1]
   p = as.args[2]
   tspan = as.args[3]
   LL = as.args[4]
   UR = as.args[5]
   if length(as.args) >= 6
       nx = as.args[6]
   else
       nx = 100
   end
   if length(as.args) >= 7
       ny = as.args[7]
   else
       ny = 100
   end
   if length(as.args) >= 8
       δ = as.args[8]
   else
       δ = 1e-9
   end

    x1 = collect(range(LL[1] + 1.e-8,stop= UR[1] - 1.e-8,length=nx))
    x2 = collect(range(LL[2] + 1.e-8, stop=UR[2] - 1.e-8,length=ny))
    if flip_y
        x2 = reverse(x2)
    end
    #Initialize FTLE-field with NaNs
    FTLE = SharedArray{Float64,2}(ny,nx)
    FTLE .= NaN
    nancounter, nonancounter = @sync @distributed ((x,y)->x.+y) for i in eachindex(x1)
        nancounter_local = 0
        nonancounter_local = 0
        for j in eachindex(x2)
            if check_inbounds(x1[i],x2[j],p)

                        FTLE[j,i] = 1 / (2(tspan[end]-tspan[1])) *
                          log(maximum(eigvals(eigen(CG_tensor_vari(odefun, [x1[i],x2[j]], [tspan[1],tspan[end]];
                                tolerance=tolerance, p=p, solver=solver)))))
                        nonancounter_local += 1
                try
                    if !vari
                        FTLE[j,i] = 1 / (2(tspan[end]-tspan[1])) *
                          log(maximum(eigvals(eigen(CG_tensor(odefun, [x1[i],x2[j]], [tspan[1],tspan[end]], δ;
                                tolerance=tolerance, p=p, solver=solver)))))
                        nonancounter_local += 1
                    else
                        FTLE[j,i] = 1 / (2(tspan[end]-tspan[1])) *
                          log(maximum(eigvals(eigen(CG_tensor_vari(odefun, [x1[i],x2[j]], [tspan[1],tspan[end]], δ;
                                tolerance=tolerance, p=p, solver=solver)))))
                        nonancounter_local += 1
                    end
                catch e
                    nancounter_local += 1
                end
            end
        end
        (nancounter_local, nonancounter_local)
    end

    print("plot_ftle ignored $nancounter NaN values ($nonancounter were good)")
    seriestype --> :heatmap

    if flip_y == true
        x2 = reverse(x2)
        x2 *= -1
    end

    x1,x2,FTLE

    #= TODO
    if existing_plot == nothing
        return Plots.heatmap(x1,x2,FTLE; kwargs...)
    else
        return Plots.heatmap!(existing_plot,x1,x2,FTLE; kwargs...)
    end
    =#
end


"""
    plot_ftle(odefun,p,tspan,LL,UR,nx,ny;
        δ=1e-9,tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),
        existing_plot=nothing,flip_y=false, check_inbounds=always_true,vari=true)

Make a heatmap of a FTLE field using finite differences.
If `existing_plot` is given a value, plot using `heatmap!` on top of it.
If `flip_y` is true, then flip the y-coordinate (needed sometimes due to a bug in Plots).
Points where `check_inbounds(x[1],x[2],p) == false` are set to `NaN` (i.e. transparent).
"""
plot_ftle
