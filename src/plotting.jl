
"""
    plot_u(ctx, dof_vals, nx=100, ny=100, LL, UR; bdata=nothing, kwargs...)

Plot the function with coefficients (in dof order, possible boundary conditions in `bdata`) given by `dof_vals` on the grid `ctx`.
The domain to be plotted on is given by `ctx.spatialBounds`.
The function is evaluated on a regular `nx` by `ny` grid, the resulting plot is a heatmap.
Keyword arguments are passed down to `plot_u_eulerian`, which this function calls internally.
"""
function plot_u(
    ctx::GridContext{dim},
    dof_vals::Vector{<:Real},
    nx=100,
    ny=100,
    LL=ctx.spatialBounds[1],
    UR=ctx.spatialBounds[2];
    bdata=nothing,
    kwargs...,
) where {dim}
    if dim in (1, 2)
        plot_u_eulerian(ctx, dof_vals, identity, LL, UR, nx, ny; bdata=bdata, kwargs...)
    else
        throw(AssertionError("Not yet implemented"))
    end
end
function plot_u(ctx, dof_vals::Vector{<:Complex}, args...; kwargs...)
    return plot_u(ctx, real.(dof_vals), args...; title="Plotting real part!", kwargs...)
end

function plot_u!(
    ctx::GridContext{dim},
    dof_vals::Vector{<:Real},
    nx=100,
    ny=100;
    bdata=nothing,
    kwargs...,
) where {dim}
    if dim in (1, 2)
        plot_u_eulerian!(
            ctx,
            dof_vals,
            identity,
            ctx.spatialBounds[1],
            ctx.spatialBounds[2],
            nx,
            ny;
            bdata=bdata,
            kwargs...,
        )
    else
        throw(AssertionError("Not yet implemented"))
    end
end

RecipesBase.@userplot Plot_U_Eulerian
RecipesBase.@recipe function f(
    as::Plot_U_Eulerian;
    bdata=nothing,
    euler_to_lagrange_points=nothing,
    z=nothing,
    postprocessor=nothing,
    return_scalar_field=false,
    throw_errors=false,
)
    ctx::GridContext = as.args[1]
    dof_vals::Vector{Float64} = as.args[2]
    inverse_flow_map::Function = as.args[3]
    LL = as.args[4]
    LL isa AbstractVector &&
        @warn "Use tuples (round brackets) rather than vectors [square brackets] for domain corners"
    UR = as.args[5]
    UR isa AbstractVector &&
        @warn "Use tuples (round brackets) rather than vectors [square brackets] for domain corners"
    if ctx isa GridContext{2}
        nx = length(as.args) >= 6 ? as.args[6] : 100
        ny = length(as.args) >= 7 ? as.args[7] : 100

        u_values = get_full_dofvals(ctx, dof_vals; bdata=bdata)
        x1 = range(LL[1], stop=UR[1], length=nx)
        x2 = range(LL[2], stop=UR[2], length=ny)
        if euler_to_lagrange_points === nothing
            euler_to_lagrange_points_raw = compute_euler_to_lagrange_points_raw(
                inverse_flow_map,
                [x1, x2],
                throw_errors=throw_errors,
            )
            euler_to_lagrange_points = [zero(Vec{2}) for y in x2, x in x1]
            for i in 1:nx, j in 1:ny
                euler_to_lagrange_points[j, i] = Vec{2}((euler_to_lagrange_points_raw[j, i, 1],
                                                         euler_to_lagrange_points_raw[j, i, 2]))
            end
        end

        if z === nothing
            z_raw = evaluate_function_from_dofvals_multiple(
                ctx,
                u_values[:, :],
                vec(euler_to_lagrange_points);
                outside_value=NaN,
                throw_errors=throw_errors,
            )
            z = reshape(z_raw.nzval, size(euler_to_lagrange_points))
        end

        if postprocessor !== nothing
            z = postprocessor(z)
        end

        seriestype --> :heatmap
        fill --> true
        aspect_ratio --> 1
        xlims --> (LL[1], UR[1])
        ylims --> (LL[2], UR[2])
        x1, x2, z
    elseif ctx isa GridContext{1}
        nx = length(as.args) >= 6 ? as.args[6] : 100
        u_values = get_full_dofvals(ctx, dof_vals; bdata=bdata)
        x1 = range(LL[1], stop=UR[1], length=nx)

        if euler_to_lagrange_points === nothing
            euler_to_lagrange_points_raw =
                compute_euler_to_lagrange_points_raw(inverse_flow_map, [x1])
            euler_to_lagrange_points = [zero(Vec{1}) for x in x1]
            for i in 1:nx
                euler_to_lagrange_points[i] = Vec{1}([euler_to_lagrange_points_raw[i]])
            end
        end

        if z === nothing
            z_raw = evaluate_function_from_dofvals_multiple(
                ctx,
                u_values[:, :],
                vec(euler_to_lagrange_points),
                outside_value=NaN,
            )
            z = reshape(z_raw.nzval, size(euler_to_lagrange_points))
        end

        if postprocessor !== nothing
            z = postprocessor(z)
        end

        seriestype --> :line
        xlim --> (LL[1], UR[1])
        x1, z
    end
end

"""
    plot_u_eulerian(ctx, dof_vals, inverse_flow_map, LL, UR[, nx, ny];
        euler_to_lagrange_points=nothing,
        only_get_lagrange_points=false,
        z=nothing,
        postprocessor=nothing,
        bdata=nothing,
        kwargs...)

Plot a heatmap of a function in Eulerian coordinates, i.e., the pushforward of
\$f\$. This is given by \$f \\circ \\Phi^{-1}\$, \$f\$ is a function defined on
the grid `ctx`, represented by coefficients given by `dof_vals` (with possible
boundary conditions given in `bdata`).

The argument `inverse_flow_map` is \$\\Phi^{-1}\$.

The resulting plot is on a regular `nx x ny` grid with lower left corner `LL`
and upper right corner `UR`. Points that fall outside of the domain represented
by `ctx` are plotted as `NaN`, which results in transparency.

One can pass values to be plotted directly by providing them in an array in the
argument `z`. `postprocessor` can modify the values being plotted,
`return_scalar_field` results in these values being returned. See the source
code for further details.  Additional keyword arguments are passed to `Plots.heatmap`.

Inverse flow maps are computed in parallel if there are multiple workers.
"""
plot_u_eulerian

function compute_euler_to_lagrange_points_raw(
    inv_flow_map,
    xi;
    throw_errors=false,
)
    @assert length(xi) ∈ [1, 2]
    if length(xi) == 1
        x1 = xi[1]
        euler_to_lagrange_points_raw = SharedArray{Float64}(length(x1))
        @sync @distributed for i in eachindex(x1)
            point = SVector{1}(x1[i])
            try
                back = inv_flow_map(point)
                euler_to_lagrange_points_raw[i] = back[1]
            catch e
                euler_to_lagrange_points_raw[i] = NaN
            end
        end
        return euler_to_lagrange_points_raw
    elseif length(xi) == 2
        x1, x2 = xi
        euler_to_lagrange_points_raw =
            SharedArray{Float64}(length(x2), length(x1), 2)
        @sync @distributed for i in eachindex(x1)
            for j in eachindex(x2)
                point = SVector{2}(x1[i], x2[j])
                try
                    back = inv_flow_map(point)
                    euler_to_lagrange_points_raw[j, i, 1] = back[1]
                    euler_to_lagrange_points_raw[j, i, 2] = back[2]
                catch e
                    # if isa(e,InexactError)
                    if !throw_errors
                        euler_to_lagrange_points_raw[j, i, 1] = NaN
                        euler_to_lagrange_points_raw[j, i, 2] = NaN
                    else
                        rethrow(e)
                    end
                end
            end
        end
        return euler_to_lagrange_points_raw
    end
end

RecipesBase.@userplot Plot_Spectrum
RecipesBase.@recipe function f(as::Plot_Spectrum)
    λ = as.args[1]
    seriestype := :scatter
    (real.(λ), imag.(λ))
end

RecipesBase.@userplot Plot_Real_Spectrum
RecipesBase.@recipe function f(as::Plot_Real_Spectrum)
    λ = as.args[1]
    seriestype := :scatter
    (1:length(λ), real.(λ))
end

struct FrameCollection
    frames::Vector{Any}
end

function Base.iterate(col::FrameCollection, state=0)
    if state == length(col.frames)
        return nothing
    end
    return col.frames[state + 1], state + 1
end

function eulerian_videos(
    ctx,
    us::Function,
    inverse_flow_map_t,
    t0,
    tf,
    nx,
    ny,
    nt,
    LL,
    UR,
    num_videos=1;
    bdata=nothing,
    extra_kwargs_fun=nothing,
    throw_errors=false,
    kwargs...,
)
    allvideos = [FrameCollection([]) for i in 1:num_videos]

    for (index, t) in enumerate(range(t0, stop=tf, length=nt))
        print("Processing frame $index")
        if t != t0
            current_inv_flow_map = x -> inverse_flow_map_t(t, x)
        else
            current_inv_flow_map = x -> x
        end
        x1 = range(LL[1], stop=UR[1], length=nx)
        x2 = range(LL[2], stop=UR[2], length=ny)
        euler_to_lagrange_points_raw = compute_euler_to_lagrange_points_raw(
            current_inv_flow_map,
            [x1, x2],
            throw_errors=throw_errors,
        )
        euler_to_lagrange_points = [zero(Vec{2}) for y in x2, x in x1]
        for i in 1:nx, j in 1:ny
            euler_to_lagrange_points[j, i] = Vec{2}([
                euler_to_lagrange_points_raw[j, i, 1],
                euler_to_lagrange_points_raw[j, i, 2],
            ])
        end

        zs_all = SharedArray{Float64}(ny, nx, num_videos)
        current_us = SharedArray{Float64}(ctx.n, num_videos)
        for i in 1:num_videos
            current_us[:, i] =
                get_full_dofvals(ctx, us(i, t); bdata=bdata)[ctx.node_to_dof]
        end
        @sync @distributed for xindex in 1:nx
            # @distributed for current_index in eachindex(z_all)
            for yindex in 1:ny
                for i in 1:num_videos
                    current_u = current_us[:, i]
                    zs_all[yindex, xindex, i] =
                        evaluate_function_from_node_or_cellvals(
                            ctx,
                            current_u,
                            euler_to_lagrange_points[yindex, xindex];
                            outside_value=NaN,
                        )
                end
            end
        end
        for i in 1:num_videos
            if extra_kwargs_fun !== nothing
                curframe = plot_u_eulerian(
                    ctx,
                    zeros(ctx.n),
                    current_inv_flow_map,
                    LL,
                    UR,
                    nx,
                    ny;
                    zs=zs_all[:, :, i],
                    euler_to_lagrange_points=euler_to_lagrange_points,
                    extra_kwargs_fun(i, t)...,
                    kwargs...,
                )
            else
                curframe = plot_u_eulerian(
                    ctx,
                    zeros(ctx.n),
                    current_inv_flow_map,
                    LL,
                    UR,
                    nx,
                    ny;
                    zs=zs_all[:, :, i],
                    euler_to_lagrange_points=euler_to_lagrange_points,
                    kwargs...,
                )
            end
            push!(allvideos[i].frames, curframe)
        end
    end
    return allvideos
end

"""
    eulerian_videos(ctx, us, inverse_flow_map_t, t0,tf, nx,ny,nt, LL,UR, num_videos=1;
        extra_kwargs_fun=nothing, ...)

Create `num_videos::Int` videos in eulerian coordinates, i.e., where the time
\$t\$ is varied, plot \$f_i \\circ \\Phi_t^0\$ for \$f_1, \\dots\$.

## Arguments
* `us(i,t)` is a vector of dofs to be plotted at time `t` for the `i`th video.
* `inverse_flow_map_t(t,x)` is \$\\Phi_t^0(x)\$.
* `t0, tf`  are initial and final time.
* `LL` and `UR` are the coordinates of the domain's corners.
* `nx`, `ny`, and `nt` determine the number of points in each direction.
* `extra_kwargs_fun(i,t)` can be used to provide additional keyword arguments to
  Plots.heatmap().

Additional kwargs are passed on to `plot_eulerian_video`.

As much as possible is done in parallel.

Returns a Vector of iterables `result`. Call `Plots.animate(result[i])` to get an animation.
"""
eulerian_videos

"""
    eulerian_video(ctx, u, inverse_flow_map_t, t0, tf, nx, ny, nt, LL, UR; extra_kwargs_fun=nothing, ...)

Like `eulerian_videos`, but `u(t)` is a vector of dofs, and `extra_kwargs_fun(t)`
gives extra keyword arguments. Returns only one result, on which `Plots.animate()`
can be applied for an animation.
"""
function eulerian_video(
    ctx,
    u::Function,
    inverse_flow_map_t,
    t0,
    tf,
    nx,
    ny,
    nt,
    LL,
    UR;
    extra_kwargs_fun=nothing,
    kwargs...,
)
    usfun = (index, t) -> u(t)
    if (extra_kwargs_fun !== nothing)
        extra_kwargs_fun_out = (i, t) -> extra_kwargs_fun(t)
    else
        extra_kwargs_fun_out = nothing
    end
    return eulerian_videos(ctx, usfun, inverse_flow_map_t, t0, tf, nx, ny, nt, LL, UR, 1;
        extra_kwargs_fun=extra_kwargs_fun_out, kwargs...)[1]
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
end =#

RecipesBase.@userplot Plot_Grid_Raw
RecipesBase.@recipe function f(
    as::Plot_Grid_Raw;
)
    ctx::GridContext = as.args[1]
    nodes = length(as.args) >= 2 ? as.args[2] : [x.x for x in ctx.grid.nodes]
    shown_triangles = length(as.args) >= 3 ? as.args[3] : [true for x in ctx.grid.cells]

    toplot_1 = Array{Float64}[]
    toplot_2 = Array{Float64}[]
    for cell in ctx.grid.cells
        x_nodes = Float64[]
        y_nodes = Float64[]
        for n in cell.nodes
            push!(x_nodes, nodes[n][1])
            push!(y_nodes, nodes[n][2])
        end
        push!(toplot_1, x_nodes)
        push!(toplot_2, y_nodes)
    end
    seriestype --> :shape
    aspect_ratio --> 1
    label --> ""
    toplot_1[shown_triangles], toplot_2[shown_triangles]
end

function plot_grid(ctx, as=[x.x for x in ctx.grid.nodes], shaded_triangles=[false for x in ctx.grid.cells];kwargs...)
    fig = plot_grid_raw(ctx, as, shaded_triangles, fillcolor=ColorTypes.RGBA(1.0, 0.0, 0.0, 0.2);kwargs...)
    return plot_grid_raw!(fig, ctx, as, map(x -> !x, shaded_triangles), fillcolor=ColorTypes.RGBA(0.0, 0.0, 0.0, 0.0);kwargs...)
end

# seriestype --> :heatmap
# fill --> true
# xlim --> (LL[1], UR[1])
# ylim --> (LL[2], UR[2])


RecipesBase.@userplot Plot_FTLE
RecipesBase.@recipe function f(
    as::Plot_FTLE;
    tolerance=1e-4,
    solver=ODE.BS5(),
    # existing_plot=nothing, TODO 1.0
    flip_y=false,
    check_inbounds=always_true,
    pass_on_errors=false,
)
    odefun = as.args[1]
    p = as.args[2]
    tspan = as.args[3]
    LL = as.args[4]
    UR = as.args[5]
    nx = length(as.args) >= 6 ? as.args[6] : 100
    ny = length(as.args) >= 7 ? as.args[7] : 100
    δ = length(as.args) >= 8 ? as.args[8] : 1e-9

    x1 = collect(range(LL[1] + 1.e-8, stop=UR[1] - 1.e-8, length=nx))
    x2 = collect(range(LL[2] + 1.e-8, stop=UR[2] - 1.e-8, length=ny))
    if flip_y
        x2 = reverse(x2)
    end
    # Initialize FTLE-field with NaNs
    FTLE = SharedArray{Float64,2}(ny, nx)
    totalelements = ny * nx
    FTLE .= NaN
    nancounter, nonancounter =
        @sync @distributed ((x, y) -> x .+ y) for c in 0:(totalelements - 1)
            j, i = divrem(c, ny) .+ (1, 1)
            nancounter_local = 0
            nonancounter_local = 0
            if check_inbounds(x1[j], x2[i], p)
                try
                    cgtensor = CG_tensor(
                        odefun,
                        [x1[j], x2[i]],
                        [tspan[1], tspan[end]],
                        δ;
                        tolerance=tolerance,
                        p=p,
                        solver=solver,
                    )
                    FTLE[c + 1] =
                        1 / (2 * (tspan[end] - tspan[1])) *
                        log(maximum(eigvals(eigen(cgtensor))))
                    if isinf(FTLE[c + 1])
                        FTLE[c + 1] = NaN
                    end
                    if !isnan(FTLE[c + 1])
                        nonancounter_local += 1
                    else
                        nancounter_local += 1
                    end
                catch e
                    if pass_on_errors
                        rethrow(e)
                    end
                    nancounter_local += 1
                end
            end
            (nancounter_local, nonancounter_local)
        end
    minval = minimum(FTLE)
    maxval = maximum(FTLE)

    @info "plot_ftle ignored $nancounter NaN values ($nonancounter were good). Bounds ($minval,$maxval)"

    if flip_y
        x2 = reverse(x2)
        x2 *= -1
    end
    seriestype := :heatmap
    x1, x2, FTLE
end

"""
    plot_ftle(odefun, p, tspan, LL, UR, nx, ny;
        δ=1e-9, tolerance=1e-4, solver=OrdinaryDiffEq.BS5(),
        existing_plot=nothing, flip_y=false, check_inbounds=always_true, pass_on_errors=false)

Make a heatmap of a FTLE field using finite differences. If `existing_plot` is
given a value, plot using `heatmap!` on top of it. If `flip_y` is true, then flip
the y-coordinate (needed sometimes due to a bug in Plots). Points where
`check_inbounds(x[1], x[2], p) == false` are set to `NaN`, i.e., plotted transparently.
Unless `pass_on_errors` is set to `true`, errors from calculating FTLE values are
caught and ignored.
"""
plot_ftle

"""
    plot_vortices(vortices, singularities, LL, UR; kwargs...)

Plots the output of [`ellipticLCS`](@ref) on the domain spanned by the lower left
corner `LL` and the upper right corner `UR`.

## Keyword arguments

* `bg=nothing`: array of some background scalar field, or nothing;
* `xspan`: x-axis for the scalar field, ignored if `bg` is an `AxisArray`;
* `yspan`: y-axis for the scalar field, ignored if `bg` is an `AxisArray`;
* `logBg=true`: whether to take the `log10` of the scalar field;
* `include_singularities=true`: whether to plot singularities;
* `showlabel=false`: whether to show labels with the respective parameter values;
* other keyword arguments are passed on to the plotting functions.
"""
function plot_vortices(
    vortices,
    singularities,
    LL,
    UR;
    bg=nothing,
    xspan=nothing,
    yspan=nothing,
    logBg=true,
    include_singularities=true,
    showlabel=false,
    kwargs...,
)
    if bg !== nothing
        fig = bg isa AxisArray ?
            plot_field(bg, bg.axes[1].val, bg.axes[2].val, LL, UR; logBg=logBg, kwargs...) :
            plot_field(bg, xspan, yspan, LL, UR; logBg=logBg, kwargs...)
    else
        fig = empty_heatmap(LL, UR; kwargs...)
    end
    for v in vortices, b in v.barriers
        plot_barrier!(b; showlabel=showlabel, kwargs...)
    end
    if include_singularities
        plot_singularities!(singularities; kwargs...)
    end
    return fig
end

RecipesBase.@userplot Plot_Field
RecipesBase.@recipe function f(as::Plot_Field; logBg=true)
    bg = as.args[1]
    xspan = as.args[2]
    yspan = as.args[3]
    LL = as.args[4]
    UR = as.args[5]
    seriestype := :heatmap
    seriescolor --> :viridis
    xlims := (LL[1], UR[1])
    ylims := (LL[2], UR[2])
    aspect_ratio --> 1
    xspan, yspan, permutedims(logBg ? log10.(bg) : bg)
end

"""
    plot_field(field, xspan, yspan, LL, UR; logBg=true)
    plot_field(field::AxisArray, LL, UR; logBg=true)

Makes a heatmap plot of the scalar field given as the `AxisArray` `field` on the
domain spanned by the lower-left corner `LL` and the upper-right corner `UR`.
The keyword argument `logBg` determines whether the `log10` of the scalar field
is plotted.
"""
plot_field

"""
    plot_field!

Same as [`plot_field`](@ref), but adds the output to the currently active plot.
"""
plot_field!

RecipesBase.@userplot Empty_Heatmap
RecipesBase.@recipe function f(as::Empty_Heatmap)
    LL = as.args[1]
    UR = as.args[2]
    xspan = range(LL[1], stop=UR[1], length=2)
    yspan = range(LL[2], stop=UR[2], length=2)
    bg = [NaN for _ in yspan, _ in xspan]
    xlims := (LL[1], UR[1])
    ylims := (LL[2], UR[2])
    colorbar := :none
    seriestype := :heatmap
    aspect_ratio --> 1
    xspan, yspan, bg
end

RecipesBase.@userplot Plot_Barrier
RecipesBase.@recipe function f(as::Plot_Barrier; showlabel=false)
    barrier = as.args[1]
    curve = [p.data for p in barrier.curve]
    linewidth --> 3
    label := showlabel ? "p = $(round(barrier.p, digits=2))" : ""
    linecolor --> :red
    if showlabel
        linecolor := :auto
    end
    aspect_ratio --> 1
    curve
end

"""
    plot_barrier(barrier::EllipticBarrier; showlabel=false)

Makes a line plot of the `barrier`, where the keyword argument `showlabel`
determines whether a legend entry showing the barrier's parameter value is
created.
"""
plot_barrier

"""
    plot_barrier!(barrier::EllipticBarrier; showlabel=false)

Same as [`plot_barrier`](@ref), but adds the output to the currently active plot.
"""
plot_barrier!

function index2color(ind)
    if ind == 1 // 1
        return :white
    end
    if ind == 1 // 2
        return :orange
    end
    if ind == -1 // 2
        return :blue
    end
    if ind == 3 // 2
        return :purple
    end
    if ind > 3 // 2
        return :brown
    end
    if ind == -3 // 2
        return :green
    end
    if ind < -3 // 2
        return :black
    end
end

RecipesBase.@userplot Plot_Singularities
RecipesBase.@recipe function f(as::Plot_Singularities)
    singularities = as.args[1]
    singularity_colors = [index2color(s.index) for s in singularities]
    points = [s.coords.data for s in singularities]
    seriestype := :scatter
    seriescolor := singularity_colors
    label := ""
    points
end

"""
    plot_singularities(singularities::Vector{Singularity})

Makes a scatter plot of the `singularities`, with coloring depending on the
respective index:
* index = 1: white
* index = 1/2: orange
* index = -1/2: blue
* index = 3/2: purple
* index > 3/2: brown
* index = -3/2: green
* ind < -3/2: black
"""
plot_singularities

"""
    plot_singularities!(singularities::Vector{Singularity})

Same as [`plot_singularities`](@ref), but adds the output to the currently active plot.
"""
plot_singularities!
