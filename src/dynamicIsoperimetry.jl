# functions are all for 2D only, as well as not really working with periodic boundaries

function apply2curve(flowmap, curve::Curve2{T}) where {T}
    moved_points = [flowmap(x) for x in zip(coordinates(curve)...)]
    moved_points = [flowmap(x) for x in zip(coordinates(curve)...)]
    return Curve2(moved_points)
end

function get_length(curve, tensorfield=(x,y) -> 1)
    xs, ys = coordinates(curve)
    n = length(xs)
    result = 0
    for j in 1:(n-1)
        current_x, current_y   = xs[j], ys[j]
        next_x, next_y         = xs[j+1], ys[j+1]
        approxTangentVector = [next_x - current_x, next_y - current_y]
        result += sqrt(approxTangentVector ⋅ (tensorfield(curx, cury) * approxTangentVector))
    end
    return result
end


# If the first point is not the last point, we might get vol(Ω)-area instead
function get_euclidean_area(ctx,curve;tolerance=0.0)

    curve = close_curve(ctx,curve,tolerance=tolerance)

    xs, ys = coordinates(curve)
    center_x, center_y = mean(xs), mean(ys)
    n = length(xs)
    result = 0.0
    for j in 1:(n-1)
        current_x, current_y   = xs[j], ys[j]
        next_x, next_y         = xs[j+1], ys[j+1]

        x1 = current_x - center_x
        x2 = next_x - center_x
        y1 = current_y - center_y
        y2 = next_y - center_y

        area = 0.5*det([x2 y2; x1 y1])
        result +=area
    end
    return abs(result)
end


# from contour.jl we only get the guarantee that either the first and last points are equal, or the contour starts and
# ends at the boundary of Ω. in the last case, we will have to construct a closed curve e.g. in order to determine the
# enclosed area
function close_curve(ctx,curve::Curve2{T};tolerance=0.0,orientation=:clockwise) where {T}
    xs, ys = coordinates(curve)
    start_x, start_y, end_x, end_y = xs[1], ys[1], xs[end], ys[end]

    if (start_x == end_x && start_y == end_y)
        return curve
    end

    points = [SA[x,y] for (x,y) in zip(xs,ys)]

    LL, UR = ctx.spatialBounds
    #Curve2 wants StaticArrays
    LL, UR = SA[LL[1], LL[2]], SA[UR[1], UR[2]]
    LR, UL = SA[UR[1], LL[2]], SA[LL[1], UR[2]]

    # We have a rectangle, and order its sides like
    #     3                         3
    #    --                        --
    #  2|  |4  for clockwise,    4|  |2 for anticlockwise orientation.
    #    --                        --
    #     1                         1
    #
    # corners[i] then is the next corner in the respective orientation.

    if orientation==:clockwise
        corners = [LL, UL, UR, LR]
    elseif orientation ==:anticlockwise
        corners = [LR, UR, UL, LL]
    else
        error("Unknown orientation '",orientation,"'!")
    end

    if     abs(end_y-corners[1][2])<=tolerance
        side = 1
    elseif abs(end_x-corners[2][1])<=tolerance
        side = 2
    elseif abs(end_y-corners[3][2])<=tolerance
        side = 3
    elseif abs(end_x-corners[4][1])<=tolerance
        side = 4
    else                              # point does not lie on the boundary
        error("A curve has either to be closed or start and end at the boundary!")
    end

    i = 0
    while(abs(start_x-end_x)>tolerance && abs(start_y-end_y)>tolerance)
        push!(points,corners[side])
        end_x = corners[side][1]
        end_y = corners[side][2]
        side = (side%4)+1
        i += 1
        if i>4  # might happen if the first point is not on the border, prevent infinite loops
            error("A curve has either to be closed or start and end at the boundary!")
        end
    end
    push!(points,points[1])  # close curve
    return Curve2(points)
end


function get_function_values(ctx, u; x_resolution=nothing, y_resolution=nothing, bdata=BoundaryData())

    if isnothing(x_resolution)
        x_resolution = ctx.numberOfPointsInEachDirection[1]
    end
    if isnothing(y_resolution)
        y_resolution = ctx.numberOfPointsInEachDirection[2]
    end

    xs = range(ctx.spatialBounds[1][1], stop=ctx.spatialBounds[2][1], length=x_resolution)
    ys = range(ctx.spatialBounds[1][2], stop=ctx.spatialBounds[2][2], length=y_resolution)

    u_dofvals = undoBCS(ctx, u, bdata)
    u_nodevals = u_dofvals[ctx.node_to_dof]

    fs = [evaluate_function_from_node_or_cellvals(ctx, u_nodevals, Vec{2}((x,y)))
        for x in xs, y in ys]

    return xs,ys,fs
end

"""
    get_levelset(ctx, u, c;
        x_resolution=nothing, y_resolution=nothing, bdata=BoundaryData())

Returns the levelset of u corresponding to the function value c
"""
function get_levelset(ctx, u, c;
        x_resolution=nothing, y_resolution=nothing, bdata=BoundaryData())

    xs, ys, fs = get_function_values(ctx,u,x_resolution=x_resolution,y_resolution=y_resolution,bdata=bdata)

    return Contour.contour(xs, ys, fs, c)
end



"""
    get_minimal_levelset(ctx, u, objective_function;
        x_resolution=nothing, y_resolution=nothing, min=minimum(u), max=maximum(u),
        n_candidates=100, bdata=BoundaryData())

Returns the levelset of u that achieves the minimal value of objective_function, considering levelsets
of values between min and max.
"""
function get_minimal_levelset(ctx, u, objective_function;
        x_resolution=nothing, y_resolution=nothing, min=minimum(u), max=maximum(u),
        n_candidates=100, bdata=BoundaryData())

    xs, ys, fs = get_function_values(ctx,u,x_resolution=x_resolution,y_resolution=y_resolution,bdata=bdata)

    currentmin = Inf
    result = nothing

    for cl in levels(contours(xs, ys, fs, range(min,stop=max,length=n_candidates)))
        curves = lines(cl)
        if length(curves) == 0  # this can happen e.g. for the levelset of max(u)
            continue
        end
        # TODO
        #if length(curves) != 1
        #    @warn "Currently only connected levelsets are allowed! Levelset: ", level(cl)
        #end
        value = objective_function(curves[1])
        if value < currentmin
            currentmin = value
            result = cl
        end
    end
    return result, currentmin
end

"""
    dynamic_cheeger_value(ctx, curve, flowmap; tolerance=0.0)

Calculate the dynamic cheeger value of a curve, given a flowmap T(x). The gridcontext ctx is needed
to close curves that intersect with the boundary, the tolerance paramter determines how close to
the boundary a curve can end to still be considered to have a point on it.
"""
function dynamic_cheeger_value(ctx, curve, flowmap; tolerance=0.0)
    LL, UR = ctx.spatialBounds
    volume_Ω = (UR[1] - LL[1]) * (UR[2] - LL[2])
    curve_closed_cw  = close_curve(ctx,curve,tolerance=tolerance,orientation=:clockwise)
    image_curve_cw   = apply2curve(flowmap,curve_closed_cw)
    curve_closed_acw = close_curve(ctx,curve,tolerance=tolerance,orientation=:anticlockwise)
    image_curve_acw  = apply2curve(flowmap,curve_closed_acw)
    combined_length  = min(get_length(curve_closed_cw)  + get_length(image_curve_cw),
                           get_length(curve_closed_acw) + get_length(image_curve_acw))
    volume_curve = get_euclidean_area(ctx,curve,tolerance=tolerance)
    return 0.5 * combined_length / min(volume_curve, (volume_Ω - volume_curve))
end


# simple finite differences for visualization only
# shouldn't be too hard to use the grid instead and do something like evaluate_function_from_node_or_cellvals
function gradient_from_values(xs,ys,fs)
    dx = diff(fs,dims=1)./diff(xs)
    dy = (diff(fs,dims=2)'./diff(ys))'
    #simply extend by constant
    dx = [dx ; dx[end,:]']
    dy = [dy dy[:,end]]
    return dx, dy
end

function get_gradient_field(ctx, u;
        x_resolution=nothing, y_resolution=nothing, bdata=BoundaryData())
    xs, ys, fs = get_function_values(ctx,u,x_resolution=x_resolution,y_resolution=y_resolution,bdata=bdata)
    dx, dy = gradient_from_values(xs,ys,fs)

    return xs, ys, dx, dy
end

"""
    get_levelset_evolution(ctx, u, u_dot;
        x_resolution=nothing, y_resolution=nothing, bdata=BoundaryData())

Returns a vectorfield describing how levelsets of u change according to the derivative u_dot.
Mainly to be used for plotting.
"""
function get_levelset_evolution(ctx, u, u_dot;
        x_resolution=nothing, y_resolution=nothing, bdata=BoundaryData())
    xs, ys, dx, dy = get_gradient_field(ctx, u, x_resolution=x_resolution, y_resolution=y_resolution,bdata=bdata)
    xs_dot, ys_dot, fs_dot = get_function_values(ctx, u_dot, x_resolution=x_resolution, y_resolution=y_resolution,bdata=bdata)

    @assert xs==xs_dot
    @assert ys==ys_dot

    norm_gradient = sqrt.(dx.*dx + dy.*dy)

    return xs, ys, -fs_dot.*dx./norm_gradient, -fs_dot.*dy./norm_gradient
end
