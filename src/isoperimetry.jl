
"""
    getLength(curve, tensorfield)

Calculates an approximation of the length of the curve `curve` as measured by the metric `tensorfield`
This is by calculating the length of each segment of the curve as measured by the metric at the starting point.
"""
function getLength(curve, tensorfield)
    n = length(curve)
    result = 0
    for j in 1:(n-1)
        curpoint = curve[j]
        nextpoint = curve[j+1]
        approxTangentVector = nextpoint - curpoint
        result += sqrt(approxTangentVector ⋅ (tensorfield(curpoint[1], curpoint[2]) * approxTangentVector))
    end
    return result
end

"""
    getEuclideanArea(curve)

Calculates the area enclosed by the curve `curve`.
Assumes that the last point is equal to the first one.
"""
function getEuclideanArea(curve)
    centrepoint = mean(curve)
    n = length(curve)
    result = 0.0
    for j in 1:(n-1)
        p1 = curve[j]
        p2 = curve[j+1]

        v2 = p2 - centrepoint
        v1 = p1 - centrepoint
        area = 0.5*det([v1[1] v1[2]; v2[1] v2[2]])
        result +=area
    end
    return result
end

function get_best_levelset(ctx, u, ncontours, nx, ny, tensorfield, scaling=2.0; bdata=bdata)
    xs = range(ctx.spatialBounds[1][1], stop=ctx.spatialBounds[2][1], length=nx)
    ys = range(ctx.spatialBounds[1][2], stop=ctx.spatialBounds[2][2], length=ny)

    u_dofvals = undoBCS(ctx, u, bdata)
    u_nodevals = u_dofvals[ctx.node_to_dof]

    fs = [evaluate_function_from_node_or_cellvals(ctx, u_nodevals, Vec{2}((x,y)))
        for x in xs, y in ys]
    cl = contours(xs, ys, fs, ncontours)

    currentmin = Inf
    result = nothing

    for cl in lines(contours(xs, ys, fs))
        area = getEuclideanArea(cl)
        len = getLength(cl, tensorfield)
        isoperim = len^scaling / area
        if isoperim < currentmin
            currentmin = isoperim
            result = cl
        end
    end
    return result
end
