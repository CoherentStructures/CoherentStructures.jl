#(c) 2018 Nathanael Schilling & Daniel Karrasch
#ulam.jl
#Implements Ulam's method within CoherentStructures.jl

function ulam(ctx::gridContext{2}, f, nx, ny)
    n = ctx.n
    npoints = nx * ny
    area = prod(ctx.spatialBounds[2][:] - ctx.spatialBounds[1][:])
    val = area / npoints
    Is = Vector{Int}(undef, npoints)
    Js = Vector{Int}(undef, npoints)
    xs = range(ctx.spatialBounds[1][1], stop=ctx.spatialBounds[2][1], length=nx)
    ys = range(ctx.spatialBounds[1][2], stop=ctx.spatialBounds[2][2], length=ny)
    idx = 0
    for x in xs
        for y in ys
            idx += 1
            m::Int = locatePoint(ctx, Tensors.Vec{2}((x, y)))[3]
            pointImage = f(Tensors.Vec{2}((x, y))) # TODO: why Vec{2}?
            # this assumes that pointImage is contained in ctx
            m2::Int = locatePoint(ctx, pointImage)[3]
            Is[idx] = ctx.cell_to_dof[m]
            Js[idx] = ctx.cell_to_dof[m2]
        end
    end
    return sparse(Is, Js, val, n, n)
end
