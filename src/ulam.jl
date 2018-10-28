#(c) 2018 Nathanael Schilling
#ulam.jl
#Implements Ulam's method within CoherentStructures.jl


function ulam(ctx::gridContext{2},f,nx,ny)
    result = spzeros(ctx.n,ctx.n)
    total_points = nx*ny
    for x in range(ctx.spatialBounds[1][1],stop=ctx.spatialBounds[2][1],length=nx)
        for y in range(ctx.spatialBounds[1][2],stop=ctx.spatialBounds[2][2],length=ny)
            m::Int = locatePoint(ctx,Tensors.Vec{2}((x,y)))[3]
            pointImage = f(Tensors.Vec{2}((x,y)))
            m2::Int = locatePoint(ctx,pointImage)[3]
            result[ctx.cell_to_dof[m],ctx.cell_to_dof[m2]] += 1/(nx*ny)
        end
    end
    return result
end
