
#TODO: Can this be made more efficient?
function plot_u(ctx::gridContext,dof_values::Vector{Float64},nx=50,ny=50;plotit=true,kwargs...)
    x1 = Float64[]
    x2 = Float64[]
    LL = ctx.spatialBounds[1]
    UR = ctx.spatialBounds[2]
    values = Float64[]
    const u_values =  dof2U(ctx,dof_values)
    x1 =  linspace(LL[1] + 1e-8, UR[1] -1.e-8,ny)
    x2 =  linspace(LL[2] + 1.e-8,UR[2]-1.e-8,nx)
    myf(x,y) =  evaluate_function(ctx, Vec{2}([x,y]),u_values)


    Plots.plot(x1,x2,values;t=:contourf)#,colormap=GR.COLORMAP_JET)
    result =  Plots.contour(x1,x2,myf,fill=true;kwargs...)#,colormap=GR.COLORMAP_JET)
    if plotit
        Plots.plot(result)
    end
    return result
end


function plot_u_eulerian(ctx::gridContext,dof_values::Vector{Float64},LL::AbstractVector{Float64}, UR::AbstractVector{Float64},inverse_flow_map::Function,nx=50,ny=60;plotit=true,kwargs...)
    x1 = Float64[]
    x2 = Float64[]
    values = Float64[]
    const u_values =  dof2U(ctx,dof_values)
    x1 =  linspace(LL[1] + 1e-8, UR[1] -1.e-8,ny)
    x2 =  linspace(LL[2] + 1.e-8,UR[2]-1.e-8,nx)
    function myf(x,y)
        try
            lagrangian_point = inverse_flow_map(Vec{2}([x,y]))
            return evaluate_function(ctx,lagrangian_point ,u_values,NaN)
        catch y
            return NaN
        end
        return
    end
    result =  Plots.heatmap(x1,x2,myf,fill=true;kwargs...)#,colormap=GR.COLORMAP_JET)
    if plotit
        Plots.plot(result)
    end
    return result
end

function plot_spectrum(λ)
    Plots.scatter(real.(λ),imag.(λ))
end
