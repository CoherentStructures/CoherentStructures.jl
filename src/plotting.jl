
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


function plot_u_eulerian(ctx::gridContext,dof_values::Vector{Float64},LL::AbstractVector{Float64}, UR::AbstractVector{Float64},inverse_flow_map::Function,nx=50,ny=60;plotit=true,euler_to_lagrange_points=nothing,only_get_lagrange_points=false,postprocessor=nothing,kwargs...)
    x1 = Float64[]
    x2 = Float64[]
    values = Float64[]
    const u_values =  dof2U(ctx,dof_values)
    x1 =  linspace(LL[1] , UR[1] ,ny)
    x2 =  linspace(LL[2] ,UR[2] ,nx)
    if euler_to_lagrange_points == nothing
        euler_to_lagrange_points = [zero(Vec{2}) for x in x1, y in x2]
        for i in 1:ny
            for j in 1:nx
                try
                    euler_to_lagrange_points[i,j] = inverse_flow_map(Vec{2}([x1[i],x2[j]]))
                catch e
                    euler_to_lagrange_points[i,j] = Vec{2}([NaN,NaN])
                end
            end
        end
    end
    if only_get_lagrange_points
        return euler_to_lagrange_points
    end
    z = zeros(nx,ny)
    for i in 1:ny
        for j in 1:nx
            if isnan((euler_to_lagrange_points[i,j])[1])
                z[j,i] = NaN
            else
                z[j,i] = evaluate_function(ctx,euler_to_lagrange_points[i,j],u_values,NaN)
            end
        end
    end
    if postprocessor != nothing
        postprocessor(z)
    end

    result =  Plots.heatmap(x1,x2,z,fill=true;kwargs...)#,colormap=GR.COLORMAP_JET)
    if plotit
        Plots.plot(result)
    end
    return result
end

function plot_spectrum(λ)
    Plots.scatter(real.(λ),imag.(λ))
end
