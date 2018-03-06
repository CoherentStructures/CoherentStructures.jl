
#TODO: Can this be made more efficient?
function plot_u(ctx::gridContext,dof_vals::Vector{Float64},nx=50,ny=50;plotit=true,kwargs...)
    id = x -> x
    plot_u_eulerian(ctx,dof_vals,ctx.spatialBounds[1],ctx.spatialBounds[2],id,nx,ny,plotit=plotit;kwargs...)
end

function plot_ftle(odefun, p,tspan, LL, UR, nx=50,ny=50;δ=1e-9,tolerance=1e-4,solver=OrdinaryDiffEq.BS5(), kwargs...)
    x1 =  linspace(LL[1] + 1e-8, UR[1] -1.e-8,nx)
    x2 =  linspace(LL[2] + 1.e-8,UR[2]-1.e-8,ny)
    DF = [linearized_flow(odefun,Vec{2}([x,y]),tspan,δ,tolerance=tolerance,p=p,solver=solver )[end] for y in x2, x in x1]
    arrayabs(x) = abs.(x)
    FTLE = 1./(2*(tspan[2]-tspan[1]))*log.(maximum.(arrayabs.(eigvals.(eigfact.(dott.(DF))))))
    #trDF = log.(abs.(trace.(DF)))
    return Plots.heatmap(x1,x2,FTLE; kwargs...)
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
                    kwargs...)

    if (ctx.n != length(dof_vals))
        dbcs = getHomDBCS(ctx)
        if length(dbcs.prescribed_dofs) + length(dof_vals) != ctx.n
            error("Input u has wrong length")
        end
        dof_values = upsample2DBCS(ctx,dof_vals,dbcs)
    else
        dof_values = dof_vals
    end
    x1 = Float64[]
    x2 = Float64[]
    values = Float64[]
    const u_values =  dof2U(ctx,dof_values)
    x1 =  linspace(LL[1] , UR[1] ,nx)
    x2 =  linspace(LL[2] ,UR[2] ,ny)
    if euler_to_lagrange_points == nothing
        euler_to_lagrange_points = [zero(Vec{2}) for y in x2, x in x1]
        for i in 1:nx
            for j in 1:ny
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
    for i in 1:nx
        for j in 1:ny
            if isnan((euler_to_lagrange_points[i,j])[1])
                z[i,j] = NaN
            else
                z[i,j] = evaluate_function(ctx,euler_to_lagrange_points[i,j],u_values,NaN)
            end
        end
    end
    if postprocessor != nothing
       z =  postprocessor(z)
    end

    result =  Plots.heatmap(x1,x2,z,fill=true,aspect_ratio=1;kwargs...)#,colormap=GR.COLORMAP_JET)
    if plotit
        Plots.plot(result)
    end
    return result
end

function plot_spectrum(λ)
    Plots.scatter(real.(λ),imag.(λ))
end

function plot_real_spectrum(λ)
    Plots.scatter(1:length(λ),real.(λ))
end
