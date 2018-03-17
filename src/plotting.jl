
#TODO: Can this be made more efficient?
function plot_u(ctx::gridContext,dof_vals::Vector{Float64},nx=50,ny=50;plotit=true,bdata=nothing,kwargs...)
    id = x -> x
    plot_u_eulerian(ctx,dof_vals,ctx.spatialBounds[1],ctx.spatialBounds[2],id,nx,ny,plotit=plotit,bdata=bdata;kwargs...)
end


function plot_ftle(odefun, p,tspan, LL, UR, nx=50,ny=50;δ=1e-9,tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),inplace=true,existing_plot=nothing,flip_y=true, inbounds_checker=always_true, kwargs...)
    x1 =  collect(linspace(LL[1] + 1e-8, UR[1] -1.e-8,nx))
    x2 =  collect(linspace(LL[2] + 1.e-8,UR[2]-1.e-8,ny))
    if flip_y
	x2 = reverse(x2)
    end
    #Initialize FTLE-field with NaNs
    FTLE = SharedArray{Float64,2}(ny,nx)
    for j in 1:ny
	for i in 1:nx
	    FTLE[j,i] = NaN
	end
    end
    nancounter = 0
    nonancounter = 0
    @sync @parallel for i in 1:nx
	for j in 1:ny
	    if inbounds_checker(x1[i],x2[j],p)
		try
		    FTLE[j,i] = 
		    1./(2*(tspan[end]-tspan[1]))*
		    log(maximum(abs.(eigvals(eigfact(dott(linearized_flow(odefun,Vec{2}([x1[i],x2[j]]),tspan,δ,tolerance=tolerance,p=p,solver=solver )[end]))))))
		    nonancounter += 1
		catch e
		    nancounter+=1
		    FTLE[j,i] = NaN
		end
	    end
	end
    end
    print("plot_ftle Ignored $nancounter NaN values ($nonancounter were good)")
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
                    plotit=true,euler_to_lagrange_points=nothing,
                    only_get_lagrange_points=false,postprocessor=nothing,
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
                    euler_to_lagrange_points[j,i] = inverse_flow_map(Vec{2}([x1[i],x2[j]]))
                catch e
                    euler_to_lagrange_points[j,i] = Vec{2}([NaN,NaN])
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
            if isnan((euler_to_lagrange_points[j,i])[1])
                z[j,i] = NaN
            else
                z[j,i] = evaluate_function(ctx,euler_to_lagrange_points[j,i],u_values,NaN)
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
