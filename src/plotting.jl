
#TODO: Can this be made more efficient?
function plot_u(ctx::gridContext,dof_values::Vector{Float64},nx=50,ny=50;color=:viridis,title="Plot")
    x1 = Float64[]
    x2 = Float64[]
    LL = ctx.spatialBounds[1]
    UR = ctx.spatialBounds[2]
    values = Float64[]
    const u_values =  dof2U(ctx,dof_values)
    x1 =  linspace(LL[2] + 1e-8, UR[2] -1.e-8,ny)
    x2 =  linspace(LL[1] + 1.e-8,UR[1]-1.e-8,nx)
    myf(x,y) =  evaluate_function(ctx, Vec{2}([x,y]),u_values)
    #Plots.plot(x1,x2,values;t=:contourf)#,colormap=GR.COLORMAP_JET)
    p1 = Plots.contour(x1,x2,myf,fill=true,color=color,title=title)#,colormap=GR.COLORMAP_JET)
    Plots.plot(p1)
end


function plot_spectrum(λ)
    Plots.plot(real.(λ),imag.(λ),"x")
end
