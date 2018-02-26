#OceanFlow.jl - based on code from Daniel Karrasch
import JLD
using juFEMDL
using Tensors

vars = JLD.load("examples/Ocean_geostrophic_velocity.jld")
Lon = vars["Lon"]
Lat = vars["Lat"]
UT = vars["UT"]
VT = vars["VT"]
time = vars["time"]

t_initial = minimum(time)
t_final = t_initial + 90

LL = [-4.0,-34.0]
UR = [6.0,-28.0]
UI, VI = interpolateVF(Lon,Lat,UT,time,VT)
p = (UI,VI)

ctx = regularP2QuadrilateralGrid((500,500),LL,UR)


f(x) = (norm(x - (LL + UR)/2) <= 1.0) ? 1.0 : 0.0
u0 = nodal_interpolation(ctx,f)
LL_big = LL - [8,8]
UR_big = UR + [8,8]
inverse_flow_map = u0 -> flow(interp_rhs,u0, [t_final,t_initial],p=(UI,VI))[end]




#With CG-Method
begin
    cgfun = (x -> invCGTensor(interp_rhs, x,[t_initial,t_final], 1.e-8,tolerance=1.e-5,p=p))
    @time K2 = assembleStiffnessMatrix(ctx,cgfun)
    @time M2 = assembleMassMatrix(ctx)
    @time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)
end

plot_u(ctx,v2[:,6])
plot_u_eulerian(ctx,v2[:,6],LL_big,UR_big,inverse_flow_map,5000,5000,color=:rainbow)
lamreal = real(λ2[6])
#Plots.title!("\\lambda=$lamreal")
using LaTeXStrings
Plots.title!("\\lambda=$lamreal")
Plots.pdf("/tmp/outputhuge.pdf")

#With adaptive TO method
begin
    ocean_flow_map = u0 -> flow(interp_rhs,u0, [t_initial,t_final],p=(UI,VI))[end]
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time S2= adaptiveTO(ctx,ocean_flow_map)
    @time λ, v = eigs(S + S2,M,which=:SM)
end


plot_spectrum(λ)
index = sortperm(real.(λ))[end-5]
title = "Eigenvector with eigenvalue $(λ[index])"
plot_u(ctx,real.(v[:,index]),title=title,color=:rainbow)

plot_spectrum(λ2)
index = sortperm(real.(λ2))[end-1]
title = "Eigenvector with eigenvalue $(λ2[index])"
plot_u(ctx,real.(v2[:,index]),title=title,color=:rainbow)
