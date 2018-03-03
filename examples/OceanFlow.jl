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
p=(UI,VI)
ctx = regularTriangularGrid((200,200),LL,UR)
ctx.mass_weights = [cos(deg2rad(x[2])) for x in ctx.quadrature_points]
Id = one(SymmetricTensor{2,2})
times = linspace(t_initial,t_final,450)
As = [
        invCGTensor(interp_rhs, x,[t_initial,t_final], 1.e-8,tolerance=1.e-3,p=p)
        #pullback_diffusion_tensor(interp_rhs, x,times, 1.e-8,Id,tolerance=1.e-4,p=p)
        for x in ctx.quadrature_points
        ]
#mean_As = mean.(As)
mean_As = As
function mean_Afun(x,index,p)
        return p[1][index]
end

#With CG-Method
begin
    @time K2 = assembleStiffnessMatrix(ctx,mean_Afun)
    @time M2 = assembleMassMatrix(ctx)
    @time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)
end

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
index = sortperm(real.(λ2))[end-2]
title = "Eigenvector with eigenvalue $(λ2[index])"
plot_u(ctx,real.(v2[:,index]),title=title,color=:rainbow)



#Solving advection/diffusion equation with implicit Euler method
anim = @animate for (tindex,t) in enumerate(times)
        q = [As, tindex]
        u = ADimplicitEulerStep(ctx,u,ϵ*dt,mean_Afun,q) #TODO: Fix this...
        plot_u(ctx,u,200,200,title="t = $t",color=:rainbow,aspect_ratio=1)
        print("Timestep $tindex")
        gc()
end every 10
mp4(anim,"/tmp/out_mean.mp4")
