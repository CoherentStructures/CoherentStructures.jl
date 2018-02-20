#OceanFlow.jl - based on code from Daniel Karrasch
import JLD
using juFEMDL
using Tensors

vars = JLD.load("examples/Ocean_geostrophic_velocity.jld")
Lon = vars["Lon"]
Lat = vars["Lat"]
UT = vars["UT"]
time = vars["time"]
VT = vars["VT"]

t_initial = minimum(time)
t_final = t_initial + 90

LL = [-4.0,-34.0]
UR = [6.0,-28.0]
UI, VI = interpolateVF(Lon,Lat,UT,time,VT)
p=(UI,VI)


ctx = regularP2TriangularGrid((100,100),LL,UR)
ctx.mass_weights = [cos(deg2rad(x[2])) for x in ctx.quadrature_points]
Id = one(SymmetricTensor{2,2})
times = linspace(t_initial,t_final,450)
As = [
        #invCGTensor(interp_rhs, x,[t_initial,t_final], 1.e-8,tolerance=1.e-3,p=p)
        pullback_diffusion_tensor(interp_rhs, x,times, 1.e-8,Id,tolerance=1.e-4,p=p)
        for x in ctx.quadrature_points
        ]
mean_As = mean.(As)
gc()

#With CG-Method
@time M2 = assembleMassMatrix(ctx)
ϵ=1.e-3
dt = times[2]-times[1]
function myAfun(x,index,q)
        return q[1][index][q[2]]
end
function myMeanAFun(x,index,q)
        return q[1][index]
end
function f(x)
        if x[1]*x[1] + x[2]*x[2] <= 1
                return 1.0
        else return 0.0
        end
end

unode = [f(n.x - (LL + UR)/2.0) for n in ctx.grid.nodes]
u = zeros(unode)
for i in 1:length(u)
        u[i] = unode[ctx.dof_to_node[i]]
end
plot_u(ctx,u,color=:rainbow,200,200,aspect_ratio=1.0)
using Plots
anim = @animate for (tindex,t) in enumerate(times)
        q = [mean_As, tindex]
        K2= assembleStiffnessMatrix(ctx,myMeanAFun,q)
        u = (M2 - ϵ*dt*K2)\(M2*u)
        plot_u(ctx,u,200,200,title="t = $t",color=:rainbow,aspect_ratio=1)
        gc()
        print("Timestep $tindex")
end every 10
As = 0
gc()
mp4(anim,"/tmp/out_mean.mp4")
@time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)

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
plot_u(ctx,real.(v[:,index]),title=title)

plot_spectrum(λ2)
index = sortperm(real.(λ2))[end-2]
title = "Eigenvector with eigenvalue $(λ2[index])"
plot_u(ctx,real.(v2[:,index]),title=title,color=:rainbow)
