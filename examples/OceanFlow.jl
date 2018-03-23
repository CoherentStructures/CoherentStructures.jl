#OceanFlow.jl - based on code from Daniel Karrasch

using juFEMDL
using Tensors
using Clustering #For kmeans function

JLD2.@load "examples/Ocean_geostrophic_velocity.jld2" Lon Lat Time UT VT

t_initial = minimum(Time)
t_final = t_initial + 90

LL = [-4.0,-34.0]
UR = [6.0,-28.0]
UI, VI = interpolateVF(Lon,Lat,Time,permutedims(UT,[2,1,3]),permutedims(VT,[2,1,3]))
p=(UI,VI)
ctx = regularDelaunayGrid((100,60),LL,UR,quadrature_order=2)


ctx.mass_weights = [cos(deg2rad(x[2])) for x in ctx.quadrature_points]
times = [t_initial,t_final]
@time As = [
        invCGTensor(interp_rhs, x,times, 1.e-8,tolerance=1.e-5,p=p)
        #pullback_diffusion_tensor(interp_rhs, x,times, 1.e-8,Id,tolerance=1.e-4,p=p)
        for x in ctx.quadrature_points
        ] # TODO: replace by pmap call
#mean_As = mean.(As)
mean_As = As
function mean_Afun(x,index,p)
        return p[1][index]
end

#With CG-Method
begin
        q = [mean_As]
        #bdata = juFEMDL.getHomDBCS(ctx)
        bdata = juFEMDL.boundaryData()
        @time K2 = assembleStiffnessMatrix(ctx,mean_Afun,q,bdata=bdata)
        @time M2 = assembleMassMatrix(ctx,bdata=bdata)
        @time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)
end
plot_real_spectrum(λ2)
plot_u(ctx,v2[:,5],200,200,bdata=bdata)

using Clustering
numclusters = 5
res = kmeans(v2[:,1:numclusters]',numclusters+1)
u = kmeansresult2LCS(res)
plot_u(ctx,u[:,3],200,200,color=:viridis)

##Make a video of the LCS being advected

#Inverse-flow map at time t
inverse_flow_map_t = (t,u0) -> flow(interp_rhs,u0,
        [t,t_initial],p=(UI,VI),tolerance=1e-4)[end]
#Function u(t), here it is just constant
current_u = t -> u[:,2]
#Where to plot the video
LL_big = [-10,-40.0]
UR_big = [6,-25.0]
#Make the video
res = juFEMDL.eulerian_video(ctx,current_u,LL_big,UR_big,
        100,100,t_initial, t_final, #nx = 100, ny=100
        30,inverse_flow_map_t)#nt = 20
#Save it
Plots.mp4(res,"/tmp/output.mp4")

#With adaptive TO method
begin
        ocean_flow_map = u0 -> flow(interp_rhs,u0, [t_initial,t_final],p=(UI,VI))[end]
        @time S = assembleStiffnessMatrix(ctx)
        @time M = assembleMassMatrix(ctx)
        @time S2= adaptiveTO(ctx,ocean_flow_map)
        @time λ, v = eigs(S + S2,M,which=:SM)
end

plot_u(ctx,v[:,3],200,200,bdata=bdata)


plot_spectrum(λ)
index = sortperm(real.(λ))[end-5]
title = "Eigenvector with eigenvalue $(λ[index])"
plot_u(ctx,real.(v[:,index]),title=title,color=:rainbow)

plot_spectrum(λ2)
index = sortperm(real.(λ2))[end-2]
title = "Eigenvector with eigenvalue $(λ2[index])"
plot_u(ctx,real.(v2[:,index]),title=title,color=:rainbow)
