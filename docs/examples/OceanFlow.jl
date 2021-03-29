#OceanFlow.jl - based on code from Daniel Karrasch

using Distributed, BenchmarkTools, Test, JLD2
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures
JLD2.@load "docs/examples/Ocean_geostrophic_velocity.jld2" Lon Lat Time UT VT

t_initial = minimum(Time)
t_final = t_initial + 90

LL = (-4.0, -34.0)
UR = (6.0, -28.0)
const p = interpolateVF(Lon, Lat, Time, UT, VT)
ctx, _ = regularP2TriangularGrid((100, 60), LL, UR, quadrature_order=2)

# ctx.mass_weights = [cos(deg2rad(x[2])) for x in ctx.quadrature_points]
const times = [t_initial,t_final]

####### for comparison, finally only one of the following lines should remain #######
# seems like mean(pullback_diffusion_tensor) is no worse than mean_diff_tensor, maybe even faster
# first the not-in-place SVector versions
mdt(x) = mean_diff_tensor(interp_rhs, x, times, 1.e-8, tolerance=1.e-5, p=p)
mdt1 = x -> mean_diff_tensor(interp_rhs, x, times, 1.e-8, tolerance=1.e-5, p=p)
mdt3(x) = let P=p
    mean_diff_tensor(interp_rhs, x, times, 1.e-8; tolerance=1.e-5, p=P)
end
@btime As = map(mdt, $ctx.quadrature_points)
@btime As = map($mdt1, $ctx.quadrature_points)
@btime As = map(mdt3, $ctx.quadrature_points)
@inferred mdt(ctx.quadrature_points[1])
@inferred mdt1(ctx.quadrature_points[1])
@inferred mdt3(ctx.quadrature_points[1])
@inferred mean_diff_tensor(interp_rhs, ctx.quadrature_points[1], times, 1.e-8, tolerance=1.e-5, p=p)
# now the mutating
@everywhere mdt!(x) = mean_diff_tensor(interp_rhs!, x, times, 1.e-8, tolerance=1.e-5, p=p)
@time As = [mdt!(x) for x in ctx.quadrature_points]
@time As = pmap(mdt!, ctx.quadrature_points)
@time As = parallel_tensor(mdt!, ctx.quadrature_points)
######### end of comparison

mean_As = As
function mean_Afun(x,index,p)
    return p[1][index]
end



#With CG-Method
begin
    q = [mean_As]
    bdata = getHomDBCS(ctx)
    @time K2 = assembleStiffnessMatrix(ctx, mean_Afun, q, bdata=bdata)
    @time M2 = assembleMassMatrix(ctx, bdata=bdata)
    @time λ2, v2 = eigs(K2, M2, which=:SM, nev=6)
end
plot_real_spectrum(λ2)
plot_u(ctx, v2[:,6], 200, 200, bdata=bdata, color=:rainbow)
length(bdata.dbc_dofs)
using Clustering
numclusters = 5
res = kmeans(permutedims(v2[:,1:numclusters]), numclusters+1)
u = kmeansresult2LCS(res)
plot_u(ctx, u[:,3], 200, 200, color=:viridis)

##Make a video of the LCS being advected

#Inverse-flow map at time t
inverse_flow_map_t = (t,u0) -> flow(interp_rhs!, u0,
        [t, t_initial], p=(UI, VI), tolerance=1e-4)[end]
#Function u(t), here it is just constant
current_u = t -> u[:,2]
#Where to plot the video
LL_big = (-10, -40.0)
UR_big = (6, -25.0)
#Make the video
res = eulerian_video(ctx,current_u, LL_big, UR_big,
        100, 100, t_initial, t_final, #nx = 100, ny=100
        30, inverse_flow_map_t)#nt = 20
#Save it
Plots.mp4(res,"/tmp/output.mp4")

using Arpack
using Plots

ctx = regularDelaunayGrid((100, 60), LL, UR, quadrature_order=2)
#With adaptive TO method
begin
    ocean_flow_map = u0 -> flow(interp_rhs, u0, [t_initial, t_final], p=(UI, VI), tolerance=1e-5)[end]
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time S2= adaptiveTO(ctx, ocean_flow_map)
    @time λ, v = eigs(S + S2, M, which=:SM)
end
plot_u(ctx, v[:,1])
Plots.gr()
I, J, V = findnz(S)
Plots.scatter(I, J, markersize=0.1)
V .= 1 .- V
S3 = sparse(I, J, V)
Plots.heatmap(S3,clim=(0.0,1.0))
Plots.pdf("/tmp/heatmap.pdf")

Plots.imshow(S2)
Plots.heatmap(S2)
plot_real_spectrum(λ)
plot_u(ctx,v[:,2],200,200)


plot_spectrum(λ)
index = sortperm(real.(λ))[end-5]
title = "Eigenvector with eigenvalue $(λ[index])"
plot_u(ctx,real.(v[:,index]),title=title,color=:rainbow)

plot_spectrum(λ2)
index = sortperm(real.(λ2))[end-2]
title = "Eigenvector with eigenvalue $(λ2[index])"
plot_u(ctx,real.(v2[:,index]),title=title,color=:rainbow)
