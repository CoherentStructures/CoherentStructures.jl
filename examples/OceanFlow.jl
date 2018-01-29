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

LL = Vec{2}([-4.0,-34.0])
UR = Vec{2}([6.0,-28.0])
UI, VI = interpolateVF(Lon,Lat,UT,time,VT)
p = (UI,VI)
#ctx = regularP2QuadrilateralGrid((50,50),LL,UR)
ctx = regularP2DelaunayGrid((25,25),LL,UR)
cgfun = (x -> invCGTensor(interp_rhs, x,[t_initial,t_final], 1.e-8,tolerance=1.e-3,p=p))
ocean_flow_map = u0 -> flow(interp_rhs,u0, [t_initial,t_final],p=p)[end]

#With CG-Method
begin
    @time K2 = assembleStiffnessMatrix(ctx,cgfun)
    @time M2 = assembleMassMatrix(ctx)
    @time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)
end

#With adaptive TO method
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time S2= adaptiveTO(ctx,ocean_flow_map)
    @time λ, v = eigs(S + S2,M,which=:SM)
end


plot_spectrum(λ)
index = sortperm(real.(λ))[end-1]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),100,100,LL,UR)

plot_spectrum(λ2)
index = sortperm(real.(λ2))[end-1]
GR.title("Eigenvector with eigenvalue $(λ2[index])")
plot_u(ctx,real.(v2[:,index]),100,100,LL,UR)
