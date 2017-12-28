#OceanFlow.jl - based on code from Daniel Karrasch
import JLD
include("velocityFields.jl")
include("GridFunctions.jl")
include("plotting.jl")
include("FEMassembly.jl")
include("PullbackTensors.jl")

vars = JLD.load("Ocean_geostrophic_velocity.jld")
Lon = vars["Lon"]
Lat = vars["Lat"]
UT = vars["UT"]
time = vars["time"]
VT = vars["VT"]

UI, VI = interpolateOceanFlow(Lon,Lat,UT,time,VT)
ocean_flow_map = ( (t,u,du) ->  oceanVF(t,u,du,UI,VI))

LL = Vec{2}([-4.0,-34.0])
UR = Vec{2}([6.0,-28.0])

t_initial = minimum(time)
t_final = maximum(time)
#Make the Grid smaller..
#Pick a smaller time_interval
t_final = t_initial + 90

ctx = regularP2DelaunayGrid((25,30),LL,UR)
#cgfun = (x -> invCGTensor(x,collect(linspace(t_initial,t_final,4))], 1.e-8,ocean_flow_map,1.e-3))
cgfun = (x -> invCGTensor(x,[t_initial,t_final], 1.e-8,ocean_flow_map,1.e-3))

#With CG-Method
begin
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(K,M,which=:SM)
end


plot_spectrum(λ)
index = sortperm(real.(λ))[end-5]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),25,25,LL,UR)
