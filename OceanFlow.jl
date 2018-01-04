#OceanFlow.jl - based on code from Daniel Karrasch
begin
    import JLD
    include("velocityFields.jl")
    include("GridFunctions.jl")
    include("plotting.jl")
    include("FEMassembly.jl")
    include("PullbackTensors.jl")
    include("TO.jl")
end

vars = JLD.load("Ocean_geostrophic_velocity.jld")
Lon = vars["Lon"]
Lat = vars["Lat"]
UT = vars["UT"]
time = vars["time"]
VT = vars["VT"]

UI, VI = interpolateOceanFlow(Lon,Lat,UT,time,VT)
ocean_vector_field = ( (t,u,du) ->  oceanVF(t,u,du,UI,VI))

#The computational domain
LL = Vec{2}([-4.0,-34.0])
UR = Vec{2}([6.0,-28.0])

t_initial = minimum(time)
t_final = maximum(time)
#Make the Grid smaller..
#Pick a smaller time_interval
t_final = t_initial + 90

ctx = regularP2DelaunayGrid((50,50),LL,UR)
#cgfun = (x -> invCGTensor(x,collect(linspace(t_initial,t_final,4))], 1.e-8,ocean_flow_map,1.e-3))
cgfun = (x -> invCGTensor(x,[t_initial,t_final], 1.e-8,ocean_vector_field,1.e-3))

ocean_flow_map = u0 -> flow2D(ocean_vector_field,u0, [t_initial,t_final],1.e-5)
t = t_initial + 1
du = [0.0,0.0]
u = [LL[1],LL[2]]
@time UI[-5.0,-35.0,t_initial]
@time ocean_flow_map(LL)

#With CG-Method
begin
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(K,M,which=:SM,nev=12)
end


plot_spectrum(λ)
index = sortperm(real.(λ))[end-4]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),50,50,LL,UR)
