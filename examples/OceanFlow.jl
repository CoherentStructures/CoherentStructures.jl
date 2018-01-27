#OceanFlow.jl - based on code from Daniel Karrasch
begin
    import JLD
    include("velocityFields.jl")
    include("GridFunctions.jl")
    include("plotting.jl")
    include("FEMassembly.jl")
    include("PullbackTensors.jl")
    include("TO.jl")
    include("numericalExperiments.jl")
end



vars = JLD.load("Ocean_geostrophic_velocity.jld")
Lon = vars["Lon"]
Lat = vars["Lat"]
UT = vars["UT"]
time = vars["time"]
VT = vars["VT"]

t_initial = minimum(time)
t_final = t_initial + 90

LL = Vec{2}([-4.0,-34.0])
UR = Vec{2}([6.0,-28.0])
#ctx = regularP2QuadrilateralGrid((50,50),LL,UR)
ctx = regularP2DelaunayGrid((250,150),LL,UR)
#cgfun = (x -> invCGTensor(x,collect(linspace(t_initial,t_final,4))], 1.e-8,ocean_flow_map,1.e-3))
cgfun = (x -> invCGTensor(x,[t_initial,t_final], 1.e-8,ocean_vector_field,1.e-3))
UI, VI = interpolateOceanFlow(Lon,Lat,UT,time,VT)
ocean_vector_field = ( (t,u,du) ->  oceanVF(t,u,du,UI,VI))
ocean_flow_map = u0 -> flow2D(ocean_vector_field,u0, [t_initial,t_final],1.e-5)

u = [LL[1],LL[2]]
@time UI[-5.0,-35.0,t_initial]
@time ocean_flow_map(LL)

#With CG-Method
begin
    @time K2 = assembleStiffnessMatrix(ctx,cgfun)
    @time M2 = assembleMassMatrix(ctx)
    @time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)
end
@time λ2, v2 = eigs(K2,M2,which=:SM,nev=12)

begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    @time S2= adaptiveTO(ctx,u0->flow2D(ocean_vector_field,u0,[t_initial,t_final]))
    @time λ, v = eigs(S + S2,M,which=:SM)
end
@time λ, v = eigs(S + S2,M,which=:SM,nev=12)


plot_spectrum(λ)
index = sortperm(real.(λ))[end]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),100,100,LL,UR)

plot_spectrum(λ2)
index = sortperm(real.(λ2))[end]
GR.title("Eigenvector with eigenvalue $(λ2[index])")
plot_u(ctx,real.(v2[:,index]),100,100,LL,UR)
