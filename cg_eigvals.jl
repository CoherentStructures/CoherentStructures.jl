#Based on static_Laplace_eigvs.jl
import GR
include("velocityFields.jl")
include("TO.jl")
include("GridFunctions.jl")
include("plotting.jl")
include("PullbackTensors.jl")
include("FEMassembly.jl")

ctx = regularDelaunayGrid((50,50))


Id = one(Tensor{2,2})
cgfun = (x -> invCGTensor(x,[0.0,1.0], 1.e-8,rot_double_gyre2,1.e-3))

function myIdentity(x::Vec{2})
    return Id
end

#With CG-Method
begin
    @time S = assembleStiffnessMatrix(ctx,myIdentity)
    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx)
    @time λ, v = eigs(S+K,M,which=:SM)
end

#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx,myIdentity)
    @time ALPHA = getAlphaMatrix(ctx,u0->flow2D(rot_double_gyre2,u0,[0.0,-1.0]))
    @time λ, v = eigs(S + ALPHA'*S*ALPHA,M,which=:SM)
end

#@time apply!(K, dbc)

index = sortperm(real.(λ))[end-1]
GR.title("Eigenvector with eigenvalue $(λ[index])")
plot_u(ctx,real.(v[:,index]),25,25)
plot_spectrum(λ)
#GR.contourf(reshape(real(dof2U(dh,v[:,index])),m,m),colormap=GR.COLORMAP_JET)
#savefig("output.png")
