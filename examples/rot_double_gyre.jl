using CoherentStructures
using Arpack

import StaticArrays
const SA = StaticArrays

ctx = regularTriangularGrid((25,25),quadrature_order=5)
u = zeros(ctx.n)


x0 = @SVector Float64[0.5, 0.5]

@time begin
    #cgfun = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-8,tolerance= 1.e-3)
    cgfun = x -> mean(dott.(inv.(CoherentStructures.linearized_flow_vari(rot_double_gyreEqVari,SVector{2,Float64}(x),[0.0,1.0],tolerance=1e-10))))

    @time K = assembleStiffnessMatrix(ctx,cgfun)
    @time M = assembleMassMatrix(ctx,lumped=false)
    @time λ, v = eigs(-1*K,M,which=:SM,nev=6)
end

using Plots
plot_u(ctx,v[:,3]/maximum(abs.(v[:,3])),200,200,color=:rainbow,clim=(-1,1))


function g(x,y)
    return log(norm(cgfun(@SArray [x,y])))
end

Plots.heatmap(range(0,stop=1,length=200),range(0,stop=1,length=200),g)

g(0.5,0.1)


using Plots



@time assembleMassMatrix(ctx)

Profile.print()
using ProfileView
ProfileView.view()

plot_u(ctx,v[:,2])

function checkerboard(x)
    return ((-1)^(floor((x[1]*10)%10)))*((-1)^(floor((x[2]*10)%10)))
end

ctx2 = regularTriangularGrid((200,200))
u2 = nodal_interpolation(ctx2,checkerboard)
plot_u(ctx2,u2,100,100)
inverse_flow_map_t = (t,u0) -> flow(rot_double_gyre,u0,[t,0.0])[end]
inverse_flow_map_t(0.5,[0.5,0.5])
u(t) = u2
res = CoherentStructures.eulerian_video(ctx2,u,inverse_flow_map_t,
    0.0,1.0, 500,500,100, [0.0,0.0],[1.0,1.0],colorbar=false,title="Rotating Double Gyre")
Plots.mp4(res ,"/tmp/res.mp4")


#With non-adaptive TO-method:
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    inverse_flow = u0 -> flow(rot_double_gyre,u0,[1.0,0.0])[end]
    @time ALPHA = nonAdaptiveTO(ctx,inverse_flow)
    R = -1*(S + ALPHA'*S*ALPHA)
    R = 0.5(R + R')
    @time λ, v = eigs(R,M,which=:SM)
end

plot_u(ctx,v[:,2])
#With adaptive TO method
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx)
    forwards_flow = u0->flow(rot_double_gyre, u0,[0.0,1.0],tolerance=1e-3)[end]
    @time S2= adaptiveTO(ctx,forwards_flow)
    @time λ, v = eigs(-1*(S + S2),M,which=:SM)
end
#With L2-Galerkin calculation of ALPHA
ctx = regularP2TriangularGrid((20,20),quadrature_order=4)
using LinearMaps
begin
    @time S = assembleStiffnessMatrix(ctx)
    @time M = assembleMassMatrix(ctx,lumped=false)

    invFlowMap = u0->flow(rot_double_gyre,u0,[1.0,0.0],tolerance=1.e-4)[end]
    flowMap = u0->flow(rot_double_gyre,u0,[0.0,1.0],tolerance=1.e-8)[end]
    @time preALPHAS= CoherentStructures.L2GalerkinTOFromInverse(ctx,invFlowMap)
    #@time preALPHAS= CoherentStructures.L2GalerkinTO(ctx,flowMap)
    function mulby(x)
        return -0.5(preALPHAS'*(M'\(S*(M\(preALPHAS*x)))) + S*x)
    end
    L = LinearMap(mulby,size(S)[1],issymmetric=true)
    @time λ, v = eigs(L,M,which=:SR,nev=6,maxiter=100000000)
end
index= 1
title = "\\\lambda = $(λ[index])"
plot_u(ctx,real.(v[:,index]),200,200,color=:rainbow,title=title)

using LinearMaps
inv(LinearMap(M)
