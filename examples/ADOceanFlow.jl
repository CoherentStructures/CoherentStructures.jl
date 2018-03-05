using juFEMDL
using Tensors
using Plots # For @animate


tC = makeOceanFlowTestCase()
times = linspace(tC.t_initial,tC.t_final,450)
ctx = regularTriangularGrid((50,50),tC.LL,tC.UR)

#Initial values
u = nodal_interpolation(ctx, x -> norm(x - (tC.LL + tC.UR)/2) > 1 ? 1.0 : 0.0)
plot_u(ctx,u)

#Calculate mean pullback tensor
mean_As = [
        invCGTensor(interp_rhs, x,times, 1.e-8,tolerance=1.e-3,p=tC.p)
        #pullback_diffusion_tensor(interp_rhs, x,times, 1.e-8,Id,tolerance=1.e-4,p=p)
        for x in ctx.quadrature_points
        ]

function mean_Afun(x,index,p)
        return p[1][index]
end

#Solving advection/diffusion equation with implicit Euler method
dt = times[2] - times[1]
ϵ = 1.e-5
anim = @animate for (tindex,t) in enumerate(times)
        q = [mean_As, tindex]
        u = ADimplicitEulerStep(ctx,u,ϵ*dt,mean_Afun,q) #TODO: Fix this...
        plot_u(ctx,u,200,200,title="t = $t",color=:rainbow,aspect_ratio=1)
        print("Timestep $tindex")
        gc()
end every 10
mp4(anim,"/tmp/out_mean.mp4")

#Fokker-Planck in Lagrangian coordinates in one fell swoop
using juFEMDL

tC = makeOceanFlowTestCase()
ctx = regularTriangularGrid((50,50),tC.LL,tC.UR)
#times = [tC.t_initial,tC.t_final]
times = [tC.t_initial,tC.t_initial + 2]
n_quadpoints = length(ctx.quadrature_points)
u_full = Vector{Float64}(4*2*n_quadpoints + ctx.n)
δ = 1e-9
for i in 1:(4*n_quadpoints)
        quadpoint = (divrem((i-1),2)[1]) % n_quadpoints
        u_full[2*(i-1)+1] = ctx.quadrature_points[quadpoint+1][1]
        u_full[2*(i-1)+2] = ctx.quadrature_points[quadpoint+1][2]
        factor = 0
        if (i-1) < 2*n_quadpoints
                factor = +1
        else
                factor = -1
        end
        shift = 0
        if (i-1)%2 == 1
                shift = 1
        end
        u_full[2*(i-1)+1+shift] += factor*δ
end
u_full[3] - ctx.quadrature_points[1][1]
u0 = nodal_interpolation(ctx, x -> norm(x - (tC.LL + tC.UR)/2) > 1 ? 1.0 : 0.0)
u_full[2*4*n_quadpoints + 1 : end] = u0
M = assembleMassMatrix(ctx)
p2 = Dict(
        "n_quadpoints" => n_quadpoints,
        "n" => ctx.n,
        "ctx" => ctx,
        "M"=>M,
        "p"=>tC.p,
        "δ" => δ,
        "ϵ" => 1e-4
        )
myExtendedRHS = (du,u,p2,t) -> juFEMDL.extendedRHS(interp_rhs,du,u,p2,t)
using OrdinaryDiffEq
prob = OrdinaryDiffEq.ODEProblem(
        myExtendedRHS,u_full,
        (tC.t_initial,tC.t_initial+1), p2)
using Sundials
sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.TRBDF2(), saveat=[tC.t_final],
                             save_everystep=false, dense=false,
                             reltol=1e-4, abstol=1e-4)
