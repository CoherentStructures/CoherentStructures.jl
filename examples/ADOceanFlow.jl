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
