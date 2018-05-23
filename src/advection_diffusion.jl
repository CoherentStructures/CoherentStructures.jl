#(c) 2018 Nathanael Schilling
#Routines for numerically solving the Advection-Diffusion Equation

"""
Single step with implicit Euler method.
"""
function ADimplicitEulerStep(ctx,u,edt, Afun,q=nothing,M=nothing,K=nothing)
    if M == nothing
        M = assembleMassMatrix(ctx)
    end
    if K == nothing
        K = assembleStiffnessMatrix(ctx,Afun,q)
    end
    return (M - edt*K)\(M*u)
end


"""
    advect_serialized_quadpoints(ctx,t0,tf,odefun,δ=1e-9

Advect all quadrature points + finite difference stencils from t0
to tf with ode rhs given by `odefun`
Returns an ODE solution object
"""
function advect_serialized_quadpoints(ctx,t0,tf,odefun!,p=nothing,δ=1e-9;solver=default_solver,tolerance=default_tolerance)
    u0 = setup_fd_quadpoints_serialized(ctx,δ)
    p2 = Dict(
        "ctx" => ctx,
        "p"=>p,
        )

    prob = OrdinaryDiffEq.ODEProblem((du,u,p,t) -> large_rhs(odefun!,du,u,p,t), u0,(t0,tf),p2)
    return OrdinaryDiffEq.solve(prob,solver,abstol=tolerance,reltol=tolerance)
end


function stiffnessMatrixTimeT(ctx,sol,t,δ=1e-9)
        if t < 0
                return assembleStiffnessMatrix(ctx)
        end
        p = sol(t)
        function Afun(x,index,p)
                Df = Tensors.Tensor{2,2}((p[(8*(index-1) + 1):(8*(index-1) + 4)] - p[ (8*(index-1)+5):(8*(index-1) + 8)])/(2δ))
                return Tensors.dott(Tensors.inv(Df))
        end
        return assembleStiffnessMatrix(ctx,Afun,p)
end



@inline function large_rhs(odefun!, du,u,p,t)
    n_quadpoints = length(p["ctx"].quadrature_points)
    @views CoherentStructures.arraymap!(du,u,p["p"],t,odefun!,4*n_quadpoints,2)
end

"""
    setup_serialized_quadpoints(ctx,δ=1e-9)

For each quadrature point in `ctx.quadrature_points`, setup a finite difference scheme around that point.
Then write the resulting points into a (flat) array onto which arraymap can be applied.
Only works in 2D at the moment
"""
function setup_fd_quadpoints_serialized(ctx,δ=1e-9)
    n_quadpoints = length(ctx.quadrature_points)
    u_full = zeros(8*n_quadpoints)
    for i in 1:n_quadpoints
            quadpoint = ctx.quadrature_points[i]
            for j in 1:4
                factor = (j>2)? -1 : 1
                shift = (j % 2 == 0) ? 1 : 0
                u_full[8*(i-1)+2(j-1) + 1] = quadpoint[1]
                u_full[8*(i-1)+2(j-1) + 2] = quadpoint[2]
                u_full[8*(i-1)+2(j-1) + 1 + shift] += factor*δ
            end
    end
    return u_full
end




#Helper function for the case where we've precomputed the diffusion tensors
function PCDiffTensors(x,index,p)
    return p[index]
 end

 function extendedRHSNonStiff(oderhs,du,u,p,t)
     print("t = $t")
    n_quadpoints = p["n_quadpoints"]
    @views arraymap(du,u,p["p"],t,oderhs,4*n_quadpoints,2)
end

function extendedRHSStiff!(A,u,p,t)
    #First 4*n_quadpoints*2 points are ODE for flow-map at quadrature points
    #The next n points are for the solution of the AD-equation

    n_quadpoints = p["n_quadpoints"]
    n = p["n"]
    δ = p["δ"]
    DF = Vector{Tensor{2,2}}(n_quadpoints)
    for i in 1:n_quadpoints
        DF[i] = Tensor{2,2}(
            (u[(4*(i-1) +1):(4*i)] - u[(4*n_quadpoints +1 +4*(i-1)):(4*n_quadpoints+4*i)])/2δ
            )
    end
    invDiffTensors = dott.(inv.(DF))
    ctx = p["ctx"]
    ϵ = p["ϵ"]
    K = assembleStiffnessMatrix(ctx,PCDiffTensors,invDiffTensors)
    I,J,V = findnz(K)
    M = assembleMassMatrix(ctx,lumped=true)
    for index in 1:length(I)
        i = I[index]
        j = J[index]
        A[i,j] = ϵ*V[index]/M[i,i]
    end
    print("t = $t")
    return
end
