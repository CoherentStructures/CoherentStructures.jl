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


#Helper function for the case where we've precomputed the diffusion tensors
function PCDiffTensors(x,index,p)
    return p[index]
 end

function extendedRHS(oderhs,du,u, p,t)
    #First 4*n_quadpoints*2 points are ODE for flow-map at quadrature points
    #The next n points are for the solution of the AD-equation
    n_quadpoints = p["n_quadpoints"]
    @views arraymap(du,u,p["p"],t,oderhs,4*n_quadpoints,2)
    n = p["n"]
    δ = p["δ"]
    DF = Vector{Tensor{2,2}}(n_quadpoints)
    for i in 1:n_quadpoints
        DF[i] = Tensor{2,2}(
            (u[(4*(i-1) +1):(4*i)] - u[(4*n_quadpoints +1 +4*(i-1)):(4*n_quadpoints+4*i)])/2δ
            )
    end
    invDiffTensors = dott.(inv.(DF))
    M = p["M"]
    ctx = p["ctx"]
    ϵ = p["ϵ"]
    K = assembleStiffnessMatrix(ctx,PCDiffTensors,invDiffTensors)
    du[4*2*n_quadpoints+1:end] = ϵ*M\(K*u[4*2*n_quadpoints+1:end])
    print("t = $t")
end
