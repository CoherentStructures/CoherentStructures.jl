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
