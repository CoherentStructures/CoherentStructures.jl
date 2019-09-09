# (c) 2018-2019 Philipp Rohrmüller

################################################################################
### LinearImplicitEuler:
################################################################################

# RosenbrockAlgorithm instead of OrdinaryDiffEqAlgorithm (massmatrix issues)
struct LinearImplicitEuler{CS,AD,F}  <: OrdinaryDiffEq.OrdinaryDiffEqRosenbrockAlgorithm{CS,AD}
	linsolve::F
end
LinearImplicitEuler(;chunk_size=0,autodiff=false,linsolve=DEFAULT_LINSOLVE) = LinearImplicitEuler{chunk_size,autodiff,typeof(linsolve)}(linsolve)
OrdinaryDiffEq.alg_order(alg::LinearImplicitEuler) = 1

mutable struct LinearImplicitEulerCache{uType,rateType,J,F} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache  # removed: @cache
    u::uType
    uprev::uType
    k::rateType
    W::J
    step::Bool
    linsolve::F
end

function OrdinaryDiffEq.alg_cache(alg::LinearImplicitEuler,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,
    tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{true})

    k = zero(rate_prototype)
    W = similar(f.f*1)
    linsolve = alg.linsolve(Val{:init},W,u)
    LinearImplicitEulerCache(u,uprev,k,W,true,linsolve)
end

function DiffEqBase.initialize!(integrator, cache::LinearImplicitEulerCache)
    integrator.fsalfirst = zero(cache.k)
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
    integrator.fsallast = zero(integrator.fsalfirst)
    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

OrdinaryDiffEq.@muladd function OrdinaryDiffEq.perform_step!(integrator, cache::LinearImplicitEulerCache, repeat_step=false)
    OrdinaryDiffEq.@unpack t,dt,uprev,u,f,p = integrator
    OrdinaryDiffEq.@unpack k = cache
    alg = OrdinaryDiffEq.unwrap_alg(integrator, true)

    if DiffEqOperators.is_constant(f.f)
        if cache.step
            cache.W = f.mass_matrix - dt*f.f
            cache.linsolve(vec(u), cache.W, vec(f.mass_matrix*k), true)
            cache.step = false
        end
        cache.linsolve(vec(u), cache.W, vec(f.mass_matrix*uprev), false)
    else
        L = f.f
        DiffEqOperators.update_coefficients!(L,u,p,t+dt)
        cache.W = f.mass_matrix - dt*L
        cache.linsolve(vec(u), cache.W, vec(f.mass_matrix*uprev), true)
    end
    f(integrator.fsallast, u, p, t + dt)
end


################################################################################
### LinearMEBDF2:
################################################################################

# RosenbrockAlgorithm instead of OrdinaryDiffEqAlgorithm (massmatrix issues)
struct LinearMEBDF2{CS,AD,F}  <: OrdinaryDiffEq.OrdinaryDiffEqRosenbrockAlgorithm{CS,AD}
	linsolve::F
end
LinearMEBDF2(;chunk_size=0,autodiff=false,linsolve=DEFAULT_LINSOLVE) = LinearMEBDF2{chunk_size,autodiff,typeof(linsolve)}(linsolve)
OrdinaryDiffEq.alg_order(alg::LinearMEBDF2) = 2

mutable struct LinearMEBDF2Cache{uType,rateType,J,F} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache #removed: @cache
    u::uType
    uprev::uType
    z₁::rateType
    z₂::rateType
    z₃::rateType
    tmp::rateType
    k::rateType
    fsalfirst::rateType
    W::J
    W₂::J
    step::Bool
    linsolve::F
    linsolve2::F
end

function OrdinaryDiffEq.alg_cache(alg::LinearMEBDF2,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,
    tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{true})

    k = zero(rate_prototype)
    fsalfirst = zero(rate_prototype)
    W  = similar(f.f*1)
    W₂ = similar(W)
    z₁ = zero(rate_prototype)
    z₂ = similar(z₁)
    z₃ = similar(z₁)
    tmp = similar(z₁)
    linsolve = alg.linsolve(Val{:init},W,u)
    linsolve2 = alg.linsolve(Val{:init},W,u)
    LinearMEBDF2Cache(u,uprev,z₁,z₂,z₃,tmp,k,fsalfirst,W,W₂,true,linsolve,linsolve2)
end

function OrdinaryDiffEq.initialize!(integrator, cache::LinearMEBDF2Cache)
    integrator.kshortsize = 2
    integrator.fsalfirst = cache.fsalfirst
    integrator.fsallast = cache.k
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
end

OrdinaryDiffEq.@muladd function OrdinaryDiffEq.perform_step!(integrator, cache::LinearMEBDF2Cache, repeat_step=false)
    OrdinaryDiffEq.@unpack t,dt,uprev,u,f,p = integrator
    OrdinaryDiffEq.@unpack k,z₁,z₂,tmp = cache
    alg = OrdinaryDiffEq.unwrap_alg(integrator, true)

    if DiffEqOperators.is_constant(f.f)
        if cache.step
            cache.W = f.mass_matrix - dt*f.f
            z₁ = f.mass_matrix*uprev
            cache.linsolve(vec(tmp), cache.W, vec(z₁), true)
            cache.step = false
        end

        if f.mass_matrix == LinearAlgebra.I
            ##### STEP 1:
            cache.linsolve(vec(z₁), cache.W, vec(uprev), false)
            ##### STEP 2:
            cache.linsolve(vec(z₂), cache.W, vec(z₁), false)
            ### precalculation for STEP 3:
            OrdinaryDiffEq.@.. tmp = -0.5z₂ + z₁ + 0.5uprev
            z₁ .= tmp
            ### M ≠ I:
        else
            ##### STEP 1:
            mul!(tmp,f.mass_matrix,uprev)
            cache.linsolve(vec(z₁), cache.W, vec(tmp), false)
            ##### STEP 2:
            mul!(tmp,f.mass_matrix,z₁)
            cache.linsolve(vec(z₂), cache.W, vec(tmp), false)
            # precalculation for STEP 3:
            OrdinaryDiffEq.@.. tmp = -0.5z₂ + z₁ + 0.5uprev
            mul!(z₁,f.mass_matrix,tmp)
        end
        ##### STEP 3:
        cache.linsolve(vec(u), cache.W, vec(z₁), false)
        # update integrator:
        f(integrator.fsallast, u, p, t + dt)
    else  # time-dependent case
        L = f.f
        if cache.step
            DiffEqOperators.update_coefficients!(L,u,p,t+dt)
            cache.W = f.mass_matrix - dt*L
            cache.step = false
        end

        if f.mass_matrix == I
            ##### STEP 1:
            cache.linsolve(vec(z₁), cache.W, vec(uprev), true)
            ##### STEP 2:
            DiffEqOperators.update_coefficients!(L,u,p,t+2dt)
            cache.W₂ = f.mass_matrix - dt*L
            cache.linsolve2(vec(z₂), cache.W₂, vec(z₁), true)
            # precalculation for STEP 3:
            OrdinaryDiffEq.@.. tmp = -0.5z₂ + z₁ + 0.5uprev
            z₁ .= tmp
        else ### M ≠ I
            ##### STEP 1:
            mul!(tmp,f.mass_matrix,uprev)
            cache.linsolve(vec(z₁), cache.W, vec(tmp), true)
            ##### STEP 2:
            DiffEqOperators.update_coefficients!(L,u,p,t+2dt)
            cache.W₂ = f.mass_matrix - dt*L
            mul!(tmp,f.mass_matrix,z₁)
            cache.linsolve2(vec(z₂), cache.W₂, vec(tmp), true)
            # precalculation for STEP 3:
            OrdinaryDiffEq.@.. tmp = -0.5z₂ + z₁ + 0.5uprev
            mul!(z₁,f.mass_matrix,tmp)
        end
        ##### STEP 3:
        cache.linsolve(vec(u), cache.W, vec(z₁), false)
        # update integrator:
        cache.W .= cache.W₂
        f(integrator.fsallast, u, p, t + dt)
    end
end
