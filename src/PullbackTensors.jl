#Functions for pulling back Tensors



"""function flow(rhs,  u0, tspan; tolerance, p, solver)

Solve the ODE with right hand side given by `@param rhs` and initial value given by `@param u0`.
`dim` is the dimension of the ODE.
`p` is a parameter passed as the third argument to `rhs`
"""
function flow(
            rhs::Function,
            u0::AbstractArray{T,1},
            tspan::AbstractVector{Float64};
            tolerance = 1.e-3,
            p = nothing,
            solver = OrdinaryDiffEq.BS5(),
            ctx_for_boundscheck=nothing
        ) where {T<:Real}
        
    callback = nothing
    if ctx_for_boundscheck != nothing
       LL::Vec{2,Float64} = ctx_for_boundscheck.spatialBounds[1]
       UR::Vec{2,Float64} = ctx_for_boundscheck.spatialBounds[2]
       leftSide(x,y,integrator) = x[1] - LL[1]
       bottomSide(x,y,integrator) = x[2] - LL[2]
       rightSide(x,y,integrator) = UR[1] - x[1]
       topSide(x,y,integrator) = UR[2] - x[2]
       callback = OrdinaryDiffEq.CallbackSet(
           map(x-> OrdinaryDiffEq.ContinuousCallback(x,OrdinaryDiffEq.terminate!),
           [leftSide,rightSide,topSide,bottomSide])...)
   end
   prob = OrdinaryDiffEq.ODEProblem(rhs, Array{T}(u0),# is this Array a no-op for arrays? if not, dispatch
       (tspan[1],tspan[end]), p,callback=callback)
   sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                         save_everystep=false, dense=false,
                         reltol=tolerance, abstol=tolerance)
   return sol.u
end

#Calculate derivative of flow map by finite differences.
#TODO: implement this for dim==3, currently it only works if dim==2
@inline function linearized_flow(
            odefun::Function,
            x::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            tolerance::Float64 = 1.e-3,
            p = nothing,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real}

    const dx = [δ, zero(δ)];
    const dy = [zero(δ), δ];

    #In order to solve only one ODE, write all the initial values
    #one after the other in one big vector
    stencil = zeros(8)
    @inbounds stencil[1:2] .= x .+ dx
    @inbounds stencil[3:4] .= x .+ dy
    @inbounds stencil[5:6] .= x .- dx
    @inbounds stencil[7:8] .= x .- dy

    const num_tsteps = length(tspan)
    prob = OrdinaryDiffEq.ODEProblem((du,u,p,t) -> arraymap(du,Array{T}(u),p,t,odefun, 4,2),
                                     stencil, (tspan[1],tspan[end]), p)

    sol = OrdinaryDiffEq.solve(prob, solver, saveat = tspan, save_everystep = false,
                               dense = false, reltol = tolerance, abstol = tolerance).u

    result = zeros(Tensor{2,2}, num_tsteps)
    @inbounds for i in 1:num_tsteps
        #The ordering of the stencil vector was chosen so
        #that  a:= stencil[1:4] - stencil[5:8] is a vector
        #so that Tensor{2,2}(a) approximates the Jacobi-Matrix
    	@inbounds result[i] = Tensor{2,2}( (sol[i][1:4] - sol[i][5:8])/2δ)
    end
    return result
end

#TODO: document this
# This is the autodiff-version of linearized_flow
function linearized_flow(
            odefun::Function,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64};
            tolerance::Float64 = 1.e-3,
            p = nothing,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real}  # TODO: add dim

    Flow(x) = flow(odefun,x,tspan,tolerance=tolerance,p=p,solver=solver)
    DF      = ForwardDiff.jacobian(Flow,u)
    df      = [Tensor{2,2}(DF[i:i+(dim-1),:]) for i=1:dim:size(DF,1)]
    return df
end

"""`invCGTensor(odefun, x, tspan,  ::Float64; tolerance, p)`

Returns the average (inverse) CG-Tensor at a point along a set of times
Derivatives are computed with finite differences

  -`x::Vec{2,Float64}` the initial point
  -`tspan::Array{Float64}` is the times
  -`` is the stencil width for the finite differences
  -`odefun` is a function that takes arguments `(x,t,result)`

`odefun` needs to store its result in `result`.
`odefun` evaluates the rhs of the ODE being integrated at `(x,t)`
"""
@inline function invCGTensor(
            odefun,
            x::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            kwargs...
        ) where {T<:Real} # TODO: add dim
    return mean(dott.(inv.(linearized_flow(odefun,x,tspan,δ;kwargs...))))
end

#TODO: Document the functions below, then add to exports.jl
#TODO: Pass through tolerance for ODE solver etc.
function pullback_tensors(
            odefun::Function,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real}  # TODO: add dim for 3D

    G = inv(D)

    iszero(δ) ?
      DF = linearized_flow(odefun, u, tspan,    p=p, tolerance=tolerance, solver=solver) :
      DF = linearized_flow(odefun, u, tspan, δ, p=p, tolerance=tolerance, solver=solver)

    MT = [symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
    DF .= inv.(DF)
    DT = [symmetric(df ⋅ D ⋅ transpose(df)) for df in DF]
    return MT, DT # MT is pullback metric tensor, DT is pullback diffusion tensor
end

function pullback_metric_tensor(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            G::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real} # TODO: add dim for 3D

    iszero(δ) ?
      DF = linearized_flow(odefun, u, tspan,    p=p, tolerance=tolerance, solver=solver) :
      DF = linearized_flow(odefun, u, tspan, δ, p=p, tolerance=tolerance, solver=solver)

    return [symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
end

function pullback_diffusion_tensor(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real} # TODO: add dim for 3D

    iszero(δ) ?
      DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
      DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)

    DF .= inv.(DF)
    return [symmetric(df ⋅ D ⋅ transpose(df)) for df in DF]
end

function met2deg(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2,T,3}, [1/cos(deg2rad(u[2])), one(T)])
end

function deg2met(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2,T,3}, [cos(deg2rad(u[2])), one(T)])
end

function pullback_tensors_geo(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{T},
            δ::T;
            D::SymmetricTensor{2,2}=one(SymmetricTensor{2,2}),
            tol::Float64=1e-3,
            p=nothing,
            solver=OrdinaryDiffEq.BS5()
        ) where {T<:Real}

    G = inv(D)
    met2deg_init = met2deg(u)
    prob = ODEProblem(odefun,Array{T}(u),(tspan[1],tspan[end]),p)
    sol = solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=1e-5,abstol=1e-5).u
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)
    
    PBmet = [deg2met(sol[i]) ⋅ DF[i] ⋅ met2deg_init for i in eachindex(DF,sol)]
    PBdiff = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return [symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PBmet], [symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PBdiff]
end

function pullback_metric_tensor_geo(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{T},
            δ::T;
            G::SymmetricTensor{2,2}=one(SymmetricTensor{2,2}),
            tolerance::Float64=1e-3,
            p=nothing,
            solver=OrdinaryDiffEq.BS5()
        ) where {T<:Real}

    met2deg_init = met2deg(u)
    prob = ODEProblem(odefun,Array{T}(u),(tspan[1],tspan[end]),p)
    sol = solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=1e-5,abstol=1e-5).u
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)
    
    PB = [deg2met(sol[i]) ⋅ DF[i] ⋅ met2deg_init for i in eachindex(DF,sol)]
    return [symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PB]
end

@everywhere function pullback_diffusion_tensor_geo(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{T},
                δ::T;
                D::SymmetricTensor{2,2}=one(SymmetricTensor{2,2}),
                tolerance::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real}

    prob = OrdinaryDiffEq.ODEProblem(odefun,Array{T}(u),(tspan[1],tspan[end]),p)
    sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)
    
    PB = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return [symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PB]
end

@everywhere function pullback_SDE_diffusion_tensor_geo(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{T},
                δ::T;
                D::SymmetricTensor{2,2}=one(SymmetricTensor{2,2}),
                tol::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real}

    prob = ODEProblem(odefun,Array{T}(u),(tspan[1],tspan[end]),p)
    sol = solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)
    
    B = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return B
end
