#Functions for pulling back Tensors



const default_tolerance = 1e-3
const default_solver = OrdinaryDiffEq.BS5()
"""function flow(rhs,  u0, tspan; tolerance, p, solver)

Solve the ODE with right hand side given by `rhs` and initial value `u0`.
`p` is a parameter passed to `rhs`.
`tolerance` is passed as both relative and absolute tolerance to the solver,
which is determined by `solver`.
"""
function flow(
            rhs::Function,
            u0::T,
            tspan::AbstractVector{Float64};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T}

    callback = nothing
    if ctx_for_boundscheck != nothing
       LL1::Float64 = ctx_for_boundscheck.spatialBounds[1][1]
       LL2::Float64 = ctx_for_boundscheck.spatialBounds[1][2]
       UR1::Float64 = ctx_for_boundscheck.spatialBounds[2][1]
       UR2::Float64 = ctx_for_boundscheck.spatialBounds[2][2]
       leftSide(x,y,integrator) = (x[1] - LL1) <= 0.0
       bottomSide(x,y,integrator) = (x[2] - LL2) <= 0.0
       rightSide(x,y,integrator) = (UR1 - x[1]) <= 0.0
       topSide(x,y,integrator) = (UR2 - x[2]) <= 0.0
       function affect!(integrator)
           return terminate!(integrator)
       end
       callback = OrdinaryDiffEq.CallbackSet(
           map(x-> OrdinaryDiffEq.DiscreteCallback(x,affect!),
           [leftSide,rightSide,topSide,bottomSide])...)
   end
   prob = OrdinaryDiffEq.ODEProblem(rhs, u0, (tspan[1],tspan[end]), p)
   sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                         save_everystep=false, dense=false,
                         reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
   return sol.u
end

# this is a flow-function that works with ForwardDiff
function ad_flow(
            odefun::Function,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64};
            tolerance = 1.e-3,
            p = nothing,
            solver = OrdinaryDiffEq.BS5()
        ) where {T<:Real}

    prob = OrdinaryDiffEq.ODEProblem(odefun,u,T.((tspan[1], tspan[end])),p)
    sol = convert(Array,OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance))
end

"""
Calculate derivative of flow map by finite differences.
Currently assumes dim=2
"""
@inline function linearized_flow(
            odefun::Function,
            x::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            tolerance=default_tolerance,
            solver=default_solver,
            p=nothing,
            kwargs...
        ) where {T <: Real}

    const num_tsteps = length(tspan)
    const num_args = DiffEqBase.numargs(odefun)
    if num_args == 4
        const dx = [δ, zero(δ)];
        const dy = [zero(δ), δ];

        #In order to solve only one ODE, write all the initial values
        #one after the other in one big vector
        stencil = zeros(T, 8)
        @inbounds stencil[1:2] .= x .+ dx
        @inbounds stencil[3:4] .= x .+ dy
        @inbounds stencil[5:6] .= x .- dx
        @inbounds stencil[7:8] .= x .- dy

        rhs = (du,u,p,t) -> arraymap!(du,u,p,t,odefun, 4,2)
        prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
        sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u

        result = Tensor{2,2,T,4}[]
        sizehint!(result,num_tsteps)
        @inbounds for i in 1:num_tsteps
            #The ordering of the stencil vector was chosen so
            #that  a:= stencil[1:4] - stencil[5:8] is a vector
            #so that Tensor{2,2}(a/2δ) approximates the Jacobi-Matrix
        	push!(result,Tensor{2,2,T}( (sol[i][1:4] - sol[i][5:8])/2δ))
        end
        return  result
    elseif num_args == 3
        #In order to solve only one ODE, write all the initial values
        #one after the other in one big vector
        sstencil::SVector{8,Float64} = @SVector [x[1] + δ, x[2], x[1],x[2] + δ, x[1] - δ, x[2], x[1],x[2] - δ]
        srhs = (u,p,t) -> arraymap(u,p,t,odefun)
        sprob = OrdinaryDiffEq.ODEProblem(srhs,sstencil,(tspan[1],tspan[end]),p)
        ssol = OrdinaryDiffEq.solve(sprob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u

        sresult = Tensor{2,2,T,4}[]
        sizehint!(sresult,num_tsteps)
        @inbounds for i in 1:num_tsteps
            #The ordering of the stencil vector was chosen so
            #that  a:= stencil[1:4] - stencil[5:8] is a vector
            #so that Tensor{2,2}(a/2δ) approximates the Jacobi-Matrix
        	push!(sresult,Tensor{2,2,T}( (ssol[i][1:4] - ssol[i][5:8])/2δ))
        end
        return  sresult
    else
        error("odefun has invalid number of arguments")
    end

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

    dim = length(u)
    Flow(x) = ad_flow(odefun,x,tspan,tolerance=tolerance,p=p,solver=solver)
    DF      = ForwardDiff.jacobian(Flow,u)
    df      = [Tensor{2,2}(DF[i:i+(dim-1),:]) for i=1:dim:size(DF,1)]
    return df
end

"""`mean_diff_tensor(odefun, u, tspan, δ; tolerance, p)`

Returns the averaged diffusion tensor at a point along a set of times.
Derivatives are computed with finite differences.

  -`odefun` is the RHS of an ODE
  -`u` the initial value of the ODE
  -`tspan` set of time instances at which to save the trajectory
  -`δ` is the stencil width for the finite differences
  -`kwargs` are passed to `linearized_flow`
"""

@inline function mean_diff_tensor(
            odefun,
            u::T,
            tspan::AbstractVector{Float64},
            δ::Float64;
            kwargs...
        ) where T
    return mean(dott.(inv.(linearized_flow(odefun,u,tspan,δ;kwargs...))))
end

"""`pullback_tensors(odefun, u, tspan, δ; D, kwargs...)`

Returns the time-resolved pullback tensors of both the diffusion and
the metric tensor along a trajectory.
Derivatives are computed with finite differences.

  -`odefun` is the RHS of an ODE
  -`u` the initial value of the ODE
  -`tspan` set of time instances at which to save the trajectory
  -`δ` is the stencil width for the finite differences
  -`D` is the (constant) diffusion tensor, the metric tensor is computed via inversion
  -`kwargs` are passed through to `linearized_flow`
"""

function pullback_tensors(
            odefun::Function,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real}  # TODO: add dim for 3D

    G = inv(D)

    iszero(δ) ?
        DF = linearized_flow(odefun, u, tspan;    kwargs...) :
        DF = linearized_flow(odefun, u, tspan, δ; kwargs...)

    MT = [symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
    DF .= inv.(DF)
    DT = [symmetric(df ⋅ D ⋅ transpose(df)) for df in DF]
    return MT, DT # MT is pullback metric tensor, DT is pullback diffusion tensor
end

"""`pullback_metric_tensor(odefun, u, tspan, δ; G, kwargs...)`

Returns the time-resolved pullback tensors of the metric tensor along a trajectory,
aka right Cauchy-Green strain tensor.
Derivatives are computed with finite differences.

  -`odefun` is the RHS of an ODE, can be mutating or not, then it should return
    a StaticVector
  -`u` the initial value of the ODE
  -`tspan` set of time instances at which to save the trajectory
  -`δ` is the stencil width for the finite differences
  -`G` is the (constant) metric tensor
  -`kwargs` are passed through to `linearized_flow`
"""

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

"""`pullback_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...)`

Returns the time-resolved pullback tensors of the diffusion tensor along a trajectory.
Derivatives are computed with finite differences.

  -`odefun` is the RHS of an ODE, can be mutating or not, then it should return
    a StaticVector
  -`u` the initial value of the ODE
  -`tspan` set of time instances at which to save the trajectory
  -`δ` is the stencil width for the finite differences
  -`D` is the (constant) diffusion tensor
  -`kwargs` are passed through to `linearized_flow`
"""

function pullback_diffusion_tensor(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real} # TODO: add dim for 3D

    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan;    kwargs...) :
        DF = linearized_flow(odefun,u,tspan, δ; kwargs...)

    DF .= inv.(DF)
    return [symmetric(df ⋅ D ⋅ transpose(df)) for df in DF]
end

function pullback_diffusion_tensor_function(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64,
            Dfun::Function;
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real} # TODO: add dim for 3D

    DF, pos = iszero(δ) ?
        linearized_flow(odefun,u,tspan, p=p,tolerance=tolerance, solver=solver,give_back_position=true) :
        linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver,give_back_position=true)

    DF .= inv.(DF)
    tlen = length(tspan)
    result = SymmetricTensor{2,2,Float64,3}[]
    sizehint!(result,tlen)
    for i in 1:tlen
	    push!(result,symmetric(DF[i] ⋅ Dfun(pos[i]) ⋅ transpose(DF[i])))
    end
    return result
end

"""`pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...)`

Returns the time-resolved pullback tensors of the diffusion tensor in SDEs.
Derivatives are computed with finite differences.

  -`odefun` is the RHS of an ODE, can be mutating or not, then it should return
    a StaticVector
  -`u` the initial value of the ODE
  -`tspan` set of time instances at which to save the trajectory
  -`δ` is the stencil width for the finite differences
  -`D` is the (constant) diffusion tensor
  -`kwargs` are passed through to `linearized_flow`
"""

function pullback_SDE_diffusion_tensor(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{T},
                δ::T;
                B::Tensors.Tensor{2,2,T,4}=one(Tensors.Tensor{2,2,T,4}),
                kwargs...
            ) where {T<:Real}

    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan;    kwargs...) :
        DF = linearized_flow(odefun,u,tspan, δ; kwargs...)

    return inv.(DF) ⋅ B
end

function met2deg(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2,T,4}, [1/cos(deg2rad(u[2])), one(T)])
end

function deg2met(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2,T,4}, [cos(deg2rad(u[2])), one(T)])
end

function pullback_tensors_geo(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{T},
            δ::T;
            D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            tolerance::Float64=1e-3,
            p=nothing,
            solver=OrdinaryDiffEq.BS5()
        ) where {T<:Real}

    G = inv(D)
    met2deg_init = met2deg(u)
    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
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
            G::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
            tolerance::Float64=1e-3,
            p=nothing,
            solver=OrdinaryDiffEq.BS5()
        ) where {T<:Real}

    met2deg_init = met2deg(u)
    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)

    PB = [deg2met(sol[i]) ⋅ DF[i] ⋅ met2deg_init for i in eachindex(DF,sol)]
    return [symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PB]
end

function pullback_diffusion_tensor_geo(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{T},
                δ::T;
                D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
                tolerance::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real}

    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)

    PB = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return [symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PB]
end

function pullback_SDE_diffusion_tensor_geo(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{T},
                δ::T;
                D::SymmetricTensor{2,2,T,3}=one(SymmetricTensor{2,2,T,3}),
                tolerance::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real}

    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    iszero(δ) ?
        DF = linearized_flow(odefun,u,tspan,    p=p,tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver)

    B = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return B
end
