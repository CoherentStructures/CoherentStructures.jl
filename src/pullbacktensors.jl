#Functions for pulling back Tensors



const default_tolerance = 1e-3
const default_solver = OrdinaryDiffEq.BS5()


"""
    flow(rhs,  u0, tspan; tolerance, p, solver)

Solve the ODE with right hand side given by `rhs` and initial value `u0`.
`p` is a parameter passed to `rhs`.
`tolerance` is passed as both relative and absolute tolerance to the solver,
which is determined by `solver`.
"""
function flow(
            rhs::Function,
            u0::AbstractVector{T},
            tspan::AbstractVector{Float64};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            #ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T<:Real}

    #callback = nothing
    #if ctx_for_boundscheck != nothing
    #   LL1::Float64 = ctx_for_boundscheck.spatialBounds[1][1]
    #   LL2::Float64 = ctx_for_boundscheck.spatialBounds[1][2]
    #   UR1::Float64 = ctx_for_boundscheck.spatialBounds[2][1]
    #   UR2::Float64 = ctx_for_boundscheck.spatialBounds[2][2]
    #   leftSide(x,y,integrator) = (x[1] - LL1) <= 0.0
    #   bottomSide(x,y,integrator) = (x[2] - LL2) <= 0.0
    #   rightSide(x,y,integrator) = (UR1 - x[1]) <= 0.0
    #   topSide(x,y,integrator) = (UR2 - x[2]) <= 0.0
    #   function affect!(integrator)
    #           return terminate!(integrator)#
    #   end
    #   callback = OrdinaryDiffEq.CallbackSet(
    #           map(x-> OrdinaryDiffEq.DiscreteCallback(x,affect!),
    #       [leftSide,rightSide,topSide,bottomSide])...)
   #end
   const num_args = DiffEqBase.numargs(rhs)
   if num_args == 4
       prob = OrdinaryDiffEq.ODEProblem(rhs, Vector{T}(u0), (tspan[1],tspan[end]), p)
       sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                             save_everystep=false, dense=false,
                             reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
       return sol.u
   elseif num_args == 3

       sprob = OrdinaryDiffEq.ODEProblem(rhs,StaticArrays.SVector{2}(u0[1],u0[2]), (tspan[1],tspan[end]), p)
       ssol = OrdinaryDiffEq.solve(sprob, solver, saveat=tspan,
                             save_everystep=false, dense=false,
                             reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
       return ssol.u
   else
       error("Invalid format of rhs")
   end
end

"""
    parallel_flow(flow_fun,P)

Apply the `flow_fun` to each element in `P` in parallel, if possible. Returns
a 3D array with dimensions (space dim x no. of time instances x no. of trajectories).
"""
function parallel_flow(flow_fun,P::AbstractArray)
    dim = length(P[1])
    T = eltype(P[1])
    dummy = flow_fun(P[1])
    q = length(dummy)

    sol = SharedArray{T}(dim,q,length(P));
    @sync @parallel for index in eachindex(P)
        @async begin
            u = flow_fun(P[index])
            for d=1:dim, t=1:q
                sol[d,t,index] = u[t][d]
            end
        end
    end
    return sol
    # TODO: think about rearring the array similarly to parallel_tensor
end

"""
    linearized_flow(odefun, x, tspan,δ; ...)

Calculate derivative of flow map by finite differences.
Caution: Currently assumes dim=2!
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
        stencil::Vector{T} = zeros(T, 8)
        @inbounds stencil[1:2] .= x .+ dx
        @inbounds stencil[3:4] .= x .+ dy
        @inbounds stencil[5:6] .= x .- dx
        @inbounds stencil[7:8] .= x .- dy

        rhs = (du,u,p,t) -> arraymap!(du,u,p,t,odefun, 4,2)
        prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
        sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u

        result = Tensors.Tensor{2,2,T,4}[]
        sizehint!(result,num_tsteps)
        @inbounds for i in 1:num_tsteps
            #The ordering of the stencil vector was chosen so
            #that  a:= stencil[1:4] - stencil[5:8] is a vector
            #so that Tensor{2,2}(a/2δ) approximates the Jacobi-Matrix
        	push!(result,Tensors.Tensor{2,2,T}( (sol[i][1:4] - sol[i][5:8])/2δ))
        end
        return result
    elseif num_args == 3
        #In order to solve only one ODE, write all the initial values
        #one after the other in one big vector
        sstencil::StaticArrays.SVector{8,Float64} = StaticArrays.SVector{8}(x[1] + δ, x[2], x[1],x[2] + δ, x[1] - δ, x[2], x[1],x[2] - δ)
        srhs = (u,p,t) -> arraymap(u,p,t,odefun)
        sprob = OrdinaryDiffEq.ODEProblem(srhs,sstencil,(tspan[1],tspan[end]),p)
        ssol = OrdinaryDiffEq.solve(sprob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u

        sresult = Tensors.Tensor{2,2,T,4}[]
        sizehint!(sresult,num_tsteps)
        @inbounds for i in 1:num_tsteps
            #The ordering of the stencil vector was chosen so
            #that  a:= stencil[1:4] - stencil[5:8] is a vector
            #so that Tensor{2,2}(a/2δ) approximates the Jacobi-Matrix
        	push!(sresult,Tensors.Tensor{2,2,T}( (ssol[i][1:4] - ssol[i][5:8])/2δ))
        end
        return sresult
    else
        error("odefun has invalid number of arguments")
    end

end

"""
    parallel_tensor(tensor_fun,P)

Computes a tensor field via `tensor_fun` for each element of `P`.
`tensor_fun` is a function that takes initial conditions as input and returns
a symmetric(!) tensor. The final tensor field array has the same size as `P`.
"""
function parallel_tensor(tensor_fun::Function,P::AbstractArray{T,N}) where T where N

    T_shared = SharedArray{Float64}(3,length(P))
    @sync @parallel for index in eachindex(P)
        T_shared[:,index] = tensor_fun(P[index])[[1,2,4]]
    end

    Tfield = Array{Tensors.SymmetricTensor{2,2,Float64,3}}(size(P))
    for index in eachindex(P)
        Tfield[index] = Tensors.SymmetricTensor{2,2}(T_shared[:,index])
    end
    return Tfield
end

"""
    mean_diff_tensor(odefun, u, tspan, δ; kwargs...)

Returns the averaged diffusion tensor at a point along a set of times.
Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `kwargs...`: are passed to `linearized_flow`
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

"""
    CG_tensor(odefun, u, tspan, δ; kwargs...)

Returns the classic right Cauchy--Green strain tensor. Derivatives are computed
with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `kwargs...`: are passed to `linearized_flow`
"""
@inline function CG_tensor(
            odefun,
            u::T,
            tspan::AbstractVector{Float64},
            δ::Float64;
            kwargs...
        ) where T
    return tdot(linearized_flow(odefun,u,[tspan[1],tspan[end]],δ;kwargs...)[end])
end

"""
    pullback_tensors(odefun, u, tspan, δ; D, kwargs...)

Returns the time-resolved pullback tensors of both the diffusion and
the metric tensor along a trajectory. Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `D`: (constant) diffusion tensor, metric tensor is computed via inversion; defaults to `eye(2)`
   * `kwargs...` are passed through to `linearized_flow`
"""

function pullback_tensors(
            odefun::Function,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
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

"""
    pullback_metric_tensor(odefun, u, tspan, δ; G, kwargs...)

Returns the time-resolved pullback tensors of the metric tensor along a trajectory,
aka right Cauchy-Green strain tensor.
Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `G`: (constant) metric tensor
   * `kwargs...` are passed through to `linearized_flow`
"""

@inline function pullback_metric_tensor(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            G::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real} # TODO: add dim for 3D

    iszero(δ) ?
        DF = linearized_flow(odefun, u, tspan,    p=p, tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun, u, tspan, δ, p=p, tolerance=tolerance, solver=solver)

    return [symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
end

"""
    pullback_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...)

Returns the time-resolved pullback tensors of the diffusion tensor along a trajectory.
Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `D`: (constant) diffusion tensor
   * `kwargs...` are passed through to `linearized_flow`
"""

@inline function pullback_diffusion_tensor(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
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
    result = Tensors.SymmetricTensor{2,2,Float64,3}[]
    sizehint!(result,tlen)
    for i in 1:tlen
	    push!(result,symmetric(DF[i] ⋅ Dfun(pos[i]) ⋅ transpose(DF[i])))
    end
    return result
end

"""
    pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...)

Returns the time-resolved pullback tensors of the diffusion tensor in SDEs.
Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `D`: (constant) diffusion tensor
   * `kwargs...` are passed through to `linearized_flow`
"""

@inline function pullback_SDE_diffusion_tensor(
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
    DF .= inv.(DF)
    return [df ⋅ B for df in DF]
end

"""
    av_weighted_CG_tensor(odefun, u, tspan, δ; G, kwargs...)

Returns the transport tensor of a trajectory, aka  time-averaged,
di ffusivity-structure-weighted version of the classic right Cauchy–Green strain
tensor. Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `G`: (constant) metric tensor
   * `kwargs...` are passed through to `linearized_flow`
"""

function av_weighted_CG_tensor(
            odefun,
            u::AbstractArray{T,1},
            tspan::AbstractVector{Float64},
            δ::Float64;
            D::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real} # TODO: add dim for 3D

    G = inv(D)
    iszero(δ) ?
        DF = linearized_flow(odefun, u, tspan,    p=p, tolerance=tolerance, solver=solver) :
        DF = linearized_flow(odefun, u, tspan, δ, p=p, tolerance=tolerance, solver=solver)

    return det(D)*mean([symmetric(transpose(df) ⋅ G ⋅ df) for df in DF])
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
            D::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
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
            G::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
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
                D::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
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
                D::Tensors.SymmetricTensor{2,2,T,3}=one(Tensors.SymmetricTensor{2,2,T,3}),
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
