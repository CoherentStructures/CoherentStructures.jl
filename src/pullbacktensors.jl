#Functions for pulling back Tensors
const SA = StaticArrays

const default_tolerance = 1e-3
const default_solver = OrdinaryDiffEq.BS5()



"""
    flow(odefun,  u0, tspan; tolerance, p, solver) -> Vector{Vector}

Solve the ODE with right hand side given by `odefun` and initial value `u0`.
`p` is a parameter passed to `odefun`.
`tolerance` is passed as both relative and absolute tolerance to the solver,
which is determined by `solver`.
"""
# function flow(
#             odefun::Function,
#             u0::SA.SVector{dim,T},
#             tspan::AbstractVector{Float64};
#             tolerance = default_tolerance,
#             p = nothing,
#             solver = default_solver,
#             #ctx_for_boundscheck=nothing,
#             force_dtmin=false
#         ) where {T<:Real,dim}
#     #callback = nothing
#     #if ctx_for_boundscheck != nothing
#     #   LL1::Float64 = ctx_for_boundscheck.spatialBounds[1][1]
#     #   LL2::Float64 = ctx_for_boundscheck.spatialBounds[1][2]
#     #   UR1::Float64 = ctx_for_boundscheck.spatialBounds[2][1]
#     #   UR2::Float64 = ctx_for_boundscheck.spatialBounds[2][2]
#     #   leftSide(x,y,integrator) = (x[1] - LL1) <= 0.0
#     #   bottomSide(x,y,integrator) = (x[2] - LL2) <= 0.0
#     #   rightSide(x,y,integrator) = (UR1 - x[1]) <= 0.0
#     #   topSide(x,y,integrator) = (UR2 - x[2]) <= 0.0
#     #   function affect!(integrator)
#     #           return terminate!(integrator)#
#     #   end
#     #   callback = OrdinaryDiffEq.CallbackSet(
#     #           map(x-> OrdinaryDiffEq.DiscreteCallback(x,affect!),
#     #       [leftSide,rightSide,topSide,bottomSide])...)
#    #end
#    num_args = DiffEqBase.numargs(odefun)
#    if num_args == 4
#        prob = OrdinaryDiffEq.ODEProblem(odefun, Vector{T}(u0), (tspan[1],tspan[end]), p)
#        sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
#                              save_everystep=false, dense=false,
#                              reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
#        return sol.u
#    elseif num_args == 3
#        sprob = OrdinaryDiffEq.ODEProblem(odefun,u0, (tspan[1],tspan[end]), p)
#        ssol = OrdinaryDiffEq.solve(sprob, solver, saveat=tspan,
#                              save_everystep=false, dense=false,
#                              reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
#        return ssol.u
#    else
#        error("Invalid format of odefun")
#    end
# end
#
# function flow(odefun::Function,u0::Tensors.Vec{dim,Float64},args...;kwargs...) where dim
#     return flow(odefun,SA.SVector{dim}(u0),args...;kwargs...)
# end
#
# function flow(rhs::Function,u0::AbstractVector{Float64},args...;kwargs...)
#     if length(u0) == 2
#         return flow(rhs,SA.SVector{2}(u0[1],u0[2]),args...;kwargs...)
#     elseif length(u0) == 3
#         return flow(rhs,SA.SVector{3}(u0[1],u0[2],u0[3]),args...;kwargs...)
#     else
#         error("length(u0) ∉ [2,3]")
#     end
# end

function flow(
            odefun::Function,
            u0::AbstractVector{T},
            tspan::AbstractVector{S};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            #ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T <: Real, S <: Real}
    # if needed, add callback to ODEProblems
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
    num_args = DiffEqBase.numargs(odefun)
    dim = length(u0)
    if num_args == 3
        if dim == 2
            _flow(Val{false}, Val{2}, odefun, SA.SVector{2,T}(u0[1], u0[2]), tspan;
                    tolerance = tolerance, p = p, solver = solver,
                    #ctx_for_boundscheck=nothing,
                    force_dtmin=force_dtmin)
        elseif dim == 3
            _flow(Val{false}, Val{3}, odefun, SA.SVector{3,T}(u0[1], u0[2], u0[3]), tspan;
                    tolerance = tolerance, p = p, solver = solver,
                    #ctx_for_boundscheck=nothing,
                    force_dtmin=force_dtmin)
        else
            error("length(u0) ∉ [2,3]")
        end
    elseif num_args == 4
        if dim == 2
            _flow(Val{true}, Val{2}, odefun, Vector{T}(u0), tspan;
                    tolerance = tolerance, p = p, solver = solver,
                    #ctx_for_boundscheck=nothing,
                    force_dtmin=force_dtmin)
        elseif dim == 3
            _flow(Val{true}, Val{3}, odefun, Vector{T}(u0), tspan;
                    tolerance = tolerance, p = p, solver = solver,
                    #ctx_for_boundscheck=nothing,
                    force_dtmin=force_dtmin)
        else
            error("length(u0) ∉ [2,3]")
        end
    else
        error("vector field has wrong number of arguments, must be 3 or 4")
    end
end

function _flow(
            ::Type{Val{false}},
            ::Type{Val{2}},
            odefun::Function,
            u0::SA.SVector{2,T},
            tspan::AbstractVector{S};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            #ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T <: Real, S <: Real}
    prob = OrdinaryDiffEq.ODEProblem(odefun, u0, (tspan[1],tspan[end]), p)
    sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                          save_everystep=false, dense=false,
                          reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
    return sol.u
end

function _flow(
            ::Type{Val{false}},
            ::Type{Val{3}},
            odefun::Function,
            u0::SA.SVector{3,T},
            tspan::AbstractVector{S};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            #ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T <: Real, S <: Real}
    prob = OrdinaryDiffEq.ODEProblem(odefun, u0, (tspan[1],tspan[end]), p)
    sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                          save_everystep=false, dense=false,
                          reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
    return sol.u
end

function _flow(
            ::Type{Val{true}},
            ::Type{Val{2}},
            odefun::Function,
            u0::AbstractVector{T},
            tspan::AbstractVector{S};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            #ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T <: Real, S <: Real}
    prob = OrdinaryDiffEq.ODEProblem(odefun, u0, (tspan[1],tspan[end]), p)
    sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                          save_everystep=false, dense=false,
                          reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
    return sol.u
end

function _flow(
            ::Type{Val{false}},
            ::Type{Val{3}},
            odefun::Function,
            u0::AbstractVector{T},
            tspan::AbstractVector{S};
            tolerance = default_tolerance,
            p = nothing,
            solver = default_solver,
            #ctx_for_boundscheck=nothing,
            force_dtmin=false
        ) where {T <: Real, S <: Real}
    prob = OrdinaryDiffEq.ODEProblem(odefun, u0, (tspan[1],tspan[end]), p)
    sol = OrdinaryDiffEq.solve(prob, solver, saveat=tspan,
                          save_everystep=false, dense=false,
                          reltol=tolerance, abstol=tolerance,force_dtmin=force_dtmin)
    return sol.u
end

"""
    parallel_flow(flow_fun,P) -> Array

Apply the `flow_fun` to each element in `P` in parallel, if possible. Returns
a 2D array with dimensions ((space dim x no. of time instances) x no. of
trajectories), in which each column corresponds to a concatenated trajectory,
i.e., represented in delay coordinates.
"""
function parallel_flow(flow_fun,P::AbstractArray{S}) where S <: AbstractArray
    dim::Int = length(P[1])
    T = eltype(S)
    dummy = flow_fun(P[1])
    q::Int = length(dummy)

    sol_shared = SharedArray{T}(dim*q,length(P));
    @inbounds @sync @parallel for index in eachindex(P)
        # @async begin
            u = flow_fun(P[index])
            for t=1:q, d=1:dim
                sol_shared[(t-1)*dim+d:t*dim,index] = u[t][d]
            end
        # end
    end
    # sol = Array{Array{Float64,2}}(length(P))
    # for index in eachindex(P)
    #     sol[index] = sol_shared[:,:,index]
    # end
    return collect(sol_shared)
end

"""
    linearized_flow(odefun, x, tspan,δ; ...) -> Vector{Tensor{2,2}}

Calculate derivative of flow map by finite differences. Return time-resolved
linearized flow maps.
"""
# @inline function linearized_flow(
#             odefun::Function,
#             x::SA.SVector{2,T},
#             tspan::AbstractVector{Float64},
#             δ::Float64;
#             tolerance=default_tolerance,
#             solver=default_solver,
#             p=nothing,
#             kwargs...
#         )::Vector{Tensors.Tensor{2,2,T,4}} where {T <: Real}
#
#     num_tsteps = length(tspan)
#     num_args = DiffEqBase.numargs(odefun)
#     if num_args == 4
#         dx = [δ, zero(δ)];
#         dy = [zero(δ), δ];
#
#         #In order to solve only one ODE, write all the initial values
#         #one after the other in one big vector
#         stencil::Vector{T} = zeros(T, 8)
#         @inbounds stencil[1:2] .= x .+ dx
#         @inbounds stencil[3:4] .= x .+ dy
#         @inbounds stencil[5:6] .= x .- dx
#         @inbounds stencil[7:8] .= x .- dy
#
#         rhs = (du,u,p,t) -> arraymap!(du,u,p,t,odefun,4,2)
#         prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
#         sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
#         return map(s->Tensors.Tensor{2,2}((s[1:4] - s[5:8])/2δ), sol)
#     elseif num_args == 3
#         #In order to solve only one ODE, write all the initial values
#         #one after the other in one big vector
#         sstencil::SA.SVector{8,Float64} = SA.SVector{8}(x[1] + δ, x[2], x[1],x[2] + δ, x[1] - δ, x[2], x[1],x[2] - δ)
#         srhs = (u,p,t) -> arraymap2(u,p,t,odefun)
#         sprob = OrdinaryDiffEq.ODEProblem(srhs,sstencil,(tspan[1],tspan[end]),p)
#         ssol = OrdinaryDiffEq.solve(sprob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
#         return map(s->Tensors.Tensor{2,2}((s[1:4] - s[5:8])/2δ), ssol)
#     else
#         error("odefun has invalid number of arguments")
#     end
# end
#
# @inline function linearized_flow(
#             odefun::Function,
#             x::SA.SVector{3,T},
#             tspan::AbstractVector{Float64},
#             δ::Float64;
#             tolerance=default_tolerance,
#             solver=default_solver,
#             p=nothing,
#             kwargs...
#         )::Vector{Tensors.Tensor{2,3,T,9}} where {T <: Real}
#
#     num_tsteps = length(tspan)
#     num_args = DiffEqBase.numargs(odefun)
#     if num_args == 4
#         dx = [δ, zero(δ), zero(δ)]
#         dy = [zero(δ), δ, zero(δ)]
#         dz = [zero(δ), zero(δ), δ]
#
#         #In order to solve only one ODE, write all the initial values
#         #one after the other in one big vector
#         stencil3::Vector{T} = zeros(T, 18)
#         @inbounds stencil3[1:3] .= x .+ dx
#         @inbounds stencil3[4:6] .= x .+ dy
#         @inbounds stencil3[7:9] .= x .+ dz
#         @inbounds stencil3[10:12] .= x .- dx
#         @inbounds stencil3[13:15] .= x .- dy
#         @inbounds stencil3[16:18] .= x .- dz
#
#         rhs = (du,u,p,t) -> arraymap!(du,u,p,t,odefun,6,3)
#         prob = OrdinaryDiffEq.ODEProblem(rhs,stencil3,(tspan[1],tspan[end]),p)
#         sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
#         return map(s->Tensors.Tensor{2,3}((s[1:9] - s[10:18])/2δ), sol)
#     elseif num_args == 3
#         #In order to solve only one ODE, write all the initial values
#         #one after the other in one big vector
#         sstencil::StaticArrays.SVector{18,Float64} = SA.SVector{18,T}(
#                 x[1] + δ, x[2], x[3],
#                 x[1], x[2] + δ, x[3],
#                 x[1], x[2], x[3] + δ,
#                 x[1] - δ, x[2], x[3],
#                 x[1], x[2] - δ, x[3],
#                 x[1], x[2], x[3] - δ
#                 )
#         srhs = (u,p,t) -> arraymap3(u,p,t,odefun)
#         sprob = OrdinaryDiffEq.ODEProblem(srhs,sstencil,(tspan[1],tspan[end]),p)
#         ssol = OrdinaryDiffEq.solve(sprob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
#         return map(s->Tensors.Tensor{2,3}((s[1:9] - s[10:18])/2δ), ssol)
#     else
#         error("odefun has invalid number of arguments")
#     end
# end
#
# @inline function linearized_flow(odefun::Function, u::Tensors.Vec{2,T},args...;kwargs...)::Vector{Tensors.Tensor{2,2,T,4}} where {T<:Real}
#     return linearized_flow(odefun, SA.SVector{2,T}(u),args...;kwargs...)
# end
#
# @inline function linearized_flow(odefun::Function, u::Tensors.Vec{3,T},args...;kwargs...)::Vector{Tensors.Tensor{2,3,T,9}} where {T<:Real}
#     return linearized_flow(odefun, SA.SVector{3,T}(u),args...;kwargs...)
# end
#
# @inline function linearized_flow(odefun::Function, u::AbstractVector{T},args...;kwargs...) where {T<:Real}
#     if length(u) == 2
#         return linearized_flow(odefun, SA.SVector{2,T}(u[1],u[2]),args...;kwargs...)
#     elseif length(u) == 3
#         return linearized_flow(odefun, SA.SVector{3,T}(u[1],u[2],u[3]),args...;kwargs...)
#     else
#         error("length(u) ∉ [2,3]")
#     end
# end

function linearized_flow(
                        odefun,
                        u0::AbstractVector{T},
                        tspan::AbstractVector{S},
                        δ::Real;
                        tolerance=default_tolerance,
                        solver=default_solver,
                        p=nothing
                    ) where {T <: Real, S <: Real}

    num_args = DiffEqBase.numargs(odefun)
    dim = length(u0)
    if num_args == 3
        if iszero(δ)
            if dim == 2
                _linearized_flow(Val{false}, Val{2}, odefun, u0, tspan;
                                tolerance = tolerance, p = p, solver = solver)
            elseif dim == 3
                _linearized_flow(Val{false}, Val{3}, odefun, u0, tspan;
                                tolerance = tolerance, p = p, solver = solver)
            else
                error("length(u0) ∉ [2,3]")
            end
        else
            if dim == 2
                _linearized_flow(Val{false}, Val{2}, odefun, u0, tspan, δ;
                                tolerance = tolerance, p = p, solver = solver)
            elseif dim == 3
                _linearized_flow(Val{false}, Val{3}, odefun, u0, tspan, δ;
                                tolerance = tolerance, p = p, solver = solver)
            else
                error("length(u0) ∉ [2,3]")
            end
        end
    elseif num_args == 4
        if iszero(δ)
            if dim == 2
                _linearized_flow(Val{true}, Val{2}, odefun, u0, tspan;
                                tolerance = tolerance, p = p, solver = solver)
            elseif dim == 3
                _linearized_flow(Val{true}, Val{3}, odefun, u0, tspan;
                                tolerance = tolerance, p = p, solver = solver)
            else
                error("length(u0) ∉ [2,3]")
            end
        else
            if dim == 2
                _linearized_flow(Val{true}, Val{2}, odefun, u0, tspan, δ;
                                tolerance = tolerance, p = p, solver = solver)
            elseif dim == 3
                _linearized_flow(Val{true}, Val{3}, odefun, u0, tspan, δ;
                                tolerance = tolerance, p = p, solver = solver)
            else
                error("length(u0) ∉ [2,3]")
            end
        end
    else
        error("vector field has wrong number of arguments, must be 3 or 4")
    end
end

function _linearized_flow(
                        ::Type{Val{false}},
                        ::Type{Val{2}},
                        odefun,
                        u0::AbstractVector{T},
                        tspan::AbstractVector{S},
                        δ::Real;
                        tolerance=default_tolerance,
                        solver=default_solver,
                        p=nothing
                    )::Vector{Tensors.Tensor{2,2}} where {T <: Real, S <: Real}

    stencil = SA.SVector{8}(u0[1]+δ, u0[2]  ,
                             u0[1]  , u0[2]+δ,
                             u0[1]-δ, u0[2]  ,
                             u0[1]  , u0[2]-δ)
    rhs = (u,p,t) -> arraymap2(u,p,t,odefun)
    prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
    sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,
                                save_everystep=false,
                                dense=false,
                                reltol=tolerance,abstol=tolerance).u
    return map(s -> Tensors.Tensor{2,2}((s[1:4] - s[5:8]) / 2δ), sol)
end

function _linearized_flow(
                        ::Type{Val{false}},
                        ::Type{Val{3}},
                        odefun,
                        u0::AbstractVector{T},
                        tspan::AbstractVector{S},
                        δ::Real;
                        tolerance=default_tolerance,
                        solver=default_solver,
                        p=nothing
                    )::Vector{Tensors.Tensor{2,3}} where {T <: Real, S <: Real}

    stencil = SA.SVector{18}(
            u0[1] + δ, u0[2]    , u0[3]    ,
            u0[1]    , u0[2] + δ, u0[3]    ,
            u0[1]    , u0[2]    , u0[3] + δ,
            u0[1] - δ, u0[2]    , u0[3]    ,
            u0[1]    , u0[2] - δ, u0[3]    ,
            u0[1]    , u0[2]    , u0[3] - δ)
    rhs = (u,p,t) -> arraymap3(u,p,t,odefun)
    prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
    sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,
                                save_everystep=false,dense=false,
                                reltol=tolerance,abstol=tolerance).u
    return map(s -> Tensors.Tensor{2,3}((s[1:9] - s[10:18]) / 2δ), sol)
end

function _linearized_flow(
                        ::Type{Val{true}},
                        ::Type{Val{2}},
                        odefun,
                        u0::AbstractVector{T},
                        tspan::AbstractVector{S},
                        δ::Real;
                        tolerance=default_tolerance,
                        solver=default_solver,
                        p=nothing
                    )::Vector{Tensors.Tensor{2,2}} where {T <: Real, S <: Real}

    stencil = [u0[1]+δ, u0[2], u0[1], u0[2]+δ, u0[1]-δ, u0[2], u0[1], u0[2]-δ]
    rhs = (du,u,p,t) -> arraymap!(du,u,p,t,odefun,4,2)
    prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
    sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,
                                save_everystep=false,
                                dense=false,
                                reltol=tolerance,abstol=tolerance).u
    return map(s -> Tensors.Tensor{2,2}((s[1:4] - s[5:8]) / 2δ), sol)
end

function _linearized_flow(
                        ::Type{Val{true}},
                        ::Type{Val{3}},
                        odefun,
                        u0::AbstractVector{T},
                        tspan::AbstractVector{S},
                        δ::Real;
                        tolerance=default_tolerance,
                        solver=default_solver,
                        p=nothing
                    )::Vector{Tensors.Tensor{2,3}} where {T <: Real, S <: Real}

    stencil = [
                u0[1] + δ, u0[2]    , u0[3]    ,
                u0[1]    , u0[2] + δ, u0[3]    ,
                u0[1]    , u0[2]    , u0[3] + δ,
                u0[1] - δ, u0[2]    , u0[3]    ,
                u0[1]    , u0[2] - δ, u0[3]    ,
                u0[1]    , u0[2]    , u0[3] - δ
               ]
    rhs = (du, u, p, t) -> arraymap!(du, u, p, t, odefun, 6, 3)
    prob = OrdinaryDiffEq.ODEProblem(rhs,stencil,(tspan[1],tspan[end]),p)
    sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,
                                save_everystep=false,dense=false,
                                reltol=tolerance,abstol=tolerance).u
    return map(s -> Tensors.Tensor{2,3}((s[1:9] - s[10:18]) / 2δ), sol)
end

"""
    parallel_tensor(tensor_fun,P) -> Array{SymmetricTensor}

Computes a tensor field via `tensor_fun` for each element of `P`, which is an
array of vectors. `tensor_fun` is a function that takes initial conditions as
input and returns a *symmetric* tensor. The final tensor field array has the
same size as `P`.
"""
function parallel_tensor(tensor_fun,P::AbstractArray{T,N}) where T where N

    dim = length(P[1])
    T_shared = SharedArray{Float64}(div(dim*(dim+1), 2), length(P))
    idxs = tril(ones(Bool,dim,dim))
    @everywhere @eval idxs = $idxs
    @sync @parallel for index in eachindex(P)
        T_shared[:,index] = tensor_fun(P[index])[idxs]
    end

    Tfield = Array{Tensors.SymmetricTensor{2,dim,eltype(T),div(dim*(dim+1),2)}}(size(P))
    for index in eachindex(P)
        Tfield[index] = Tensors.SymmetricTensor{2,dim}(T_shared[:,index])
    end
    return Tfield
end

"""
    mean_diff_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor

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
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            kwargs...
        ) where {T <: Real, S <: Real}
    return mean(Tensors.dott.(inv.(linearized_flow(odefun, u, tspan, δ; kwargs...))))
end

"""
    CG_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor

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
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            kwargs...
        ) where {T <: Real, S <: Real}
    return Tensors.tdot(linearized_flow(odefun, u, [tspan[1],tspan[end]], δ; kwargs...)[end])
end

"""
    pullback_tensors(odefun, u, tspan, δ; D, kwargs...) -> Tuple(Vector{SymmetricTensor},Vector{SymmetricTensor})

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
            odefun,
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            D::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real, S <: Real, dim, N}

    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    MT = [Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
    DF .= inv.(DF)
    DT = [Tensors.symmetric(df ⋅ D ⋅ transpose(df)) for df in DF]
    return MT, DT # MT is pullback metric tensor, DT is pullback diffusion tensor
end

"""
    pullback_metric_tensor(odefun, u, tspan, δ; G, kwargs...) -> Vector{SymmetricTensor}

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
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            G::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real, S <: Real, dim, N}

    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    return [Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
end

"""
    pullback_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...) -> Vector{SymmetricTensor}

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
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            D::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real, S <: Real, dim, N}

    DFinv = inv.(linearized_flow(odefun, u, tspan, δ; kwargs...))
    return [Tensors.symmetric(df ⋅ D ⋅ transpose(df)) for df in DFinv]
end

# TODO: this function likely doesn't work, uses an unsupported give_back_position keyword argument
function pullback_diffusion_tensor_function(
            odefun,
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64,
            Dfun;
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real, S <: Real}

    DF, pos = iszero(δ) ?
        linearized_flow(odefun,u,tspan, p=p,tolerance=tolerance, solver=solver,give_back_position=true) :
        linearized_flow(odefun,u,tspan, δ, p=p,tolerance=tolerance, solver=solver,give_back_position=true)

    DF .= inv.(DF)
    tlen = length(tspan)
    result = map(eachindex(DF, pos)) do i
        Tensors.symmetric(DF[i] ⋅ Dfun(pos[i]) ⋅ transpose(DF[i]))
    end
    return result
end

"""
    pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...) -> Vector{SymmetricTensor}

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
                tspan::AbstractVector{S},
                δ::Float64;
                B::Tensors.Tensor{2,dim,T,N}=one(Tensors.Tensor{2,2,T,4}),
                kwargs...
            ) where {T <: Real, S <: Real, dim, N}

    DFinv = inv.(linearized_flow(odefun, u, tspan, δ; kwargs...))
    return [df ⋅ B for df in DFinv]
end

"""
    av_weighted_CG_tensor(odefun, u, tspan, δ; G, kwargs...) -> SymmetricTensor

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
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            D::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
            p = nothing,
            tolerance = 1.e-3,
            solver = OrdinaryDiffEq.BS5()
        ) where {T <: Real, S <: Real, dim, N}

    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    return det(D) * mean([Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF])
end

function met2deg(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2,T,4}, [1/cos(deg2rad(u[2])), one(T)])
end

function deg2met(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2,T,4}, [cos(deg2rad(u[2])), one(T)])
end

function pullback_tensors_geo(
            odefun,
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::T;
            D::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
            tolerance::Float64=1e-3,
            p=nothing,
            solver=OrdinaryDiffEq.BS5()
        ) where {T<:Real, S <: Real, dim, N}

    G = inv(D)
    met2deg_init = met2deg(u)
    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    DF = linearized_flow(odefun, u, tspan, δ; p=p,tolerance=tolerance, solver=solver)
    PBmet = [deg2met(sol[i]) ⋅ DF[i] ⋅ met2deg_init for i in eachindex(DF,sol)]
    PBdiff = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return [Tensors.symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PBmet], [Tensors.symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PBdiff]
end

function pullback_metric_tensor_geo(
            odefun,
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::T;
            G::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
            tolerance::Float64=1e-3,
            p=nothing,
            solver=OrdinaryDiffEq.BS5()
        ) where {T<:Real, S <: Real, dim, N}

    met2deg_init = met2deg(u)
    sol = flow(odefun, u, tspan, solver=solver, tolerance=tolerance, p=p)
    DF = linearized_flow(odefun, u, tspan, δ; p=p, tolerance=tolerance, solver=solver)
    PB = [deg2met(sol[i]) ⋅ DF[i] ⋅ met2deg_init for i in eachindex(DF, sol)]
    return [Tensors.symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PB]
end

function pullback_diffusion_tensor_geo(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{S},
                δ::T;
                D::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
                tolerance::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real, S <: Real, dim, N}

    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    DF = linearized_flow(odefun, u, tspan, δ; p=p,tolerance=tolerance, solver=solver)
    PB = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return [Tensors.symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PB]
end

function pullback_SDE_diffusion_tensor_geo(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{S},
                δ::T;
                D::Tensors.SymmetricTensor{2,dim,T,N}=one(Tensors.SymmetricTensor{2,2,T,3}),
                tolerance::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real, S <: Real, dim, N}

    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    DF = linearized_flow(odefun, u, tspan, δ; p=p,tolerance=tolerance, solver=solver)
    B = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return B
end
