#Functions for pulling back Tensors

const default_tolerance = 1e-3
const default_solver = OrdinaryDiffEq.BS5()

struct Trajectory{dim,T}
    F::Vector{SVector{dim,T}}
    DF::Vector{Tensor{2,dim,T}}
end

struct LinFlowMap{dim,T,N}
    p::Array{Trajectory{dim,T},N}
end


"""
    flow(odefun,  u0, tspan; p, solver, tolerance, force_dtmin) -> Vector{Vector}

Solve the ODE with right hand side given by `odefun` and initial value `u0` over
the time interval `tspan`, evaluated at each element of `tspan`.

## Keyword arguments
   * `p`: parameter that is passed to `odefun`, e.g., in [`interp_rhs`](@ref);
   * `solver=OrdinaryDiffEq.BS5()`: ODE solver;
   * `tolerance=1e-3`: relative and absolute tolerance for ODE integration;
   * `force_dtmin=false`: force the ODE solver to step forward with `dtmin`, even
     if the adaptive scheme would reject the step.

# Example
```
julia> f = u -> flow(bickleyJet, u, range(0., stop=100, length=21))
```
"""
@inline function flow(odefun, u0::AbstractVector{T}, tspan::AbstractVector{S}; kwargs...) where {T <: Real, S <: Real}
    flow(OrdinaryDiffEq.ODEFunction(odefun), u0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{true}, u0::AbstractVector{T},
                tspan::AbstractVector{S}; kwargs...) where {T <: Real, S <: Real}
    v0 = convert(Vector{T}, u0)::Vector{T}
    _flow(odefun, v0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{true}, u0::Vector{T},
                tspan::AbstractVector{S}; kwargs...) where {T <: Real, S <: Real}
    _flow(odefun, u0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{false}, u0::AbstractVector{T},
                tspan::AbstractVector{S}; kwargs...) where {T <: Real, S <: Real}
    v0 = convert(SVector{length(u0), T}, u0)
    _flow(odefun, v0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{false}, u0::SVector{dim,T},
                tspan::AbstractVector{S}; kwargs...) where {dim, T <: Real, S <: Real}
    _flow(odefun, u0, tspan; kwargs...)
end
@inline function _flow(odefun, u0::T, tspan;
                p=nothing,
                solver=default_solver,
                tolerance=default_tolerance,
                #ctx_for_boundscheck=nothing,
                force_dtmin=false)::Vector{T} where {T}

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
    prob = OrdinaryDiffEq.ODEProblem(odefun, u0, (tspan[1], tspan[end]), p)
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

    sol_shared = SharedArray{T,2}(dim*q, length(P))
    @inbounds @sync Distributed.@distributed for index in eachindex(P)
            u = flow_fun(P[index])
            for t=1:q
                sol_shared[(t-1)*dim+1:t*dim,index] = u[t]
            end
    end
    # sol = Array{Array{Float64,2}}(length(P))
    # for index in eachindex(P)
    #     sol[index] = sol_shared[:,:,index]
    # end
    return collect(sol_shared)
end

"""
    linearized_flow(odefun, x, tspan,δ; ...) -> Vector{Tensor{2,2}}

Calculate derivative of flow map by finite differences if δ != 0.
If δ==0, attempts to solve variational equation (odefun is assumed to be the rhs of
variational equation in this case).
Return time-resolved linearized flow maps.
"""
function linearized_flow(
        odefun, x::AbstractVector{T}, tspan, δ; kwargs...
    ) where {T <: Real}
    dim = length(x)
    !(dim ∈ (2, 3)) && error("length(u) ∉ [2,3]")
    linearized_flow(OrdinaryDiffEq.ODEFunction(odefun), convert(SVector{dim, T}, x), tspan, δ; kwargs...)
end
function linearized_flow(
            odefun::OrdinaryDiffEq.ODEFunction{iip},
            x::SVector{2,T},
            tspan::AbstractVector{Float64},
            δ::Real;
            tolerance=default_tolerance,
            solver=default_solver,
            p=nothing
        )::Tuple{Vector{SVector{2,T}}, Vector{Tensor{2,2,T,4}}} where {iip, T <: Real}

    if iip
        if δ != 0 # use finite differencing
            stencil::Vector{T} = [x[1], x[2],
                                x[1] + δ, x[2], x[1], x[2] + δ,
                                x[1] - δ, x[2], x[1], x[2] - δ]
            rhs = (du, u, p, t) -> arraymap!(du, u, p, t, odefun, 5, 2)
            sol = _flow(rhs, stencil, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{2}(s[1], s[2]), sol),
                    map(s -> Tensor{2,2}((s[3:6] - s[7:10]) / 2δ), sol))
        else # δ = 0
            u0 = [x[1] one(T) zero(T);
                  x[2] zero(T) one(T)]
            evsol = _flow(odefun, u0, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{2}(s[1,1], s[2,1]), evsol),
                    map(s -> Tensor{2,2}((s[1,2], s[2,2], s[1,3], s[2,3])), evsol))
        end # δ
    else # !iip
        if δ != 0 # use finite differencing
            sstencil::SVector{10,T} = SVector{10}(x[1], x[2],
                                                    x[1] + δ, x[2], x[1], x[2] + δ,
                                                    x[1] - δ, x[2], x[1], x[2] - δ)
            srhs = (u, p, t) -> arraymap2(u, p, t, odefun)
            ssol = _flow(srhs, sstencil, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{2}(s[1], s[2]), ssol),
                    map(s -> Tensor{2,2}((s[3:6] - s[7:10]) / 2δ), ssol))
        else
            v0 = @SMatrix [x[1] one(T) zero(T);
                           x[2] zero(T) one(T)]
            eqvsol = _flow(odefun, v0, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{2}(s[1,1], s[2,1]), eqvsol),
                    map(s -> Tensor{2,2}((s[1,2], s[2,2], s[1,3], s[2,3])), eqvsol))
        end # δ
    end # iip
end
function linearized_flow(
            odefun::OrdinaryDiffEq.ODEFunction{iip},
            x::SVector{3,T},
            tspan::AbstractVector{Float64},
            δ::Real;
            tolerance=default_tolerance,
            solver=default_solver,
            p=nothing
        )::Tuple{Vector{SVector{3,T}}, Vector{Tensor{2,3,T,9}}} where {iip, T <: Real}

    if iip
        if δ != 0 # use finite differencing
            stencil::Vector{T} = [x[1], x[2], x[3],
                              x[1] + δ, x[2], x[3],
                              x[1], x[2] + δ, x[3],
                              x[1], x[2], x[3] + δ,
                              x[1] - δ, x[2], x[3],
                              x[1], x[2] - δ, x[3],
                              x[1], x[2], x[3] - δ]
            rhs = (du, u, p, t) -> arraymap!(du, u, p, t, odefun, 7, 3)
            sol = _flow(rhs, stencil, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{3}(s[1], s[2], s[3]), sol),
                    map(s -> Tensor{2,3}((s[4:12] - s[13:21]) / 2δ), sol))
        else # δ = 0
            V0 = [x[1] one(T) zero(T) zero(T);
                  x[2] zero(T) one(T) zero(T);
                  x[3] zero(T) zero(T) one(T)]
            eqvsol = _flow(odefun, V0, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{3}(s[1,1], s[2,1], s[3,1]), eqvsol),
                    map(s -> Tensor{2,3}((s[1,2], s[2,2], s[3,2],
                                          s[1,3], s[2,3], s[3,3],
                                          s[1,4], s[2,4], s[3,4])), eqvsol))
        end # δ
    else # !iip
        if δ != 0 # use finite differencing
            sstencil::SVector{21,T} = SVector{21,T}(x[1], x[2], x[3],
                              x[1] + δ, x[2], x[3],
                              x[1], x[2] + δ, x[3],
                              x[1], x[2], x[3] + δ,
                              x[1] - δ, x[2], x[3],
                              x[1], x[2] - δ, x[3],
                              x[1], x[2], x[3] - δ)
            srhs = (u, p, t) -> arraymap3(u, p, t, odefun)
            ssol = _flow(srhs, sstencil, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{3}(s[1], s[2], s[3]), ssol),
                    map(s -> Tensor{2,3}((s[4:12] - s[13:21]) / 2δ), ssol))
        else # δ = 0
            u0 = @SMatrix [x[1] one(T) zero(T) zero(T);
                           x[2] zero(T) one(T) zero(T);
                           x[3] zero(T) zero(T) one(T)]
            evsol = _flow(odefun, u0, tspan; tolerance=tolerance, solver=solver, p=p)
            return (map(s -> SVector{3}(s[1,1], s[2,1], s[3,1]), evsol),
                    map(s -> Tensor{2,3}((s[1,2], s[2,2], s[3,2],
                                          s[1,3], s[2,3], s[3,3],
                                          s[1,4], s[2,4], s[3,4])), evsol))
        end # δ
    end # iip
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
    return mean(Tensors.dott.(inv.(linearized_flow(odefun, u, tspan, δ; kwargs...)[2])))
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
    return Tensors.tdot(linearized_flow(odefun, u, [tspan[1],tspan[end]], δ; kwargs...)[2][end])
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
            D::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real, S <: Real, dim, N}

    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
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
            G::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real, S <: Real, dim, N}

    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
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
            D::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real, S <: Real, dim, N}

    DFinv = inv.(linearized_flow(odefun, u, tspan, δ; kwargs...)[2])
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
    pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; B, kwargs...) -> Vector{SymmetricTensor}

Returns the time-resolved pullback tensors of the diffusion tensor in SDEs.
Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `B`: (constant) SDE tensor
   * `kwargs...` are passed through to `linearized_flow`
"""
@inline function pullback_SDE_diffusion_tensor(
                odefun,
                u::AbstractVector{T},
                tspan::AbstractVector{S},
                δ::Float64;
                B::Tensor{2,dim,T,N}=one(Tensor{2,2,T,4}),
                kwargs...
            ) where {T <: Real, S <: Real, dim, N}

    DFinv = inv.(linearized_flow(odefun, u, tspan, δ; kwargs...)[2])
    return [df ⋅ B for df in DFinv]
end

"""
    av_weighted_CG_tensor(odefun, u, tspan, δ; D, kwargs...) -> SymmetricTensor

Returns the transport tensor of a trajectory, aka  time-averaged,
di ffusivity-structure-weighted version of the classic right Cauchy–Green strain
tensor. Derivatives are computed with finite differences.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences
   * `D`: (constant) diffusion tensor
   * `kwargs...` are passed through to `linearized_flow`
"""
function av_weighted_CG_tensor(
            odefun,
            u::AbstractVector{T},
            tspan::AbstractVector{S},
            δ::Float64;
            D::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
            kwargs...
        ) where {T <: Real, S <: Real, dim, N}

    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
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
            D::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
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
            G::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
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
                D::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
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
                D::SymmetricTensor{2,dim,T,N}=one(SymmetricTensor{2,2,T,3}),
                tolerance::Float64=1e-3,
                p=nothing,
                solver=OrdinaryDiffEq.BS5()
            ) where {T<:Real, S <: Real, dim, N}

    sol = flow(odefun,u,tspan,solver=solver,tolerance=tolerance,p=p)
    DF = linearized_flow(odefun, u, tspan, δ; p=p,tolerance=tolerance, solver=solver)
    B = [inv(deg2met(sol[i]) ⋅ DF[i]) for i in eachindex(DF,sol)]
    return B
end
