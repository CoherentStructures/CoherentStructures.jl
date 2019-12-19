#Functions for pulling back Tensors

const default_tolerance = 1e-3
const default_solver = OrdinaryDiffEq.BS5()

# TODO: exploit throughout
struct Trajectory{dim,ts,T,N}
    F::SVector{ts,SVector{dim,T}}
    DF::SVector{ts,Tensor{2,dim,T,N}}
end

"""
    flow(odefun,  u0, tspan; p, solver, tolerance, force_dtmin)

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
@inline function flow(odefun, u0::AbstractVector, tspan::AbstractVector; kwargs...)
    flow(OrdinaryDiffEq.ODEFunction(odefun), u0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{true}, u0::AbstractVector{T},
                tspan::AbstractVector; kwargs...) where {T<:Real}
    v0::Vector{T} = convert(Vector{T}, u0)
    _flow(odefun, v0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{true}, u0::Vector,
                tspan::AbstractVector; kwargs...)
    _flow(odefun, u0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{false}, u0::AbstractVector{T},
                tspan::AbstractVector; kwargs...) where {T<:Real}
    v0 = convert(SVector{length(u0), T}, u0)
    _flow(odefun, v0, tspan; kwargs...)
end
@inline function flow(odefun::OrdinaryDiffEq.ODEFunction{false}, u0::SVector{dim},
                tspan::AbstractVector; kwargs...) where {dim}
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
    linearized_flow(odefun, x, tspan,δ; ...) -> Vector{Tensor{2,2}}

Calculate derivative of flow map by finite differences if δ != 0.
If δ==0, attempts to solve variational equation (odefun is assumed to be the rhs of
variational equation in this case).
Return time-resolved linearized flow maps.
"""
function linearized_flow(odefun, x::AbstractVector, tspan, δ; kwargs...)
    dim = length(x)
    !(dim ∈ (2, 3)) && error("length(u) ∉ [2,3]")
    linearized_flow(OrdinaryDiffEq.ODEFunction(odefun), convert(SVector{dim}, x), tspan, δ; kwargs...)
end
function linearized_flow(
            odefun::OrdinaryDiffEq.ODEFunction{iip},
            x::SVector{2,T},
            tspan::AbstractVector{Float64},
            δ::Real;
            tolerance=default_tolerance,
            solver=default_solver,
            p=nothing) where {iip,T}#::Tuple{Vector{SVector{2,T}}, Vector{Tensor{2,2,T,4}}} where {iip, T <: Real}

    if iip
        if δ != 0 # use finite differencing
            stencil = [x[1], x[2],
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
            sstencil = SVector{10}(x[1], x[2],
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
        ) where {iip,T}#::Tuple{Vector{SVector{3,T}}, Vector{Tensor{2,3,T,9}}} where {iip, T <: Real}

    if iip
        if δ != 0 # use finite differencing
            stencil = [x[1], x[2], x[3],
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
            sstencil = SVector{21,T}(x[1], x[2], x[3],
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
function mean_diff_tensor(odefun, u::AbstractVector, tspan::AbstractVector, δ::Float64; kwargs...)
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
function CG_tensor(odefun, u::AbstractVector, tspan::AbstractVector, δ::Real; kwargs...)
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
function pullback_tensors(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        D::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
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
function pullback_metric_tensor(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        G::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
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
function pullback_diffusion_tensor(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        D::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
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
function pullback_SDE_diffusion_tensor(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
                B::Tensor{2,dim,T}=one(Tensor{2,2,T,4}), kwargs...) where {T,dim}
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
function av_weighted_CG_tensor(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        D::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
    return det(D) * mean([Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF])
end

function met2deg(u::AbstractVector{T}) where {T}
    diagm(Tensor{2,2,T,4}, [1/cos(deg2rad(u[2])), one(T)])
end

function deg2met(u::AbstractVector{T}) where {T}
    diagm(Tensor{2,2,T,4}, [cos(deg2rad(u[2])), one(T)])
end

function pullback_tensors_geo(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        D::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
    G = inv(D)
    met2deg_init = met2deg(u)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    PBmet = [deg2met(xi) ⋅ DFi ⋅ met2deg_init for (xi, DFi) in DF]
    PBdiff = [inv(deg2met(xi) ⋅ DFi) for (xi, DFi) in DF]
    return [Tensors.symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PBmet], [Tensors.symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PBdiff]
end

function pullback_metric_tensor_geo(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        G::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
    met2deg_init = met2deg(u)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    PB = [deg2met(xi) ⋅ DFi ⋅ met2deg_init for (xi, DFi) in DF]
    return [Tensors.symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PB]
end

function pullback_diffusion_tensor_geo(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        D::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    PB = [inv(deg2met(xi) ⋅ DFi) for (xi, DFi) in DF]
    return [Tensors.symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PB]
end

function pullback_SDE_diffusion_tensor_geo(odefun, u::AbstractVector{T}, tspan::AbstractVector, δ::Real;
        D::SymmetricTensor{2,dim,T}=one(SymmetricTensor{2,2,T,3}), kwargs...) where {T,dim}
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    B = [inv(deg2met(xi) ⋅ DFi) for (xi, DFi) in DF]
    return B
end
