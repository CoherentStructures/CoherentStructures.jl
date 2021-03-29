# Functions for pulling back tensors

const default_tolerance = 1e-3
const default_solver = ODE.BS5()

# TODO: exploit throughout
struct Trajectory{dim,ts,T,N}
    F::SVector{ts,SVector{dim,T}}
    DF::SVector{ts,Tensor{2,dim,T,N}}
end

"""
    flow(odefun, u0, tspan; p, solver, tolerance, force_dtmin)

Solve the ODE with right hand side given by `odefun` and initial value `u0` over
the time interval `tspan`, evaluated at each element of `tspan`.

## Keyword arguments
   * `p`: parameter that is passed to `odefun`, e.g., in [`interp_rhs`](@ref);
   * `solver=OrdinaryDiffEq.BS5()`: ODE solver;
   * `tolerance=1e-3`: relative and absolute tolerance for ODE integration;
   * `force_dtmin=false`: force the ODE solver to step forward with `dtmin`, even
     if the adaptive scheme would reject the step.

# Example
```julia
julia> f = u -> flow((u, p, t) -> vf(u, p, t), u, range(0., stop=100, length=21))
```
"""
@inline function flow(odefun, u0, tspan; kwargs...)
    return flow(ODE.ODEFunction(odefun), u0, tspan; kwargs...)
end
@inline function flow(odefun::ODE.ODEFunction{true}, u0::Union{Tuple{Vararg{T}},AbstractVecOrMat{T}}, tspan; kwargs...) where {T<:Real}
    return _flow(odefun, convert(Vector, u0), tspan; kwargs...)
end
@inline function flow(odefun::ODE.ODEFunction{false}, u0::Union{Tuple{Vararg{T}},AbstractVecOrMat{T}}, tspan; kwargs...) where {T<:Real}
    return _flow(odefun, convert(SVector{length(u0)}, u0), tspan; kwargs...)
end
@inline function _flow(
    odefun::ODE.ODEFunction,
    u0::T,
    tspan;
    p = nothing,
    solver = default_solver,
    tolerance = default_tolerance,
    #ctx_for_boundscheck=nothing,
    force_dtmin = false,
    saveat = tspan,
)::Vector{T} where {T} # TODO: type assertion needed in Julia v1.0
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
    #   callback = ODE.CallbackSet(
    #           map(x-> ODE.DiscreteCallback(x,affect!),
    #       [leftSide,rightSide,topSide,bottomSide])...)
    #end
    prob = ODE.ODEProblem(odefun, u0, (first(tspan), last(tspan)), p)
    sol = ODE.solve(
        prob,
        solver;
        saveat = saveat,
        save_everystep = false,
        dense = false,
        reltol = tolerance,
        abstol = tolerance,
        force_dtmin = force_dtmin,
    )
    return sol.u
end

"""
    linearized_flow(odefun, x, tspan, δ; kwargs...)

Calculate derivative of flow map by finite differencing (if `δ != 0`) or by
solving the variational equation (if `δ = 0`).

Return the time-resolved base trajectory and its associated linearized flow maps.
"""
@inline function linearized_flow(odefun, x, tspan, δ; kwargs...)
    dim = length(x)
    !(dim ∈ (2, 3)) && throw(ArgumentError("length(u) ∉ [2,3]"))
    return linearized_flow(
        ODE.ODEFunction(odefun),
        convert(SVector{dim}, x),
        tspan,
        δ;
        kwargs...,
    )
end
function linearized_flow(
    odefun::ODE.ODEFunction{iip},
    x::T,
    tspan,
    δ;
    tolerance=default_tolerance,
    solver=default_solver,
    p=nothing,
)::Tuple{Vector{T},Vector{<:Tensor{2,2}}} where {iip,T<:SVector{2}}
    @inbounds if iip
        if δ != 0 # use finite differencing
            stencil =
                [x[1], x[2], x[1] + δ, x[2], x[1], x[2] + δ, x[1] - δ, x[2], x[1], x[2] - δ]
            rhs = ODE.ODEFunction{iip}((du, u, p, t) -> arraymap!(du, u, p, t, odefun, 5, 2))
            sol = _flow(rhs, stencil, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{2}(s[1], s[2]), sol),
                map(s -> Tensor{2,2}((s[3:6] .- s[7:10]) ./ 2δ), sol),
            )
        else # δ = 0
            u0 = [
                x[1] 1 0
                x[2] 0 1
            ]
            evsol = _flow(odefun, u0, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{2}(s[1, 1], s[2, 1]), evsol),
                map(s -> Tensor{2,2}((s[1, 2], s[2, 2], s[1, 3], s[2, 3])), evsol),
            )
        end # δ
    else # !iip
        if δ != 0 # use finite differencing
            sstencil = SVector{10}(
                x[1],
                x[2],
                x[1] + δ,
                x[2],
                x[1],
                x[2] + δ,
                x[1] - δ,
                x[2],
                x[1],
                x[2] - δ,
            )
            srhs = ODE.ODEFunction{iip}((u, p, t) -> arraymap2(u, p, t, odefun))
            ssol =
                _flow(srhs, sstencil, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{2}(s[1], s[2]), ssol),
                map(s -> Tensor{2,2}((@views s[3:6] .- s[7:10]) ./ 2δ), ssol),
            )
        else
            v0 = SMatrix{2,3}(x[1], x[2], 1, 0, 0, 1)
            eqvsol = _flow(odefun, v0, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{2}(s[1, 1], s[2, 1]), eqvsol),
                map(s -> Tensor{2,2}((s[1, 2], s[2, 2], s[1, 3], s[2, 3])), eqvsol),
            )
        end # δ
    end # iip
end
function linearized_flow(
    odefun::ODE.ODEFunction{iip},
    x::T,
    tspan,
    δ;
    tolerance=default_tolerance,
    solver=default_solver,
    p=nothing,
)::Tuple{Vector{T},Vector{<:Tensor{2,3}}} where {iip,T<:SVector{3}}
    @inbounds if iip
        if δ != 0 # use finite differencing
            stencil = [
                x[1],
                x[2],
                x[3],
                x[1] + δ,
                x[2],
                x[3],
                x[1],
                x[2] + δ,
                x[3],
                x[1],
                x[2],
                x[3] + δ,
                x[1] - δ,
                x[2],
                x[3],
                x[1],
                x[2] - δ,
                x[3],
                x[1],
                x[2],
                x[3] - δ,
            ]
            rhs = ODE.ODEFunction{iip}((du, u, p, t) -> arraymap!(du, u, p, t, odefun, 7, 3))
            sol = _flow(rhs, stencil, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{3}(s[1], s[2], s[3]), sol),
                map(s -> Tensor{2,3}((@views s[4:12] .- s[13:21]) ./ 2δ), sol),
            )
        else # δ = 0
            V0 = [
                x[1] 1 0 0
                x[2] 0 1 0
                x[3] 0 0 1
            ]
            eqvsol = _flow(odefun, V0, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{3}(s[1, 1], s[2, 1], s[3, 1]), eqvsol),
                map(
                    s -> Tensor{2,3}((
                        s[1, 2],
                        s[2, 2],
                        s[3, 2],
                        s[1, 3],
                        s[2, 3],
                        s[3, 3],
                        s[1, 4],
                        s[2, 4],
                        s[3, 4],
                    )),
                    eqvsol,
                ),
            )
        end # δ
    else # !iip
        if δ != 0 # use finite differencing
            sstencil = SVector{21}(
                x[1],
                x[2],
                x[3],
                x[1] + δ,
                x[2],
                x[3],
                x[1],
                x[2] + δ,
                x[3],
                x[1],
                x[2],
                x[3] + δ,
                x[1] - δ,
                x[2],
                x[3],
                x[1],
                x[2] - δ,
                x[3],
                x[1],
                x[2],
                x[3] - δ,
            )
            srhs = ODE.ODEFunction{iip}((u, p, t) -> arraymap3(u, p, t, odefun))
            ssol =
                _flow(srhs, sstencil, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{3}(s[1], s[2], s[3]), ssol),
                map(s -> Tensor{2,3}((@views s[4:12] .- s[13:21]) ./ 2δ), ssol),
            )
        else # δ = 0
            u0 = SMatrix{3,4}(x[1], x[2], x[3], 1, 0, 0, 0, 1, 0, 0, 0, 1)
            evsol = _flow(odefun, u0, tspan; tolerance = tolerance, solver = solver, p = p)
            return (
                map(s -> SVector{3}(s[1, 1], s[2, 1], s[3, 1]), evsol),
                map(
                    s -> Tensor{2,3}((
                        s[1, 2],
                        s[2, 2],
                        s[3, 2],
                        s[1, 3],
                        s[2, 3],
                        s[3, 3],
                        s[1, 4],
                        s[2, 4],
                        s[3, 4],
                    )),
                    evsol,
                ),
            )
        end # δ
    end # iip
end

"""
    mean_diff_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor

Return the averaged diffusion tensor at a point along a set of times.
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `kwargs`: are passed to `linearized_flow`
"""
function mean_diff_tensor(odefun, u, tspan, δ; kwargs...)
    return mean(Tensors.dott.(inv.(linearized_flow(odefun, u, tspan, δ; kwargs...)[2])))
end

"""
    CG_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor

Returns the classic right Cauchy--Green strain tensor.
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `kwargs`: are passed to `linearized_flow`
"""
function CG_tensor(odefun, u, tspan, δ; kwargs...)
    Tensors.tdot(linearized_flow(odefun, u, [tspan[1], tspan[end]], δ; kwargs...)[2][end])
end

"""
    pullback_tensors(odefun, u, tspan, δ; D, kwargs...) -> Tuple(Vector{SymmetricTensor},Vector{SymmetricTensor})

Returns the time-resolved pullback tensors of both the diffusion and the metric
tensor along a trajectory.
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `D`: (constant) diffusion tensor, metric tensor is computed via inversion;
     defaults to identity tensor
   * `kwargs` are passed to `linearized_flow`
"""
function pullback_tensors(odefun, u, tspan, δ; D::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
    MT = [Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
    return MT, inv.(MT)
end

"""
    pullback_metric_tensor(odefun, u, tspan, δ; G, kwargs...) -> Vector{SymmetricTensor}

Returns the time-resolved pullback tensors of the metric tensor along a trajectory,
aka right Cauchy-Green strain tensor.
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `G`: (constant) metric tensor
   * `kwargs...` are passed through to `linearized_flow`
"""
function pullback_metric_tensor(odefun, u, tspan, δ; G::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
    return [Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF]
end

"""
    pullback_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...) -> Vector{SymmetricTensor}

Returns the time-resolved pullback tensors of the diffusion tensor along a trajectory.
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `D`: (constant) diffusion tensor
   * `kwargs...` are passed through to `linearized_flow`
"""
function pullback_diffusion_tensor(odefun, u, tspan, δ; D::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
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
    p=nothing,
    tolerance=1.e-3,
    solver=ODE.BS5(),
) where {T<:Real,S<:Real}

    DF, pos = iszero(δ) ?
        linearized_flow(
        odefun,
        u,
        tspan,
        p = p,
        tolerance = tolerance,
        solver = solver,
        give_back_position = true,
    ) :
        linearized_flow(
        odefun,
        u,
        tspan,
        δ,
        p = p,
        tolerance = tolerance,
        solver = solver,
        give_back_position = true,
    )

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
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `B`: (constant) SDE tensor
   * `kwargs...` are passed through to `linearized_flow`
"""
function pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; B::Tensor{2}=one(Tensor{2,2}), kwargs...)
    DFinv = inv.(linearized_flow(odefun, u, tspan, δ; kwargs...)[2])
    return [df ⋅ B for df in DFinv]
end

"""
    av_weighted_CG_tensor(odefun, u, tspan, δ; D, kwargs...) -> SymmetricTensor

Returns the transport tensor of a trajectory, aka  time-averaged,
di ffusivity-structure-weighted version of the classic right Cauchy–Green strain
tensor.
Linearized flow maps are computed with [`linearized_flow`](@ref), see its
documentation for the meaning of `δ`.

   * `odefun`: RHS of the ODE
   * `u`: initial value of the ODE
   * `tspan`: set of time instances at which to save the trajectory
   * `δ`: stencil width for the finite differences, see [`linearized_flow`](@ref)
   * `D`: (constant) diffusion tensor
   * `kwargs...` are passed through to `linearized_flow`
"""
function av_weighted_CG_tensor(odefun, u, tspan, δ; D::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    G = inv(D)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)[2]
    return det(D) * mean([Tensors.symmetric(transpose(df) ⋅ G ⋅ df) for df in DF])
end

met2deg(u) = diagm(SymmetricTensor{2,2}, (1 / cos(deg2rad(u[2])), 1))

deg2met(u) = diagm(SymmetricTensor{2,2}, (cos(deg2rad(u[2])), 1))

function pullback_tensors_geo(odefun, u, tspan, δ; D::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    G = inv(D)
    met2deg_init = met2deg(u)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    PBmet = [deg2met(xi) ⋅ DFi ⋅ met2deg_init for (xi, DFi) in DF]
    PBdiff = [inv(deg2met(xi) ⋅ DFi) for (xi, DFi) in DF]
    return [Tensors.symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PBmet],
    [Tensors.symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PBdiff]
end

function pullback_metric_tensor_geo(odefun, u, tspan, δ; G::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    met2deg_init = met2deg(u)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    PB = [deg2met(xi) ⋅ DFi ⋅ met2deg_init for (xi, DFi) in DF]
    return [Tensors.symmetric(transpose(pb) ⋅ G ⋅ pb) for pb in PB]
end

function pullback_diffusion_tensor_geo(odefun, u, tspan, δ; D::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    PB = [inv(deg2met(xi) ⋅ DFi) for (xi, DFi) in DF]
    return [Tensors.symmetric(pb ⋅ D ⋅ transpose(pb)) for pb in PB]
end

# TODO: this is probably broken, the diffusion tensor is not used!
function pullback_SDE_diffusion_tensor_geo(odefun, u, tspan, δ; D::SymmetricTensor{2}=one(SymmetricTensor{2,2}), kwargs...)
    DF = linearized_flow(odefun, u, tspan, δ; kwargs...)
    B = [inv(deg2met(xi) ⋅ DFi) for (xi, DFi) in DF]
    return B
end
