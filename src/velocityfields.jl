# (c) 2018 Alvaro de Diego, Daniel Karrasch & Nathanael Schilling

# Bickley jet flow [Rypina et al., 2010]
@define_stream Ψ_bickley begin
    Ψ_bickley = psi₀ + psi₁
    psi₀   = - U₀ * L₀ * tanh(y / L₀)
    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term

    re_sum_term =  Σ₁ + Σ₂ + Σ₃

    Σ₁  =  ε₁ * cos(k₁*(x - c₁*t))
    Σ₂  =  ε₂ * cos(k₂*(x - c₂*t))
    Σ₃  =  ε₃ * cos(k₃*(x - c₃*t))

    k₁ = 2/r₀      ; k₂ = 4/r₀    ; k₃ = 6/r₀

    ε₁ = 0.0075    ; ε₂ = 0.15    ; ε₃ = 0.3
    c₂ = 0.205U₀   ; c₃ = 0.461U₀ ; c₁ = c₃ + (√5-1)*(c₂-c₃)
    U₀ = 62.66e-6  ; L₀ = 1770e-3 ; r₀ = 6371e-3
end
bickleyJet = @velo_from_stream Ψ_bickley
bickleyJet! = OrdinaryDiffEq.ODEFunction{true}((du, u, p, t) -> du .= bickleyJet(u, p, t))
bickleyJetEqVari = @var_velo_from_stream Ψ_bickley
bickleyJetEqVari! = OrdinaryDiffEq.ODEFunction{true}((DU, U, p, t) -> DU .= bickleyJetEqVari(U, p, t))

# rotating double gyre flow  [Mosovsky & Meiss, 2011]
@define_stream Ψ_rot_dgyre begin
    st          = heaviside(t)*heaviside(1-t)*t^2*(3-2*t) + heaviside(t-1)
    Ψ_P         = sin(2π*x)*sin(π*y)
    Ψ_F         = sin(π*x)*sin(2π*y)
    Ψ_rot_dgyre = (1-st) * Ψ_P + st * Ψ_F
end
rot_double_gyre = @velo_from_stream Ψ_rot_dgyre
rot_double_gyre! = OrdinaryDiffEq.ODEFunction{true}((du, u, p, t) -> du .= rot_double_gyre(u, p, t))
rot_double_gyreEqVari = @var_velo_from_stream Ψ_rot_dgyre
rot_double_gyreEqVari! = OrdinaryDiffEq.ODEFunction{true}((DU, U, p, t) -> DU .= rot_double_gyreEqVari(U, p, t))

# interpolated vector field components
"""
    interpolateVF(xspan, yspan, tspan, u, v, itp_type=ITP.BSpline(ITP.Cubic(ITP.Free())))) -> VI

`xspan`, `yspan` and `tspan` span the space-time domain on which the
velocity-components `u` and `v` are given. `u` corresponds to the ``x``- or
eastward component, `v` corresponds to the ``y``- or northward component.
For interpolation, the `Interpolations.jl` package is used; see their
documentation for how to declare other interpolation types.

# Usage
```julia
julia> uv = interpolateVF(xs, ys, ts, u, v)

julia> uv(x, y, t)
2-element SArray{Tuple{2},Float64,1,2} with indices SOneTo(2):
 -44.23554926984537
  -4.964069022198859
```
"""
function interpolateVF(X::AbstractRange{S1},
                       Y::AbstractRange{S1},
                       T::AbstractRange{S1},
                       U::AbstractArray{S2,3},
                       V::AbstractArray{S2,3},
                       itp_type=ITP.BSpline(ITP.Cubic(ITP.Free(ITP.OnGrid())))
                       ) where {S1 <: Real, S2 <: Real}
    UV = map(SVector{2,S2}, U, V)::Array{SVector{2,S2},3}
    return ITP.scale(ITP.interpolate(UV, itp_type), X, Y, T)
end

"""
    interp_rhs(u, p, t) -> SVector{2}

Defines an out-of-place 2D vector field that is readily usable for trajectory
integration from a vector field interpolant. It assumes that the interpolant is
provided via the parameter `p`, usually in the [`flow`](@ref) or tensor functions.

# Example
```
julia> UI = interpolateVF(X, Y, T, U, V)

julia> f = u -> flow(interp_rhs, u, tspan; p=UI)

julia> mCG_tensor = u -> CG_tensor(interp_rhs, u, tspan, δ; p=UI)
```
"""
interp_rhs = OrdinaryDiffEq.ODEFunction{false}((u, p, t) -> p(u[1], u[2], t))

"""
    interp_rhs!(du, u, p, t) -> Vector

Defines a mutating/inplace 2D vector field that is readily usable for trajectory
integration from a vector field interpolant. It assumes that the interpolant is
provided via the parameter `p`.

# Example
```
julia> UI = interpolateVF(X, Y, T, U, V)

julia> f = u -> flow(interp_rhs!, u, tspan; p=UI)

julia> mCG_tensor = u -> CG_tensor(interp_rhs!, u, tspan, δ; p=UI)
```
"""
interp_rhs! = OrdinaryDiffEq.ODEFunction{true}((du, u, p, t) -> du .= p(u[1], u[2], t))

# standard map
const standard_a = 0.971635
standardMap(u) = SVector{2,Float64}((
        rem2pi(u[1] + u[2] + standard_a*sin(u[1]), RoundDown),
        rem2pi(u[2] + standard_a*sin(u[1]), RoundDown)
        ))

function standardMapInv(Tu::AbstractArray{T}) where T <: Number
    return SVector{2,T}((
        goodmod(Tu[1] - Tu[2]               , 2π),
        goodmod(Tu[2] - standard_a*sin(Tu[1]-Tu[2])  , 2π)
        ))
end
DstandardMap(u) = Tensor{2,2}((1.0 + a*cos(u[1]), standard_a*cos(u[1]), 1.0, 1.0))

standardMap8(u) = SVector{2,Float64}((rem2pi(u[1] + u[2], RoundDown),
                                      rem2pi(u[2] + 8sin(u[1] + u[2]), RoundDown)))

DstandardMap8(u) = Tensor{2,2}((1.0, 8cos(u[1] + u[2]), 1.0, 1.0 + 8cos(u[1] + u[2])))

standardMap8Inv(Tu) = SVector{2,Float64}((
                            rem2pi(Tu[1]  - Tu[2] + 8sin(Tu[1]), RoundDown),
                            rem2pi(Tu[2] - 8sin(Tu[1]), RoundDown)
                            ))

# ABC flow
function ABC_flow(u, p, t)
    A, B, C = p
    return SVector{3,Float64}((
        A + 0.5 * t * sin(π * t)) * sin(u[3]) + C * cos(u[2]),
        B * sin(u[1]) + (A + 0.5 * t * sin(π * t)) * cos(u[3]),
        C * sin(u[2]) + B * cos(u[1])
        )
end
abcFlow = OrdinaryDiffEq.ODEFunction{false}(ABC_flow)

# cylinder flow [Froyland, Lloyd, and Santitissadeekorn, 2010]
function _cylinder_flow(u, p, t)
    c, ν, ε = p
    # c = 0.5
    # ν = 0.25
    # ϵ = 0.25
    A(t) = 1.0 + 0.125 * sin(2 * √5.0 * t)
    G(ψ) = 1.0 / (ψ ^ 2 + 1.0) ^ 2
    g(x,y,t) = sin(x - ν * t) * sin(y) + y / 2 - π / 4
    x = u[1]
    y = u[2]
    return SVector{2,Float64}(
        c - A(t) * sin(x - ν * t) * cos(y) + ε * G(g(x, y, t))*sin(t / 2),
        A(t) * cos(x - ν * t) * sin(y)
        )
end
cylinder_flow = OrdinaryDiffEq.ODEFunction{false}(_cylinder_flow)
