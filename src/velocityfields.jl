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
bickleyJet!(du,u,p,t) = du .= bickleyJet(u,p,t)
bickleyJetEqVari = @var_velo_from_stream Ψ_bickley
bickleyJetEqVari!(DU,U,p,t) = bickleyJetEqVari(U,p,t)

# rotating double gyre flow  [Mosovsky & Meiss, 2011]
@define_stream Ψ_rot_dgyre begin
    st          = heaviside(t)*heaviside(1-t)*t^2*(3-2*t) + heaviside(t-1)
    Ψ_P         = sin(2π*x)*sin(π*y)
    Ψ_F         = sin(π*x)*sin(2π*y)
    Ψ_rot_dgyre = (1-st) * Ψ_P + st * Ψ_F
end
rot_double_gyre = @velo_from_stream Ψ_rot_dgyre
rot_double_gyre!(du,u,p,t) = du .= rot_double_gyre(u,p,t)
rot_double_gyreEqVari   = @var_velo_from_stream Ψ_rot_dgyre
rot_double_gyreEqVari!(DU,U,p,t) = DU .= rot_double_gyreEqVari(U,p,t)

# interpolated vector field components
"""
    interpolateVF(xspan,yspan,tspan,u,v,interpolation_type=ITP.BSpline(ITP.Cubic(ITP.Free())))) -> UI, VI
"""
function interpolateVF(xspan::AbstractVector{S1},
                        yspan::AbstractVector{S1},
                        tspan::AbstractVector{S1},
                        u::AbstractArray{S2,3},
                        v::AbstractArray{S2,3},
                        interpolation_type=ITP.BSpline(ITP.Cubic(ITP.Free(ITP.OnGrid())))
                        ) where {S1 <: Real, S2 <: Real}

    # convert arrays into linspace-form for interpolation
    X = range(minimum(xspan), stop=maximum(xspan), length=length(xspan))
    Y = range(minimum(yspan), stop=maximum(yspan), length=length(yspan))
    T = range(minimum(tspan), stop=maximum(tspan), length=length(tspan))

    UI = ITP.scale(ITP.interpolate(u, interpolation_type), X, Y, T)
    #UE = extrapolate(UI,(Flat(),Flat(),Flat()))
    VI = ITP.scale(ITP.interpolate(v, interpolation_type), X, Y, T)
    #VE = extrapolate(VI,(Flat(),Flat(),Flat()))
    return UI, VI
end

#The rhs for an ODE on interpolated vector fields
#The interpolant is passed via the p argument
"""
    interp_rhs(u, p, t) -> SVector{2}

Defines a 2D vector field that is readily usable for trajectory integration from
vector field interpolants of the x- and y-direction, resp. It assumes that the
interpolants are provided as a 2-tuple `(UI, VI)` via the parameter `p`. Here,
`UI` and `VI` are the interpolants for the x- and y-components of the velocity
field.
"""
function interp_rhs(u, p, t)
    du1 = p[1](u[1], u[2], t)
    du2 = p[2](u[1], u[2], t)
    return SVector{2}(du1, du2)
end
#TODO: think of adding @inbounds here
function interp_rhs!(du, u, p, t)
    du[1] = p[1](u[1], u[2], t)
    du[2] = p[2](u[1], u[2], t)
    return du
end

# standard map
function standardMap(u)
    a = 0.971635
    return SVector{2,Float64}((
        mod(u[1] + u[2] + a*sin(u[1]), 2π),
        mod(u[2] + a*sin(u[1]), 2π)
        ))
end
function standardMapInv(Tu::AbstractArray{T}) where T <: Number
    a = 0.971635
    return SVector{2,T}((
        goodmod(Tu[1] - Tu[2]               , 2π),
        goodmod(Tu[2] - a*sin(Tu[1]-Tu[2])  , 2π)
        ))
end
function DstandardMap(u)
    a = 0.971635
    return Tensor{2,2}((
        1.0 + a*cos(u[1])   , a*cos(u[1]), 1.0                 , 1.0
        ))
end
function standardMap8(u)
    return SVector{2,Float64}((
        mod(u[1] + u[2],2π),
        mod(u[2] + 8*sin(u[1] + u[2]),2π)
        ))
end
function DstandardMap8(u)
    return Tensor{2,2}((
        1., 8*cos(u[1] + u[2]), 1., 1. + 8*cos(u[1] + u[2])
    ))
end
function standardMap8Inv(Tu)
    return SVector{2,Float64}((
    mod(Tu[1]  - Tu[2] + 8*sin(Tu[1]),2π),
    mod(Tu[2] - 8*sin(Tu[1]),2π)
    ))
end

# ABC flow
function abcFlow(u,p,t)
    A, B, C = p
    return SVector{3,Float64}((
        A + 0.5*t*sin(π*t))*sin(u[3]) + C*cos(u[2]),
        B*sin(u[1]) + (A + 0.5*t*sin(π*t))*cos(u[3]),
        C*sin(u[2]) + B*cos(u[1])
        )
end

# cylinder flow [Froyland, Lloyd, and Santitissadeekorn, 2010]
function cylinder_flow(u,p,t)
    c, ν, ε = p
    # c = 0.5
    # ν = 0.25
    # ϵ = 0.25
    A(t) = 1.0 + 0.125*sin(2*√5.0*t)
    G(ψ) = 1.0 / (ψ^2 + 1.0)^2
    g(x,y,t) = sin(x-ν*t)*sin(y) + y/2 - π/4
    x = u[1]
    y = u[2]
    return SVector{2,Float64}(
        c - A(t)*sin(x - ν*t)*cos(y) + ε*G(g(x,y,t))+sin(t/2),
        A(t)*cos(x - ν*t)*sin(y)
        )
end
