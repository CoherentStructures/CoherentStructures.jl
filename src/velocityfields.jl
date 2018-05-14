#velocityfields.jl from Daniel Karrasch

ITP = Interpolations

#TODO: See if type specification actually helps, remove redundant vector fields,
# rotating double gyre and Bickley jet can both be obtained from Alvaro's macro
# this means, apart from the interpolation functions, all other functions may be removed?
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

bickleyJet          = @velo_from_stream Ψ_bickley
bickleyJetEqVari    = @var_velo_from_stream Ψ_bickley

@define_stream Ψ_rot_dgyre begin
    st = heaviside(t)*heaviside(1-t)*t^2*(3-2*t) + heaviside(t-1)
    Ψ_P = sin(2π*x)*sin(π*y)
    Ψ_F = sin(π*x)*sin(2π*y)
    Ψ_rot_dgyre = (1-st)*Ψ_P + st*Ψ_F
end

rot_double_gyre = @velo_from_stream Ψ_rot_dgyre
rot_double_gyreEqVari = @var_velo_from_stream Ψ_rot_dgyre



#TODO: Give variables a sensible type here
function interpolateVF(xspan,yspan,tspan,u,v,interpolation_type=ITP.BSpline(ITP.Cubic(ITP.Free())))
    # convert arrays into linspace-form for interpolation
    const X = linspace(minimum(xspan),maximum(xspan),length(xspan))
    const Y = linspace(minimum(yspan),maximum(yspan),length(yspan))
    const T = linspace(minimum(tspan),maximum(tspan),length(tspan))

    ui = ITP.interpolate(u,interpolation_type,ITP.OnGrid())
    UI = ITP.scale(ui,X,Y,T)
    # UE = extrapolate(UI,(Linear(),Linear(),Flat()))
    vi = ITP.interpolate(v,interpolation_type,ITP.OnGrid())
    VI = ITP.scale(vi,X,Y,T)
    # VE = extrapolate(VI,(Linear(),Linear(),Flat()))
    return UI,VI
end

function interpolateVFPeriodic(Lon,Lat,Time, UT, VT,interpolation_type=ITP.BSpline(ITP.Linear()))
    # convert arrays into linspace-form for interpolation
    const lon = linspace(minimum(Lon),maximum(Lon),length(Lon))
    const lat = linspace(minimum(Lat),maximum(Lat),length(Lat))
    const time = linspace(minimum(Time),maximum(Time),length(Time))
    UI = ITP.interpolate(UT,interpolation_type,ITP.OnGrid())
    UI = ITP.scale(UI,lon,lat,time)
    UE = ITP.extrapolate(UI,(ITP.Periodic(),ITP.Periodic(),ITP.Flat()))
    VI = ITP.interpolate(VT,interpolation_type,ITP.OnGrid())
    VI = ITP.scale(VI,lon,lat,time)
    VE = ITP.extrapolate(VI,(ITP.Periodic(),ITP.Periodic(),ITP.Flat()))
    return UE,VE
end

function standardMap(u)
    const a = 0.971635
    return StaticArrays.SVector{2,Float64}((mod(u[1] + u[2] + a*sin(u[1]),2π), mod(u[2] + a*sin(u[1]), 2π)))
end

function standardMapInv(Tu)
    const a = 0.971635
    return StaticArrays.SVector{2,Float64}( mod(Tu[1] - Tu[2],2π), mod(Tu[2] - a*sin(Tu[1]-Tu[2]),2π))
end

function DstandardMap(u)
    const a = 0.971635
    return Tensors.Tensor{2,2,Float64,4}(
    (
        1 + a*cos(u[1]), a*cos(u[1]), 1.,1.
    ));
end

function abcFlow2(u,p,t)
    A = sqrt(3)
    B = sqrt(2)
    C = 1
    return StaticArrays.SVector{3,Float64}(
        (A + 0.5*t*sin(π*t))*sin(u[3]) + C*cos(u[2]),
        B*sin(u[1]) + (A + 0.5*t*sin(π*t))*cos(u[3]),
        C*sin(u[2]) + B*cos(u[1])
        )
end

#The function below is taken from Oliver Junge's main_rot_gyre.jl
# function rot_double_gyre!(du::AbstractVector{T},u::AbstractVector{T},p,t::T) where {T <: Real}
# #function rot_double_gyre2(t::Float64,x,dx)
#   st = ((t>0) & (t<1))*t^2*(3-2*t) + (t>=1)*1
#   dxΨP = 2π*cos(2π*u[1])*sin(π*u[2])
#   dyΨP = π*sin(2π*u[1])*cos(π*u[2])
#   dxΨF = π*cos(π*u[1])*sin(2π*u[2])
#   dyΨF = 2π*sin(π*u[1])*cos(2π*u[2])
#   du[1] = - ((1-st)*dyΨP + st*dyΨF)
#   du[2] = (1-st)*dxΨP + st*dxΨF
# end
#
# @inbounds function rot_double_gyre(u,p,t)
#     st = ((t>0) & (t<1))*t^2*(3-2*t) + (t>=1)*1
#     dxΨP = 2π*cos(2π*u[1])*sin(π*u[2])
#     dyΨP = π*sin(2π*u[1])*cos(π*u[2])
#     dxΨF = π*cos(π*u[1])*sin(2π*u[2])
#     dyΨF = 2π*sin(π*u[1])*cos(2π*u[2])
#     du1 = - ((1-st)*dyΨP + st*dyΨF)
#     du2 = (1-st)*dxΨP + st*dxΨF
#     return StaticArrays.SVector{2}(du1, du2)
# end
#
# function transientGyresEqVari!(du::AbstractVector{T},u::AbstractVector{T},p,t::T) where {T <: Real}
#     st = ((t>0)&(t<1))*t^2*(3-2*t) + (t>1)*1
#
#     # Psi_P = sin(2*pi*x)*sin(pi*y)
#     dxPsi_P = 2*pi*cos(2*pi*u[1])*sin(pi*u[2])
#     dyPsi_P = pi*sin(2*pi*u[1])*cos(pi*u[2])
#     # Psi_F = sin(pi*x)*sin(2*pi*y)
#     dxPsi_F = pi*cos(pi*u[1])*sin(2*pi*u[2])
#     dyPsi_F = 2*pi*sin(pi*u[1])*cos(2*pi*u[2])
#     # Psi = (1-st)*Psi_P + st*Psi_F;
#     dxPsi = (1-st)*dxPsi_P + st*dxPsi_F
#     dyPsi = (1-st)*dyPsi_P + st*dyPsi_F
#     # vector field
#     du[1] = -dyPsi
#     du[2] = dxPsi
#
#     dxdxPsi_P = -4*pi^2*sin(2*pi*u[1])*sin(pi*u[2])
#     dxdyPsi_P = 2*pi^2*cos(2*pi*u[1])*cos(pi*u[2])
#     dydyPsi_P = -pi^2*sin(2*pi*u[1])*sin(pi*u[2])
#
#     # dxPsi_F = pi*cos(pi*x)*sin(2*pi*y)
#     # dyPsi_F = 2*pi*sin(pi*x)*cos(2*pi*y)
#     dxdxPsi_F = -pi^2*sin(pi*u[1])*sin(2*pi*u[2])
#     dxdyPsi_F = 2*pi^2*cos(pi*u[1])*cos(2*pi*u[2])
#     dydyPsi_F = -4*pi^2*sin(pi*u[1])*sin(2*pi*u[2])
#
#     dxdxPsi = (1-st)*dxdxPsi_P + st*dxdxPsi_F
#     dxdyPsi = (1-st)*dxdyPsi_P + st*dxdyPsi_F
#     dydyPsi = (1-st)*dydyPsi_P + st*dydyPsi_F
#
#     # variational equation
#     df1 = -dxdyPsi
#     df2 = dxdxPsi
#     df3 = -dydyPsi
#     df4 = dxdyPsi
#     du[3] = df1*u[3]+df3*u[4]
#     du[4] = df2*u[3]+df4*u[4]
#     du[5] = df1*u[5]+df3*u[6]
#     du[6] = df2*u[5]+df4*u[6]
# end
#
# function transientGyresEqVari(u,p,t)
#     st = ((t>0)&(t<1))*t^2*(3-2*t) + (t>1)*1
#
#     # Psi_P = sin(2*pi*x)*sin(pi*y)
#     dxPsi_P = 2*pi*cos(2*pi*u[1])*sin(pi*u[2])
#     dyPsi_P = pi*sin(2*pi*u[1])*cos(pi*u[2])
#     # Psi_F = sin(pi*x)*sin(2*pi*y)
#     dxPsi_F = pi*cos(pi*u[1])*sin(2*pi*u[2])
#     dyPsi_F = 2*pi*sin(pi*u[1])*cos(2*pi*u[2])
#     # Psi = (1-st)*Psi_P + st*Psi_F;
#     dxPsi = (1-st)*dxPsi_P + st*dxPsi_F
#     dyPsi = (1-st)*dyPsi_P + st*dyPsi_F
#     # vector field
#     du1 = -dyPsi
#     du2 = dxPsi
#
#     dxdxPsi_P = -4*pi^2*sin(2*pi*u[1])*sin(pi*u[2])
#     dxdyPsi_P = 2*pi^2*cos(2*pi*u[1])*cos(pi*u[2])
#     dydyPsi_P = -pi^2*sin(2*pi*u[1])*sin(pi*u[2])
#
#     # dxPsi_F = pi*cos(pi*x)*sin(2*pi*y)
#     # dyPsi_F = 2*pi*sin(pi*x)*cos(2*pi*y)
#     dxdxPsi_F = -pi^2*sin(pi*u[1])*sin(2*pi*u[2])
#     dxdyPsi_F = 2*pi^2*cos(pi*u[1])*cos(2*pi*u[2])
#     dydyPsi_F = -4*pi^2*sin(pi*u[1])*sin(2*pi*u[2])
#
#     dxdxPsi = (1-st)*dxdxPsi_P + st*dxdxPsi_F
#     dxdyPsi = (1-st)*dxdyPsi_P + st*dxdyPsi_F
#     dydyPsi = (1-st)*dydyPsi_P + st*dydyPsi_F
#
#     # variational equation
#     df1 = -dxdyPsi
#     df2 = dxdxPsi
#     df3 = -dydyPsi
#     df4 = dxdyPsi
#     du3 = df1*u[3]+df3*u[4]
#     du4 = df2*u[3]+df4*u[4]
#     du5 = df1*u[5]+df3*u[6]
#     du6 = df2*u[5]+df4*u[6]
#     return StaticArrays.SVector{6}(du1, du2, du3, du4, du5, du6)
# end

# function bickleyJet!(du::AbstractVector{T},u::AbstractVector{T},p,t::T) where {T <: Real}
#     U, L, c₁, c₂, c₃, ϵ₁, ϵ₂, ϵ₃, k₁, k₂, k₃ = p
#     x = u[2]/L
#     du[1] = U*sech(x)^2+(2*ϵ₁*U*cos(k₁*(u[1]-c₁*t))+2*ϵ₂*U*cos(k₂*(u[1]-c₂*t))+2*ϵ₃*U*cos(k₃*(u[1]-c₃*t)))*tanh(x)*sech(x)^2
#     du[2] = -(ϵ₁*k₁*U*L*sin(k₁*(u[1]-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(u[1]-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(u[1]-c₃*t)))*sech(x)^2
# end
#
# function bickleyJet(u,p,t)
#     U, L, c₁, c₂, c₃, ϵ₁, ϵ₂, ϵ₃, k₁, k₂, k₃ = p
#     x = u[2]/L
#     du1 = U*sech(x)^2+(2*ϵ₁*U*cos(k₁*(u[1]-c₁*t))+2*ϵ₂*U*cos(k₂*(u[1]-c₂*t))+2*ϵ₃*U*cos(k₃*(u[1]-c₃*t)))*tanh(x)*sech(x)^2
#     du2 = -(ϵ₁*k₁*U*L*sin(k₁*(u[1]-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(u[1]-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(u[1]-c₃*t)))*sech(x)^2
#     return StaticArrays.SVector{2}(du1, du2)
# end
#
# function bickleyJetEqVari!(du::AbstractVector{T},u::AbstractVector{T},p,t::T) where {T <: Real}
#     U, L, c₁, c₂, c₃, ϵ₁, ϵ₂, ϵ₃, k₁, k₂, k₃ = p
#     du1 = U*sech(u[2]/L)^2+(2*ϵ₁*U*cos(k₁*(u[1]-c₁*t))+2*ϵ₂*U*cos(k₂*(u[1]-c₂*t))+2*ϵ₃*U*cos(k₃*(u[1]-c₃*t)))*tanh(u[2]/L)*sech(u[2]/L)^2
#     du2 = -(ϵ₁*k₁*U*L*sin(k₁*(u[1]-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(u[1]-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(u[1]-c₃*t)))*sech(u[2]/L)^2
#     df1 = -(tanh(u[2]/L)*(2*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+2*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/cosh(u[2]/L)^2
#          - (tanh(u[2]/L)*(2*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+ 2*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/cosh(u[2]/L)^2
#     df2 = -(L*U*ϵ₁*k₁^2*cos(k₁*(u[1] - c₁*t)) + L*U*ϵ₂*k₂^2*cos(k₂*(u[1] - c₂*t))+
#          L*U*ϵ₃*k₃^2*cos(k₃*(u[1] - c₃*t)))/cosh(u[2]/L)^2
#     df3 = - ((tanh(u[2]/L).^2 - 1).*(2*U*ϵ₁*cos(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*cos(k₂*(u[1] - c₂*t))+
#         2*U*ϵ₃*cos(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^2) - (2*U*sinh(u[2]/L))/(L*cosh(u[2]/L)^3)-
#         (2*sinh(u[2]/L).*tanh(u[2]/L)*(2*U*ϵ₁*cos(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*cos(k₂*(u[1] - c₂*t))+
#         2*U*ϵ₃*cos(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^3)
#     df4 = (2*sinh(u[2]/L)*(L*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + L*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+
#         L*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^3)
#     du3 = df1*u[3]+df3*u[4]
#     du4 = df2*u[3]+df4*u[4]
#     du5 = df1*u[5]+df3*u[6]
#     du6 = df2*u[5]+df4*u[6]
# end # r₀=6371e-3
#
# function bickleyJetEqVari(u,p,t)
#     U, L, c₁, c₂, c₃, ϵ₁, ϵ₂, ϵ₃, k₁, k₂, k₃ = p
#     du1 = U*sech(u[2]/L)^2+(2*ϵ₁*U*cos(k₁*(u[1]-c₁*t))+2*ϵ₂*U*cos(k₂*(u[1]-c₂*t))+2*ϵ₃*U*cos(k₃*(u[1]-c₃*t)))*tanh(u[2]/L)*sech(u[2]/L)^2
#     du2 = -(ϵ₁*k₁*U*L*sin(k₁*(u[1]-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(u[1]-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(u[1]-c₃*t)))*sech(u[2]/L)^2
#     df1 = -(tanh(u[2]/L)*(2*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+2*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/cosh(u[2]/L)^2
#          - (tanh(u[2]/L)*(2*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+ 2*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/cosh(u[2]/L)^2
#     df2 = -(L*U*ϵ₁*k₁^2*cos(k₁*(u[1] - c₁*t)) + L*U*ϵ₂*k₂^2*cos(k₂*(u[1] - c₂*t))+
#          L*U*ϵ₃*k₃^2*cos(k₃*(u[1] - c₃*t)))/cosh(u[2]/L)^2
#     df3 = - ((tanh(u[2]/L).^2 - 1).*(2*U*ϵ₁*cos(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*cos(k₂*(u[1] - c₂*t))+
#         2*U*ϵ₃*cos(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^2) - (2*U*sinh(u[2]/L))/(L*cosh(u[2]/L)^3)-
#         (2*sinh(u[2]/L).*tanh(u[2]/L)*(2*U*ϵ₁*cos(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*cos(k₂*(u[1] - c₂*t))+
#         2*U*ϵ₃*cos(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^3)
#     df4 = (2*sinh(u[2]/L)*(L*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + L*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+
#         L*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^3)
#     du3 = df1*u[3]+df3*u[4]
#     du4 = df2*u[3]+df4*u[4]
#     du5 = df1*u[5]+df3*u[6]
#     du6 = df2*u[5]+df4*u[6]
#     return StaticArrays.SVector{6}(du1, du2, du3, du4, du5, du6)
# end # r₀=6371e-3
