#velocityFields.jl from Daniel Karrasch

#TODO: Uncomment everything with @ode_def and figure out why it causes an error

#The function below is taken from Oliver Junge's main_rot_gyre.jl
function rot_double_gyre2!(dx::AbstractArray{Float64},x::AbstractArray{Float64}, p,t::Float64)
#function rot_double_gyre2(t::Float64,x,dx)
  st = ((t>0)&(t<1))*t^2*(3-2*t) + (t>=1)*1
  dxΨP = 2π*cos(2π*x[1])*sin(π*x[2])
  dyΨP = π*sin(2π*x[1])*cos(π*x[2])
  dxΨF = π*cos(π*x[1])*sin(2π*x[2])
  dyΨF = 2π*sin(π*x[1])*cos(2π*x[2])
  dx[1] = - ((1-st)dyΨP + st*dyΨF)
  dx[2] = (1-st)dxΨP + st*dxΨF
end


transientGyres = @ode_def tGyres begin
    dx = -((1-t^2*(3-2*t))*π*sin(2π*x)*cos(π*y) + t^2*(3-2*t)*2π*sin(π*x)*cos(2π*y))
    dy = (1-t^2*(3-2*t))*2π*cos(2π*x)*sin(π*y) + t^2*(3-2*t)*π*cos(π*x)*sin(2π*y)
end

function transientGyresEqVari(du,u,p,t)
    st = ((t>0)&(t<1))*t^2*(3-2*t) + (t>1)*1

    # Psi_P = sin(2*pi*x)*sin(pi*y)
    dxPsi_P = 2*pi*cos(2*pi*u[1])*sin(pi*u[2])
    dyPsi_P = pi*sin(2*pi*u[1])*cos(pi*u[2])
    # Psi_F = sin(pi*x)*sin(2*pi*y)
    dxPsi_F = pi*cos(pi*u[1])*sin(2*pi*u[2])
    dyPsi_F = 2*pi*sin(pi*u[1])*cos(2*pi*u[2])
    # Psi = (1-st)*Psi_P + st*Psi_F;
    dxPsi = (1-st)*dxPsi_P + st*dxPsi_F
    dyPsi = (1-st)*dyPsi_P + st*dyPsi_F
    # vector field
    du[1] = -dyPsi
    du[2] = dxPsi

    dxdxPsi_P = -4*pi^2*sin(2*pi*u[1])*sin(pi*u[2])
    dxdyPsi_P = 2*pi^2*cos(2*pi*u[1])*cos(pi*u[2])
    dydyPsi_P = -pi^2*sin(2*pi*u[1])*sin(pi*u[2])

    # dxPsi_F = pi*cos(pi*x)*sin(2*pi*y)
    # dyPsi_F = 2*pi*sin(pi*x)*cos(2*pi*y)
    dxdxPsi_F = -pi^2*sin(pi*u[1])*sin(2*pi*u[2])
    dxdyPsi_F = 2*pi^2*cos(pi*u[1])*cos(2*pi*u[2])
    dydyPsi_F = -4*pi^2*sin(pi*u[1])*sin(2*pi*u[2])

    dxdxPsi = (1-st)*dxdxPsi_P + st*dxdxPsi_F
    dxdyPsi = (1-st)*dxdyPsi_P + st*dxdyPsi_F
    dydyPsi = (1-st)*dydyPsi_P + st*dydyPsi_F

    # variational equation
    df1 = -dxdyPsi
    df2 = dxdxPsi
    df3 = -dydyPsi
    df4 = dxdyPsi
    du[3] = df1*u[3]+df3*u[4]
    du[4] = df2*u[3]+df4*u[4]
    du[5] = df1*u[5]+df3*u[6]
    du[6] = df2*u[5]+df4*u[6]
end

bickleyJet = @ode_def bJet begin
  U=62.66e-6
  L=1770e-3
  c₂=1.28453e-5
  c₃=2.888626e-5
  ϵ₁=0.0075
  ϵ₂=0.15
  ϵ₃=0.3
  k₁=0.31392246115209543
  k₂=0.6278449223041909
  k₃=0.9417673834562862
  c₁=9.058543015644972e-6
  dx = U*sech(y/L)^2+(2*ϵ₁*U*cos(k₁*(x-c₁*t))+2*ϵ₂*U*cos(k₂*(x-c₂*t))+2*ϵ₃*U*cos(k₃*(x-c₃*t)))*tanh(y/L)*sech(y/L)^2
  dy = -(ϵ₁*k₁*U*L*sin(k₁*(x-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(x-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(x-c₃*t)))*sech(y/L)^2
end


function bickleyJetEqVari(du::AbstractArray, u::AbstractArray, p, t::Float64)
 # velo = bickleyJet(t,[u[1],u[2]])
 # du[1] = velo[1]
 # du[2] = velo[2]
 # gradvelo = bickleyJet(Val{:jac},t,[u[1],u[2]])
 # du[3] = gradvelo[1]*u[3]+gradvelo[3]*u[4]
 # du[4] = gradvelo[2]*u[3]+gradvelo[4]*u[4]
 # du[5] = gradvelo[1]*u[5]+gradvelo[3]*u[6]
 # du[6] = gradvelo[2]*u[5]+gradvelo[4]*u[6]
 U=62.66e-6; L=1770e-3; c₂=1.28453e-5; c₃=2.888626e-5; ϵ₁=0.0075; ϵ₂=0.15; ϵ₃=0.3
 k₁=0.31392246115209543; k₂=0.6278449223041909; k₃=0.9417673834562862; c₁=9.058543015644972e-6
 du[1] = U*sech(u[2]/L)^2+(2*ϵ₁*U*cos(k₁*(u[1]-c₁*t))+2*ϵ₂*U*cos(k₂*(u[1]-c₂*t))+2*ϵ₃*U*cos(k₃*(u[1]-c₃*t)))*tanh(u[2]/L)*sech(u[2]/L)^2
 du[2] = -(ϵ₁*k₁*U*L*sin(k₁*(u[1]-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(u[1]-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(u[1]-c₃*t)))*sech(u[2]/L)^2
 df1 = -(tanh(u[2]/L)*(2*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+2*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/cosh(u[2]/L)^2
     - (tanh(u[2]/L)*(2*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+ 2*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/cosh(u[2]/L)^2
 df2 = -(L*U*ϵ₁*k₁^2*cos(k₁*(u[1] - c₁*t)) + L*U*ϵ₂*k₂^2*cos(k₂*(u[1] - c₂*t))+
     L*U*ϵ₃*k₃^2*cos(k₃*(u[1] - c₃*t)))/cosh(u[2]/L)^2
 df3 = - ((tanh(u[2]/L).^2 - 1).*(2*U*ϵ₁*cos(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*cos(k₂*(u[1] - c₂*t))+
    2*U*ϵ₃*cos(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^2) - (2*U*sinh(u[2]/L))/(L*cosh(u[2]/L)^3)-
    (2*sinh(u[2]/L).*tanh(u[2]/L)*(2*U*ϵ₁*cos(k₁*(u[1] - c₁*t)) + 2*U*ϵ₂*cos(k₂*(u[1] - c₂*t))+
    2*U*ϵ₃*cos(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^3)
 df4 = (2*sinh(u[2]/L)*(L*U*ϵ₁*k₁*sin(k₁*(u[1] - c₁*t)) + L*U*ϵ₂*k₂*sin(k₂*(u[1] - c₂*t))+
    L*U*ϵ₃*k₃*sin(k₃*(u[1] - c₃*t))))/(L*cosh(u[2]/L)^3)
 du[3] = df1*u[3]+df3*u[4]
 du[4] = df2*u[3]+df4*u[4]
 du[5] = df1*u[5]+df3*u[6]
 du[6] = df2*u[5]+df4*u[6]
end # r₀=6371e-3


#TODO: Give variables a sensible type here
function interpolateVF(Lon,Lat,UT, time,VT,interpolation_type=BSpline(Cubic(Free())))
    # convert arrays into linspace-form for interpolation
    const lon = linspace(minimum(Lon),maximum(Lon),length(Lon))
    const lat = linspace(minimum(Lat),maximum(Lat),length(Lat))
    const Time = linspace(minimum(time),maximum(time),length(time))

    UI = Interpolations.interpolate(permutedims(UT,[2,1,3]),interpolation_type,OnGrid())
    UI = Interpolations.scale(UI,lon,lat,Time)
    # UE = extrapolate(UI,(Linear(),Linear(),Flat()))
    VI = Interpolations.interpolate(permutedims(VT,[2,1,3]),interpolation_type,OnGrid())
    VI = Interpolations.scale(VI,lon,lat,Time)
    # VE = extrapolate(VI,(Linear(),Linear(),Flat()))
    return UI,VI
end

#The rhs for an ODE on interpolated vector fields
#The interpolant is passed via the p argument

#TODO: think of adding @inbounds here
function interp_rhs(du::AbstractArray{Float64},u::AbstractArray{Float64},p,t::Float64)
    du[1] = p[1][u[1],u[2],t]
    du[2] = p[2][u[1],u[2],t]
end


# @everywhere function oceanVFEqVari(t::Number,u::Vector,du::Vector)
#
#  du[1] = UI[u[1], u[2], t]
#  du[2] = VI[u[1], u[2], t]
#  du[3] =
#  du[4] =
#  du[5] =
#  du[6] =
# end

# @everywhere function bickleyJet(t::Float64,u::AbstractArray,du::AbstractArray)
#   x = u[1]; y = u[2];
#   # Parameters
#   U = 62.66e-6
#   # U=5.4138; %Froyland&Koltai2015
#   L = 1770e-3
#   c2 = 1.28453e-5 #.205*U
#   # c2=0.2054*U; %Froyland&Koltai2015
#   c3 = 2.888626e-5 #.461*U
#   # c3=0.4108*U; %Froyland&Koltai2015
#   # In Gary's paper
#   eps1 = 0.0075
#   eps2 = 0.15
#   eps3 = 0.3
#   # Cranked up a bit
#   # eps1=.9; eps2=.5; eps3=.3;
#   # eps1 = 0; eps2 = 0.1; eps3 = 0.3; %Froyland&Koltai2015
#   r0 = 6371e-3
#   k1 = 0.31392246115209543 # 2/r0
#   k2 = 0.6278449223041909 # 4/r0
#   k3 = 0.9417673834562862 # 6/r0
#   c1 = 9.058543015644972e-6 # c3+((sqrt(5)-1)/2)*(k2/k1)*(c2-c3)
#   du = [U*sech(y/L)^2+(2*eps1*U*cos(k1*(x-c1*t))+2*eps2*U*cos(k2*(x-c2*t))+2*eps3*U*cos(k3*(x-c3*t)))*tanh(y/L)*sech(y/L)^2,
#         -(eps1*k1*U*L*sin(k1*(x-c1*t)) + eps2*k2*U*L*sin(k2*(x-c2*t)) + eps3*k3*U*L*sin(k3*(x-c3*t)))*sech(y/L)^2]
# end
