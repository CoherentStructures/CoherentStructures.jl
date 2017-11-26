#velocityFields.jl from Daniel Karrasch
@everywhere using DiffEqBase, OrdinaryDiffEq, ParameterizedFunctions

@everywhere transientGyres = @ode_def tGyres begin
# @everywhere function transient_gyres(t,u,du)
 dx = -((1-t^2*(3-2*t))*π*sin(2π*x)*cos(π*y) + t^2*(3-2*t)*2π*sin(π*x)*cos(2π*y))
 dy = (1-t^2*(3-2*t))*2π*cos(2π*x)*sin(π*y) + t^2*(3-2*t)*π*cos(π*x)*sin(2π*y)
end

@everywhere bickleyJet = @ode_def bJet begin
  dx = U*sech(y/L)^2+(2*ϵ₁*U*cos(k₁*(x-c₁*t))+2*ϵ₂*U*cos(k₂*(x-c₂*t))+2*ϵ₃*U*cos(k₃*(x-c₃*t)))*tanh(y/L)*sech(y/L)^2
  dy = -(ϵ₁*k₁*U*L*sin(k₁*(x-c₁*t)) + ϵ₂*k₂*U*L*sin(k₂*(x-c₂*t)) + ϵ₃*k₃*U*L*sin(k₃*(x-c₃*t)))*sech(y/L)^2
end U=62.66e-6 L=1770e-3 c₂=1.28453e-5 c₃=2.888626e-5 ϵ₁ = 0.0075 ϵ₂=0.15 ϵ₃=0.3 k₁=0.31392246115209543 k₂=0.6278449223041909 k₃=0.9417673834562862 c₁=9.058543015644972e-6
# r₀=6371e-3

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
