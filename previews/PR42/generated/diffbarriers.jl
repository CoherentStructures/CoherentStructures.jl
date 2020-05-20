import Pkg
Pkg.add(Pkg.PackageSpec(url="https://github.com/KristofferC/JuAFEM.jl.git"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/CoherentStructures/CoherentStructures.jl.git"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/CoherentStructures/StreamMacros.jl"))
Pkg.add(["OrdinaryDiffEq", "Tensors", "StaticArrays", "AxisArrays", "JLD2", "Plots"])

using StreamMacros

bickley = @velo_from_stream stream begin
    stream = psi₀ + psi₁
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

using Distributed
nprocs() == 1 && addprocs()

using CoherentStructures, StaticArrays, AxisArrays
@everywhere using CoherentStructures, OrdinaryDiffEq, Tensors

q = 81
tspan = range(0., stop=3456000., length=q)
ny = 61
nx = (22ny) ÷ 6
xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
δ = 1.e-6

D = SymmetricTensor{2,2}([2., 0., 1/2])

mCG_tensor = let tspan=tspan, δ=δ, D=D
    u -> av_weighted_CG_tensor(bickley, u, tspan, δ;
          D=D, tolerance=1e-6, solver=Tsit5())
end
C̅ = pmap(mCG_tensor, P; batch_size=ceil(Int, length(P)/nprocs()^2))
p = LCSParameters(2.0)
vortices, singularities = ellipticLCS(C̅, p)

using Plots
trace = tensor_invariants(C̅)[5]
fig = plot_vortices(vortices, singularities, (xmin, ymin), (xmax, ymax);
    bg=trace, title="DBS field and transport barriers", showlabel=true)

C_tensor = let tspan=tspan, δ=δ
    u -> CG_tensor(bickley, u, tspan, δ; tolerance=1e-6, solver=Tsit5())
end
C = pmap(C_tensor, P; batch_size=ceil(Int, length(P)/nprocs()^2))
p = LCSParameters(2.0)
BHvortices, singularities = ellipticLCS(C, p)

foreach(v -> plot_barrier!(v.barriers[1]; color=:red, width=1), BHvortices)
fig

using Distributed
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq, StaticArrays

using JLD2
JLD2.@load("Ocean_geostrophic_velocity.jld2")
VI = interpolateVF(Lon, Lat, Time, UT, VT)

using AxisArrays
q = 91
t_initial = minimum(Time)
t_final = t_initial + 90
tspan = range(t_initial, stop=t_final, length=q)
xmin, xmax, ymin, ymax = -4.0, 7.5, -37.0, -28.0
nx = 300
ny = floor(Int, (ymax - ymin) / (xmax - xmin) * nx)
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
δ = 1.e-5
mCG_tensor = let tspan=tspan, δ=δ, p=VI
    u -> av_weighted_CG_tensor(interp_rhs, u, tspan, δ;
        p=p, tolerance=1e-6, solver=Tsit5())
end

C̅ = pmap(mCG_tensor, P; batch_size=ceil(Int, length(P)/nprocs()^2))
p = LCSParameters(2.5)
vortices, singularities = ellipticLCS(C̅, p)

using Plots
trace = tensor_invariants(C̅)[5]
fig = plot_vortices(vortices, singularities, (xmin, ymin), (xmax, ymax);
    bg=trace, title="DBS field and transport barriers", showlabel=true)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

