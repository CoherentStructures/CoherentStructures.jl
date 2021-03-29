import Pkg
Pkg.add("FourierFlows")
Pkg.add("GeophysicalFlows")
Pkg.add("Plots")
Pkg.add(Pkg.PackageSpec(url="https://github.com/CoherentStructures/CoherentStructures.jl.git"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/CoherentStructures/OceanTools.jl.git"))
Pkg.pin(Pkg.PackageSpec(name="FourierFlows", version="0.4.5"))
Pkg.pin(Pkg.PackageSpec(name="GeophysicalFlows", version="0.5.1"))

using FourierFlows, GeophysicalFlows, GeophysicalFlows.TwoDNavierStokes, Plots, Random
Random.seed!(1234)
mygrid = TwoDGrid(256, 2π)
x, y = gridpoints(mygrid)
xs = ys = range(-π, stop=π, length=257)[1:end-1];

ε = 0.001 # Energy injection rate
kf, dkf = 6, 2.0 # Waveband where to inject energy
Kr = [mygrid.kr[i] for i=1:mygrid.nkr, j=1:mygrid.nl]
force2k = @. exp(-(sqrt(mygrid.Krsq) - kf)^2 / (2 * dkf^2))
force2k[(mygrid.Krsq .< 2.0^2) .| (mygrid.Krsq .> 20.0^2) .| (Kr .< 1) ] .= 0
ε0 = FourierFlows.parsevalsum(force2k .* mygrid.invKrsq/2.0,
                                mygrid) / (mygrid.Lx*mygrid.Ly)
force2k .= ε/ε0 * force2k
function calcF!(Fh, sol, t, cl, args...)
    eta = exp.(2π * im * rand(Float64, size(sol))) / sqrt(cl.dt)
    eta[1,1] = 0
    @. Fh = eta * sqrt(force2k)
    nothing
end

prob = TwoDNavierStokes.Problem(nx=256, Lx=2π, ν=1e-2, nν=0, dt=1e-2,
    stepper="FilteredRK4", calcF=calcF!, stochastic=true)
TwoDNavierStokes.set_zeta!(prob, GeophysicalFlows.peakedisotropicspectrum(mygrid, 2, 0.5))


using Distributed
addprocs()
using SharedArrays

us = SharedArray{Float64}(256, 256, 400)
vs = SharedArray{Float64}(256, 256, 400)
zs = SharedArray{Float64}(256, 256, 400);

@time stepforward!(prob, round(Int, 500 / prob.clock.dt))
@time for i in 1:400
    stepforward!(prob, 20); TwoDTurb.updatevars!(prob)
    vs[:,:,i] = prob.vars.v
    us[:,:,i] = prob.vars.u
    zs[:,:,i] = prob.vars.zeta
end

heatmap(xs, ys, zs[:,:,1];
    color=:viridis, aspect_ratio=1, xlim=extrema(xs), ylim=extrema(ys))

@everywhere using CoherentStructures, OceanTools
const CS = CoherentStructures
const OT = OceanTools
ts = range(0.0, step=20prob.clock.dt, length=400)
const uv = OT.ItpMetadata(xs, ys, ts, (us, vs), OT.periodic, OT.periodic, OT.flat);

vortices, singularities, bg = CS.materialbarriers(
       uv_trilinear, xs, ys, range(0.0, stop=5.0, length=11),
       LCSParameters(boxradius=π/2, indexradius=0.1, pmax=1.4,
                     merge_heuristics=[combine_20_aggressive]),
       p=uv, on_torus=true);

plot_vortices(vortices, singularities, [-π, -π], [π, π];
    bg=bg, xspan=xs, yspan=ys, include_singularities=true, barrier_width=4, barrier_color=:red,
    colorbar=:false, aspect_ratio=1)

plot_vortices(vortices, singularities, [-π, -π], [π, π];
    bg=zs[:,:,1], xspan=xs, yspan=ys, logBg=false, include_singularities=false, barrier_width=3, barrier_color=:red,
    colorbar=:false, aspect_ratio=1)

vortexflow = vortex -> flow(uv_trilinear, vortex, [0., 5.]; p=uv)[end]
plot_vortices(vortexflow.(vortices), singularities, [-π, -π], [π, π];
    bg=zs[:,:,26], xspan=xs, yspan=ys, logBg=false, include_singularities=false, barrier_width=3, barrier_color=:red,
    colorbar=:false, aspect_ratio=1)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

