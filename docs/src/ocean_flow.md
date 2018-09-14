# Geostrophic Ocean Flow
For a more realistic application, we consider an unsteady ocean surface velocity
data set obtained from satellite altimetry measurements produced by SSALTO/DUACS
and distributed by AVISO. The particular space-time window has been used several
times in the literature.

## Geodesic vortices

Here, we demonstrate how to detect material barriers to diffusive transport.
```@example 1
using CoherentStructures
import JLD2, OrdinaryDiffEq, Plots
###################### load and interpolate velocity data sets #############
JLD2.@load("../../examples/Ocean_geostrophic_velocity.jld2")

UI, VI = interpolateVF(Lon,Lat,Time,UT,VT)
p = (UI,VI)

############################ integration set up ################################
q = 91
t_initial = minimum(Time)
t_final = t_initial + 90
const tspan = range(t_initial,stop=t_final,length=q)
nx = 500
ny = Int(floor(0.6*nx))
N = nx*ny
xmin, xmax, ymin, ymax = -4.0, 6.0, -34.0, -28.0
xspan, yspan = range(xmin,stop=xmax,length=nx), range(ymin,stop=ymax,length=ny)
P = vcat.(xspan',yspan)
const δ = 1.e-5
mCG_tensor = u -> av_weighted_CG_tensor(interp_rhs,u,tspan,δ,p = p,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())

C̅ = map(mCG_tensor,P)
LCSparams = (.09, 0.5, 0.05, 0.5, 1.0, 60)
vals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);
```
The result is visualized as follows:
```@example 1
using Statistics
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))
fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂),aspect_ratio=1,color=:viridis,
            title="DBS-field and transport barriers",xlims=(xmin, xmax),ylims=(ymin, ymax),leg=true)
for i in eachindex(orbits)
    Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],digits=2))")
end
Plots.plot(fig)
```


