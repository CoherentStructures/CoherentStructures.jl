using Distributed
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq, StaticArrays

using JLD2
JLD2.@load("Ocean_geostrophic_velocity.jld2")
const VI = interpolateVF(Lon, Lat, Time, UT, VT)

begin
    import AxisArrays
    const AA = AxisArrays
    q = 91
    t_initial = minimum(Time)
    t_final = t_initial + 90
    const tspan = range(t_initial, stop=t_final, length=q)
    xmin, xmax, ymin, ymax = -4.0, 7.5, -37.0, -28.0
    nx = 300
    ny = floor(Int, (ymax - ymin) / (xmax - xmin) * nx)
    xspan = range(xmin, stop=xmax, length=nx)
    yspan = range(ymin, stop=ymax, length=ny)
    P = AA.AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
    const δ = 1.e-5
    mCG_tensor = u -> av_weighted_CG_tensor(interp_rhs, u, tspan, δ;
        p=VI, tolerance=1e-6, solver=Tsit5())
end

C̅ = pmap(mCG_tensor, P; batch_size=ny)
p = LCSParameters(5*max(step(xspan), step(yspan)), 2.5, true, 60, 0.5, 2.0, 1e-4)
vortices, singularities = ellipticLCS(C̅, p)

using Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));
            aspect_ratio=1, color=:viridis, leg=true,
            title="DBS field and transport barriers")
scatter!(getcoords(singularities), color=:red)
for vortex in vortices
    plot!(vortex.curve, color=:yellow, w=3, label="T = $(round(vortex.p, digits=2))")
    scatter!(vortex.core, color=:yellow)
end
Plots.plot(fig)

using CoherentStructures
import JLD2, OrdinaryDiffEq, Plots

#Import and interpolate ocean dataset
#The @load macro initializes Lon,Lat,Time,UT,VT
JLD2.@load("../../examples/Ocean_geostrophic_velocity.jld2")
VI = interpolateVF(Lon, Lat, Time, UT, VT)

#Define a flow function from it
t_initial = minimum(Time)
t_final = t_initial + 90
times = [t_initial, t_final]
flow_map = u0 -> flow(interp_rhs, u0, times;
    p=VI, tolerance=1e-5, solver=OrdinaryDiffEq.BS5())[end]

LL = [-4.0, -34.0]
UR = [6.0, -28.0]
ctx, _  = regularTriangularGrid((150, 90), LL, UR)
bdata = getHomDBCS(ctx, "all");

M = assembleMassMatrix(ctx, bdata=bdata)
S0 = assembleStiffnessMatrix(ctx)
S1 = adaptiveTOCollocationStiffnessMatrix(ctx, flow_map)

S = applyBCS(ctx, 0.5(S0 + S1), bdata);

using Arpack

λ, v = eigs(S, M, which=:SM, nev=6);

using Clustering

ctx2, _ = regularTriangularGrid((200, 120), LL, UR)
v_upsampled = sample_to(v, ctx, ctx2, bdata=bdata)

function iterated_kmeans(numiterations, args...)
    best = kmeans(args...)
    for i in 1:(numiterations - 1)
        cur = kmeans(args...)
        if cur.totalcost < best.totalcost
            best = cur
        end
    end
    return best
end

n_partition = 4
res = iterated_kmeans(20, permutedims(v_upsampled[:,1:(n_partition-1)]), n_partition)
u = kmeansresult2LCS(res)
u_combined = sum([u[:,i] * i for i in 1:n_partition])
plot_u(ctx2, u_combined, 200, 200;
    color=:viridis, colorbar=:none, title="$n_partition-partition of Ocean Flow")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

