using Distributed
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq

using JLD2
JLD2.@load("Ocean_geostrophic_velocity.jld2")
const uv = interpolateVF(Lon, Lat, Time, UT, VT)

q = 91
t_initial = minimum(Time)
t_final = t_initial + 90
const tspan = range(t_initial, stop=t_final, length=q)
xmin, xmax, ymin, ymax = -4.0, 7.5, -37.0, -28.0
nx = 300
ny = floor(Int, (ymax - ymin) / (xmax - xmin) * nx)
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = tuple.(xspan, permutedims(yspan))
const δ = 1.e-5
mCG_tensor = u -> av_weighted_CG_tensor(interp_rhs, u, tspan, δ;
    p=uv, tolerance=1e-6, solver=Tsit5())

C̅ = pmap(mCG_tensor, P; batch_size=ceil(Int, length(P)/nprocs()^2))
p = LCSParameters(2.5)
vortices, singularities = ellipticLCS(C̅, xspan, yspan, p)

area.(vortices), clockwise.(vortices, interp_rhs, t_initial, p=uv)

using Plots
trace = tensor_invariants(C̅)[5]
fig = plot_vortices(vortices, singularities, (xmin, ymin), (xmax, ymax);
    bg=trace, xspan=xspan, yspan=yspan, title="DBS field and transport barriers", showlabel=true)
Plots.plot(fig)

using Interpolations, Tensors, StaticArrays

const V = scale(interpolate(SVector{2}.(UT[:,:,1], VT[:,:,1]), BSpline(Quadratic(Free(OnGrid())))), Lon, Lat)

rate_of_strain_tensor(xin) = let V=V
    x, y = xin
    grad = Interpolations.gradient(V, x, y)
    symmetric(Tensor{2,2}((grad[1][1], grad[1][2], grad[2][1], grad[2][2])))
end

xmin, xmax, ymin, ymax = -12.0, 7.0, -38.1, -22.0
nx = 950
ny = floor(Int, (ymax - ymin) / (xmax - xmin) * nx)
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = tuple.(xspan, permutedims(yspan))

S = rate_of_strain_tensor.(P)
p = LCSParameters(boxradius=2.5, pmin=-1, pmax=1, merge_heuristics=[Combine20(), Combine31()])
vortices, singularities = ellipticLCS(S, xspan, yspan, p; outermost=true)

λ₁ = tensor_invariants(S)[1]
fig = plot_vortices(vortices, singularities, (xmin, ymin), (xmax, ymax);
    bg=λ₁, xspan=xspan, yspan=yspan, logBg=false, title="Minor rate-of-strain field and OECSs")
Plots.plot(fig)

using CoherentStructures
import JLD2, OrdinaryDiffEq, Plots

JLD2.@load("Ocean_geostrophic_velocity.jld2")

const UV = interpolateVF(Lon, Lat, Time, UT, VT)

t_initial = minimum(Time)
t_final = t_initial + 90
const times = [t_initial, t_final]
flow_map(u0) = flow(interp_rhs, u0, times; p=UV, tolerance=1e-5, solver=OrdinaryDiffEq.BS5())[end]

LL = (-4.0, -34.0)
UR = (6.0, -28.0)
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
fig = plot_u(ctx2, u_combined, 200, 200;
    color=:viridis, colorbar=:none, title="$n_partition-partition of Ocean Flow")

Plots.plot(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
