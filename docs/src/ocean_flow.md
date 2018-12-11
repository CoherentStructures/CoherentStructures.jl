# Geostrophic Ocean Flow
For a more realistic application, we consider an unsteady ocean surface velocity
data set obtained from satellite altimetry measurements produced by SSALTO/DUACS
and distributed by AVISO. The particular space-time window has been used several
times in the literature.

## Geodesic vortices

Here, we demonstrate how to detect material barriers to diffusive transport.
```
using Distributed, AxisArrays
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq, StaticArrays

###################### load and interpolate velocity data sets #############
using JLD2
JLD2.@load("Ocean_geostrophic_velocity.jld2")
const VI = interpolateVF(Lon, Lat, Time, UT, VT)

############################ integration set up ################################
@everywhere begin
    q = 91
    t_initial = minimum(Time)
    t_final = t_initial + 90
    const tspan = range(t_initial, stop=t_final, length=q)
    nx = 500
    ny = Int(floor(0.6 * nx))
    xmin, xmax, ymin, ymax = -4.0, 6.0, -34.0, -28.0
    xspan = range(xmin, stop=xmax, length=nx)
    yspan = range(ymin, stop=ymax, length=ny)
    P = SVector{2}.(xspan, yspan')
    const δ = 1.e-5
    mCG_tensor = u -> av_weighted_CG_tensor(interp_rhs, u, tspan, δ;
        p=VI, tolerance=1e-6, solver=Tsit5())
end

############################ compute elliptic LCSs #############################
C̅ = AxisArray(pmap(mCG_tensor, P; batch_size=ny), xspan, yspan)
p = LCSParameters(3*max(step(xspan), step(yspan)), 2.0, 60, 0.5, 2.0, 1e-4)
vortices, singularities = ellipticLCS(C̅, p)
```
The result is visualized as follows:
```
using Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));
            aspect_ratio=1, color=:viridis, leg=true,
            title="DBS field and transport barriers")
scatter!(get_coords(singularities), color=:red)
for vortex in vortices
    plot!(vortex.curve, color=:yellow, w=3, label="T = $(round(vortex.p, digits=2))")
    scatter!(vortex.core, color=:yellow)
end
Plots.plot(fig)
```

## FEM-based methods

Here we showcase how the adaptive TO method can be used to calculate coherent sets.

First we setup the problem.
```@example 5
using CoherentStructures
import JLD2, OrdinaryDiffEq, Plots

#Import and interpolate ocean dataset
#The @load macro initializes Lon,Lat,Time,UT,VT
JLD2.@load("../../examples/Ocean_geostrophic_velocity.jld2")
UI, VI = interpolateVF(Lon, Lat, Time, UT, VT)
p = (UI,VI)

#Define a flow function from it
t_initial = minimum(Time)
t_final = t_initial + 90
times = [t_initial, t_final]
flow_map = u0 -> flow(interp_rhs, u0, times;
    p=p, tolerance=1e-5, solver=OrdinaryDiffEq.BS5())[end]
```

Next we set up the domain. We want to use zero Dirichlet boundary conditions here.
```@example 5
LL = [-4.0, -34.0]
UR = [6.0, -28.0]
ctx, _  = regularTriangularGrid((150, 90), LL, UR)
bdata = getHomDBCS(ctx, "all");
```

For the TO method, we seek generalized eigenpairs involving the bilinear form

$a_h(u,v) = \frac{1}{2} \left(a_0(u,v) + a_1(I_h u, I_h v) \right).$

Here $a_0$ is the weak form of the Laplacian on the initial domain, and $a_1$ is the weak form of the Laplacian on the final domain.
The operator $I_h$ is an interpolation operator onto the space of test functions on the final domain.

For the adaptive TO method, we use pointwise nodal interpolation (i.e. collocation) and the mesh on the final domain is obtained by doing
a Delaunay triangulation on the images of the nodal points of the initial domain.
This results in the representation matrix of $I_h$ being the identity, so in matrix form we get:

$S = 0.5(S_0 + S_1)$

where $S_0$ is the stiffness matrix for the triangulation at initial time, and $S_1$ is the stiffness matrix for the triangulation at final time.
```@example 5
M = assembleMassMatrix(ctx, bdata=bdata)
S0 = assembleStiffnessMatrix(ctx)
S1 = adaptiveTOCollocationStiffnessMatrix(ctx, flow_map)

#Average matrices and apply boundary conditions
S = applyBCS(ctx, 0.5(S0 + S1), bdata);
```

We can now solve the eigenproblem.
```@example 5
using Arpack

λ, v = eigs(S, M, which=:SM, nev=6);
```
We upsample the eigenfunctions and then cluster.
```@example 5
using Clustering

ctx2,_ = regularTriangularGrid((200, 120), LL, UR)
v_upsampled = sample_to(v, ctx, ctx2, bdata=bdata)

#Run k-means several times, keep the best result
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
```
