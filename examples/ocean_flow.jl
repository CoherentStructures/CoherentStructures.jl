#md # ```@meta
#md #   EditURL = "../../../examples/ocean_flow.jl"
#md # ```
# # Geostrophic Ocean Flow
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`ocean_flow.ipynb`](https://nbviewer.jupyter.org/github/CoherentStructures/CoherentStructures.jl/blob/gh-pages/dev/generated/ocean_flow.ipynb),
#md #     and as an executable julia file
#md #     [`ocean_flow.jl`](https://raw.githubusercontent.com/CoherentStructures/CoherentStructures.jl/gh-pages/dev/generated/ocean_flow.jl).
#md #
# For a more realistic application, we consider an unsteady ocean surface velocity
# data set obtained from satellite altimetry measurements produced by SSALTO/DUACS
# and distributed by AVISO. The particular space-time window has been used several
# times in the literature.

# Below is a video showing advection of the initial 90-day DBS field for 90 days.

#md # ```@raw html
# <video controls="" height="100%" width="100%">
#  <source src="https://raw.githubusercontent.com/natschil/misc/master/videos/ocean_flow.mp4" type="video/mp4" />
# Your browser does not support the video tag.
# </video>
#md # ```

# ## Geodesic vortices

# Here, we demonstrate how to detect material barriers to diffusive transport.

using Distributed
nprocs() == 1 && addprocs()

@everywhere using CoherentStructures, OrdinaryDiffEq, StaticArrays

# Next, we load and interpolate the velocity data sets.

using JLD2
JLD2.@load(OCEAN_FLOW_FILE)
const VI = interpolateVF(Lon, Lat, Time, UT, VT)

# Since we want to use parallel computing, we set up the integration LCSParameters
# on all workers, i.e., `@everywhere`.

import AxisArrays
const AA = AxisArrays
q = 91
const t_initial = minimum(Time)
const t_final = t_initial + 90
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

# Now, compute the averaged weighted Cauchy-Green tensor field and extract elliptic LCSs.

C̅ = pmap(mCG_tensor, P; batch_size=ny)
p = LCSParameters(2.5)
vortices, singularities = ellipticLCS(C̅, p)

# Finally, the result is visualized as follows.

using Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
fig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));
            aspect_ratio=1, color=:viridis, leg=true,
            title="DBS field and transport barriers",
            xlims=(xmin, xmax), ylims=(ymin, ymax))
scatter!(getcoords(singularities), color=:red, label="singularities")
scatter!([vortex.center for vortex in vortices], color=:yellow, label="vortex cores")
for vortex in vortices, barrier in vortex.barriers
    plot!(barrier.curve, w=2, label="T = $(round(barrier.p, digits=2))")
end
DISPLAY_PLOT(fig, ocean_flow_geodesic_vortices)

# ## Objective Eulerian coherent structures (OECS)

# With only minor modifications, we are also able to compute OECSs. We start by
# loading some packages and define the rate-of-strain tensor function.

using Interpolations, Tensors
function rate_of_strain_tensor(xin)
    x, y = xin
    grad = Interpolations.gradient(VI, x, y, t_initial)
    df =  Tensor{2,2}((grad[1][1], grad[1][2], grad[2][1], grad[2][2]))
    return symmetric(df)
end

# To make live more exciting, we choose a larger domain.

xmin, xmax, ymin, ymax = -12.0, 7.0, -38.1, -22.0
nx = 950
ny = floor(Int, (ymax - ymin) / (xmax - xmin) * nx)
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = AA.AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)

# Next, we evaluate the rate-of-strain tensor on the grid and compute OECSs.

S = map(rate_of_strain_tensor, P)
p = LCSParameters(boxradius=2.5, indexradius=0.25, pmin=-1, pmax=1, rdist=2.5e-3)
vortices, singularities = ellipticLCS(S, p, outermost=true)

λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(S)
fig = Plots.heatmap(xspan, yspan, permutedims((λ₁));
            aspect_ratio=1, color=:viridis, leg=true,
            title="Minor eigenvalue",
            xlims=(xmin, xmax), ylims=(ymin, ymax)
            )
scatter!([s.coords for s in singularities if s.index == 1//2 ], color=:yellow, label="wedge")
scatter!([s.coords for s in singularities if s.index == -1//2 ], color=:purple, label="trisector")
scatter!([s.coords for s in singularities if s.index == 1 ], color=:white, label="elliptic")
for vortex in vortices, barrier in vortex.barriers
    plot!(barrier.curve, w=2, label="")
end
scatter!(vec(SVector{2}.(Lon, Lat')), color=:red, label="", alpha=0.1)
DISPLAY_PLOT(fig, ocean_flow_oecs)

# ## FEM-based methods

# Here we showcase how the adaptive TO method can be used to calculate coherent sets.
#
# First we setup the problem.

using CoherentStructures
import JLD2, OrdinaryDiffEq, Plots

#Import and interpolate ocean dataset
#The @load macro initializes Lon,Lat,Time,UT,VT

JLD2.@load(OCEAN_FLOW_FILE)

VI = interpolateVF(Lon, Lat, Time, UT, VT)

#Define a flow function from it
t_initial = minimum(Time)
t_final = t_initial + 90
times = [t_initial, t_final]
flow_map = u0 -> flow(interp_rhs, u0, times;
    p=VI, tolerance=1e-5, solver=OrdinaryDiffEq.BS5())[end]

# Next we set up the domain. We want to use zero Dirichlet boundary conditions here.

LL = [-4.0, -34.0]
UR = [6.0, -28.0]
ctx, _  = regularTriangularGrid((150, 90), LL, UR)
bdata = getHomDBCS(ctx, "all");

# For the TO method, we seek generalized eigenpairs involving the bilinear form
#
# ```math
# a_h(u,v) = \frac{1}{2} \left(a_0(u,v) + a_1(I_h u, I_h v) \right).
# ```
#
# Here, $a_0$ is the weak form of the Laplacian on the initial domain, and $a_1$
# is the weak form of the Laplacian on the final domain. The operator $I_h$ is an
# interpolation operator onto the space of test functions on the final domain.
#
# For the adaptive TO method, we use pointwise nodal interpolation (i.e. collocation)
# and the mesh on the final domain is obtained by doing a Delaunay triangulation
# on the images of the nodal points of the initial domain. This results in the
# representation matrix of $I_h$ being the identity, so in matrix form we get:
#
# ```math
# S = 0.5(S_0 + S_1)
# ```
#
# where $S_0$ is the stiffness matrix for the triangulation at initial time, and
# $S_1$ is the stiffness matrix for the triangulation at final time.

M = assembleMassMatrix(ctx, bdata=bdata)
S0 = assembleStiffnessMatrix(ctx)
S1 = adaptiveTOCollocationStiffnessMatrix(ctx, flow_map)

# Averaging matrices and applying boundary conditions yields
S = applyBCS(ctx, 0.5(S0 + S1), bdata);

# We can now solve the eigenproblem.

using Arpack

λ, v = eigs(S, M, which=:SM, nev=6);

# We upsample the eigenfunctions and then cluster.

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


DISPLAY_PLOT(fig, ocean_flow_fem)
