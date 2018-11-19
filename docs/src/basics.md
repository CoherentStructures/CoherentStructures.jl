# Basics

## Dynamical Systems Utilities

```@meta
CurrentModule = CoherentStructures
```
### Definition of vector fields

`CoherentStructures.jl` is set up for handling two- and three-dimensional dynamical
systems only. For such low-dimensional flows it is advantageous (for top
performance) to obey the following syntax:
```
function vectorfield2d(u, p, t)
    du1 = ... # equation for $\dot{x}$
    du2 = ... # equation for $\dot{y}$
    return StaticArrays.SVector{2}(du1, du2)
end
```
and correspondingly for three-dimensional ODEs:
```
function vectorfield3d(u, p, t)
    du1 = ... # equation for $\dot{x}$
    du2 = ... # equation for $\dot{y}$
    du3 = ... # equation for $\dot{z}$
    return StaticArrays.SVector{3}(du1, du2, du3)
end
```
Furthermore, there are convenience macros to define two-dimensional velocity
and vorticity fields from stream functions.
```@docs
@define_stream
@velo_from_stream
@var_velo_from_stream
@vorticity_from_stream
```
In fact, two of the predefined velocity fields, the rotating double gyre
`rot_double_gyre`, and the Bickley jet flow `bickleyJet`, are generated from
these macros.

Another typical use case is when velocities are given as a data set. In this
case, one first interpolates the velocity components with [`interpolateVF`](@ref)
to obtain callable interpolation functions, say, `UI` and `VI`. The corresponding
vector field is then [`interp_rhs`](@ref), into which the velocity interpolants
enter via the parameter argument `p`; see below for examples.

```@docs
interpolateVF
interp_rhs
interp_rhs!
```

### Flow maps

```@docs
flow
```

```@docs
parallel_flow
```

### Linearized flow map

```@docs
linearized_flow
```

### Cauchy-Green and other pullback tensors

```@docs
parallel_tensor
```

```@docs
CG_tensor
```

```@docs
mean_diff_tensor
```

```@docs
av_weighted_CG_tensor
```

```@docs
pullback_tensors
pullback_metric_tensor
pullback_diffusion_tensor
pullback_SDE_diffusion_tensor
```
A second-order symmetric two-dimensional tensor (field) may be diagonalized
(pointwise), ie., an eigendecomposition is computed, by the following function.

```@docs
tensor_invariants
```

## Distance computations

To compute distances w.r.t. standard metrics, there exists the excellent
[`Distance.jl`](https://github.com/JuliaStats/Distances.jl) package. The
Euclidean distance between two points is computed by any of the following lines:
```
using Distances
x, y = rand(3), rand(3)
evaluate(Euclidean(),x,y)
euclidean(x,y)
```
Other metrics of potential interest include `Haversine(r)`, the geodesic
distance of two points on the sphere with radius `r`. In `CoherentStructures.jl`,
there are two more types of metrics implemented:
```@docs
PEuclidean
STmetric
```
That is, the distance on a periodic torus/cylinder, and a spatiotemporal metric
that interprets vectors as concatenated trajectories, applies the spatial metric
to each time instance and reduces the vector of spatial distances by computing
its ``l_p``-mean. Notably, `p` may be any "real" number, including `Inf` and
`-Inf` for the maximum- and "minimum"-norm. The spatiotemporal metric is a
mathematical metric only for ``p\geq 1``, in which case it smoothly operates
with efficient sparsification methods like `BallTree` and `inrange` as
implemented in the [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl)
package.
