# Basics

```@meta
CurrentModule = CoherentStructures
```
## Definition of vector fields

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
In a companion package [`StreamMacros.jl`](https://github.com/CoherentStructures/StreamMacros.jl), there are convenience macros to define two-dimensional velocity
and vorticity fields from stream functions.

Another typical use case is when velocities are given as a data set. In this
case, one first interpolates the velocity components with [`interpolateVF`](@ref)
to obtain a callable interpolation function, say, `UI`. The corresponding vector
field is then [`interp_rhs`](@ref), into which the velocity interpolant enters
via the parameter argument `p`; see below for examples.

```@docs
interpolateVF
interp_rhs
interp_rhs!
```

## (Linearized) Flow map

```@docs
flow
linearized_flow
```

## Cauchy-Green and other pullback tensors

```@docs
CG_tensor
mean_diff_tensor
av_weighted_CG_tensor
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

For computing distances w.r.t. standard metrics, there exists the excellent
[`Distances.jl`](https://github.com/JuliaStats/Distances.jl) package. For example,
the Euclidean distance between two points is computed by any of the last two lines:
```
using Distances
x, y = rand(3), rand(3)
Euclidean()(x, y)
euclidean(x, y)
```
Other metrics of potential interest include `Haversine(r)`, the geodesic
distance of two points on the sphere with radius `r`, and `PeriodicEuclidean(L)`,
the distance on a torus or cylinder. Here, `L` is a vector containing the period
of each dimension, `Inf` for non-periodic dimensions.
```@docs
Distances.Euclidean
Distances.Haversine
Distances.PeriodicEuclidean
```
In `CoherentStructures.jl`, there is one more metric type implemented:
```@docs
STmetric
```
This is a spatiotemporal metric that operates on trajectories in the format of
vectors of static vectors, i.e., `Vector{<:SVector}`. Each vector element
corresponds to the state vector. The `STmetric` then computes the
``\\ell_p``-mean of the spatial distances over time. Notably, `p` may be any
"real" number, including `Inf` and `-Inf` for the maximum- and "minimum"-norm.
