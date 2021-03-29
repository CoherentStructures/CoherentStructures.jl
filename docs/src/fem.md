# Finite-element discretizations of the dynamic Laplacian

These methods rely on the theory outlined by Froyland's
[*Dynamical Laplacian*](http://arxiv.org/pdf/1411.7186v4.pdf)
and the [*Geometric Heat Flow*](https://www.researchgate.net/publication/306291640_A_geometric_heat-flow_theory_of_Lagrangian_coherent_structures) of Karrasch & Keller.

The Laplace-like operators are best discretized by finite-element-based methods,
see this [paper](https://arxiv.org/pdf/1705.03640.pdf) by Froyland & Junge.

This involves the discretization of the average of a one-parameter family of
Laplace operators of the form:

$\Delta^{dyn} := \sum_{t \in \mathcal T} P_t^* \Delta P_t$

for a finite series of times $\mathcal T$, where $P_t$ is the transfer-operator
for the flow at time $t$ (in volume-preserving flows).

The resulting operator is both symmetric and uniformly elliptic. Eigenfunctions
of $\Delta^{dyn}$ can be used to find Lagrangian Coherent Structures.

See the [Examples](@ref examples_section) section for examples of how these methods can be used.

## Features

### CG and TO methods

The standard Galerkin formulation of the weak dynamical Laplace is referred to
as the CG-method here, due to the fact that the inverse Cauchy-Green tensor
appears in the weak formulation. This gives a bilinear form
$\overline a(u,v) := \sum_{t \in \mathcal T}a^t(P_t u, P_t v)$
Here $P_t$ is the Transfer-Operator (or pushforward) to time-$t$, and $a^t$ is
the weak-form of the Laplacian on the range of the time-$t$ map being
considered.

There is also a range of transfer operator-based approaches implemented here.
These approximate the weak form of the Dynamical-Laplace by a bilinear-form:

$\tilde a_h(u,v) = \sum_{t \in \mathcal T} a^t(I_hP_t u, I_h P_t v)$

where $I_h$ is a suitable interpolation operator depending on the mesh-width $h$. Options for $I_h$ implemented in this package are:

- collocation (pointwise interpolation):
  - points used are mesh points from domain grid ("adaptive TO"),
  - points used are arbitrary ("non-adaptive TO");
- the $L^2$-orthogonal projection onto an FEM-space:
  - using the forward-flow map (currently gives poor results),
  - using the inverse flow map.

Note that the $L^2$-Galerkin methods currently perform very poorly on larger problems.

For more details, see [Froyland & Junge, 2018](https://arxiv.org/pdf/1705.03640.pdf).

### Grids

Various types of regular and irregular meshes (with Delaunay triangulation using
[`VoronoiDelaunay.jl`](https://github.com/JuliaGeometry/VoronoiDelaunay.jl) ) are supported.
These are based on the corresponding elements from [`Ferrite.jl`](https://github.com/Ferrite-FEM/Ferrite.jl) and include:

- triangular P1-Lagrange elements in 2D (all methods)
- quadrilateral P1-Lagrange elements in 2D (all methods except adaptive TO)
- triangular and quadrilateral P2-Lagrange elements in 2D (all methods except adaptive TO)
- tetrahedral P1-Lagrange elements in 3D (only CG method tested, non-adaptive TO might work also)

## The `GridContext` Type

The FEM-based methods of `CoherentStructures.jl` rely heavily on the
[`Ferrite.jl`](https://github.com/Ferrite-FEM/Ferrite.jl) package.
This package is very low-level and does not provide point-location/plotting functionality.
To be able to more conveniently work with the specific types of grids that we need, all necessary variables for a single grid are combined in a `GridContext` structure - including the grid points, the quadrature formula used and the type of element used (e.g. Triangular P1, Quadrilateral P2, etc..). This makes it easier to assemble mass/stiffness matrices, and provides an interface for point-location and plotting.

In this documentation, the variable name `ctx` is exclusively used for `GridContext` objects.

See also [Constructing Grids](@ref) in the [FEM-API](@ref) section.

### Node ordering and dof ordering

Finite Element methods work with degrees of freedom (dof), which are elements of some dual space.
For nodal finite elements, these correspond to evaluation functionals at the nodes of the grid.

The nodes of the grid can be obtained in the following way `[n.x for n in ctx.grid.nodes]`.
However, most of the methods of this package do _not_ return results in this order, but instead
use `Ferrite.jl`'s dof-ordering.

See also the documentation in [`dof2node`](@ref) and [`CoherentStructures.GridContext`](@ref)

When working with (non-natural) [Boundary Conditions](@ref), the ordering is further changed, due to there being fewer degrees of freedom in total.

## Assembly

See [Stiffness and Mass Matrices](@ref) from the [FEM-API](@ref) section.

## Evaluating Functions in the Approximation Space

given a series of coefficients that represent a function in the approximation space, to evaluate a function at a point, use the `evaluate_function_from_node_or_cellvals` or `evaluate_function_from_dofvals` functions.

```@example 6
using CoherentStructures #hide
using Plots, Tensors
ctx, _ = regularP2TriangularGrid((10, 10))
u = zeros(ctx.n)
u[45] = 1.0
Plots.heatmap(range(0, stop=1, length=200), range(0, stop=1, length=200),
    (x, y) -> evaluate_function_from_dofvals(ctx, u, Vec(x, y)))
```

For more details, consult the API: [`evaluate_function_from_dofvals`](@ref),
[`evaluate_function_from_node_or_cellvals`](@ref),
[`evaluate_function_from_node_or_cellvals_multiple`](@ref)

## Nodal Interpolation

To perform nodal interpolation of a grid, use the [`nodal_interpolation`](@ref)
function.

## Boundary Conditions

To use something other than the natural homogeneous von Neumann boundary
conditions, the `BoundaryData` type can be used. This currently
supports combinations of homogeneous Dirichlet and periodic boundary conditions.

- Homogeneous Dirichlet BCs require rows and columns of the stiffness/mass
  matrices to be deleted
- Periodic boundary conditions require rows and columns of the stiffness/mass
  matrices to be added to each other.

This means that the coefficient vectors for elements of the approximation space
that satisfy the boundary conditions are potentially smaller and in a different
order. Given a `bdata` argument, functions like `plot_u` will take this into
account.

### Constructing Boundary Conditions

Natural von-Neumann boundary conditions can be constructed with `BoundaryData()`
and are generally the default.

Homogeneous Dirichlet boundary conditions can be constructed with the
`getHomDBCS(ctx[, which="all"])` function. The optional `which` parameter is a
vector of strings, corresponding to `Ferrite` face-sets, e.g.
`getHomDBCS(ctx, which=["left", "right"])`

Periodic boundary conditions are constructed by calling
`BoundaryData(ctx,predicate,[which_dbc=[]])`. The argument `predicate` is a
function that should return `true` if and only if two points should be identified.
Due to floating-point rounding errors, note that using exact comparisons (`==`)
should be avoided. Only points that are in `Ferrite.jl` boundary facesets are
considered. If this is too restrictive, use the
`BoundaryData(dbc_dofs, periodic_dofs_from, periodic_dofs_to)` constructor.

For details, see [`BoundaryData`](@ref).

### Example

Here we apply homogeneous DBC to top and bottom, and identify the left and right side:

```@example 6
using CoherentStructures, Distances, Plots
ctx, _ = regularQuadrilateralGrid((10, 10))
predicate = (p1, p2) -> peuclidean(p1, p2, [1.0, Inf]) < 2e-10
bdata = BoundaryData(ctx, predicate, ["top", "bottom"])
u = ones(nBCDofs(ctx, bdata))
u[20] = 2.0; u[38] = 3.0; u[56] = 4.0
plot_u(ctx, u, 200, 200, bdata=bdata, colorbar=:none)
```

To apply boundary conditions to a stiffness/mass matrix, use the `applyBCS` function. Note that `assembleStiffnessMatrix` and `assembleMassMatrix` take a `bdata` argument that does this internally.

## Plotting and Videos

There are some helper functions that exist for making plots and videos of functions on grids. These rely on the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) library. Plotting recipes are unfortunately not implemented.

The simplest way to plot is using the [`plot_u`](@ref) function. Plots and videos of eulerian plots like `` f \circ \Phi^0_t `` can be made with the [`plot_u_eulerian`](@ref) and  [`eulerian_videos`](@ref) functions.

## Parallelisation

Many of the plotting functions support parallelism internally.
Tensor fields can be constructed in parallel, and then passed to [`assembleStiffnessMatrix`](@ref). For an example that does this, see
TODO: Add this example

## FEM-API

```@meta
CurrentModule = CoherentStructures
```

### Stiffness and Mass Matrices

```@docs
assembleStiffnessMatrix
assembleMassMatrix
adaptiveTOCollocationStiffnessMatrix
```

### Constructing Grids

There are several helper functions available for constructing grids. The simplest is:

#### In 1D

```@docs
regular1dP1Grid
regular1dP2Grid
```

### In 2D

```@docs
regular2dGrid
```

Supported values for the `gridType` argument are:

```@example
using CoherentStructures #hide
regular2dGridTypes
```

The following functions are conceptually similar:

```@docs
regularTriangularGrid
regularDelaunayGrid
irregularDelaunayGrid
regularP2TriangularGrid
regularP2DelaunayGrid
regularQuadrilateralGrid
regularP2QuadrilateralGrid
```

In 3D we have

```@docs
regularTetrahedralGrid
regularP2TetrahedralGrid
```

All of these methods return a `GridContext` object and a `BoundaryData` object. The latter is only relevant when using
a Delaunay grid with `on_torus==true`.

```@docs
CoherentStructures.GridContext
```

### Boundary Conditions API

```@docs
BoundaryData
getHomDBCS
undoBCS
applyBCS
```

### Helper functions

```@docs
dof2node
getDofCoordinates
```

```@docs
evaluate_function_from_dofvals
evaluate_function_from_node_or_cellvals
evaluate_function_from_node_or_cellvals_multiple
```

```@docs
nodal_interpolation
```

```@docs
getH
```

### Plotting API

#### FEM

```@docs
plot_u
plot_u_eulerian
eulerian_videos
eulerian_video
```

### Other plotting utilities

```@docs
plot_ftle
```

### Defaults

```julia
const default_quadrature_order=5
const default_solver = OrdinaryDiffEq.BS5()
```
