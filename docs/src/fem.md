# FEM-based Methods

These methods rely on the theory outlined by Froyland's [*Dynamical Laplacian*]
(http://arxiv.org/pdf/1411.7186v4.pdf)
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

## Example

Here is an example of how one might use these methods.
```@example 4
using CoherentStructures
LL = [0.0,0.0]; UR = [1.0,1.0];
ctx = regularTriangularGrid((50,50),LL,UR)

A = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-10,tolerance= 1.e-4)
K = assembleStiffnessMatrix(ctx,A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K,M,which=:SM);
```
Here we have a time-dependent velocity field that describes the
[rotating double gyre](http://dx.doi.org/10.1137/100794110) dynamical system.
This velocity field is given by the `rot_double_gyre` function. The second
argument to `mean_diff_tensor` are the times at which we average the pullback
diffusion tensors. The third parameter is the step size δ used for the
finite-difference scheme, `tolerance` is passed to the ODE solver from
[DifferentialEquations.jl](http://juliadiffeq.org/). In the above, `A(x)`
approximates the mean diffusion tensor given by

$A(x) = \sum_{t \in \mathcal T}(D\Phi^t(x))^{-1} (D\Phi^t x)^{-T}.$

The eigenfunctions saved in `v` approximate those of $\Delta^{dyn}$
```@example 4
import Plots
res = [plot_u(ctx, v[:,i],colorbar=:none,clim=(-3,3)) for i in 1:6];
Plots.plot(res...,margin=-10Plots.px)
```
Looking at the spectrum, there appears a gap after the third eigenvalue:
```@example 4
Plots.scatter(range(1,6), real.(λ))
```
We can use the [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) package to compute coherent structures from the first two nontrivial eigenfunctions:
```@example 4
using Clustering
numclusters=2
res = kmeans(v[:,2:numclusters+1]',numclusters+1)
u = kmeansresult2LCS(res)
Plots.plot([plot_u(ctx,u[:,i],200,200,color=:viridis) for i in [1,2,3]]...)

```
## Features
TODO: finish this, describe CG and TO-based approaches, supported elements and grids, etc...
## The `gridContext` Type

The FEM-based methods of `CoherentStructures.jl` rely heavily on the [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) package.
This package is very low-level and does not provide point-location/plotting functionality.
To be able to more conveniently work with the specific types of grids that we need, all necessary variables for a single grid are combined in a `gridContext` structure - including the grid points, the quadrature formula used and the type of element used (e.g. Triangular P1, Quadrilateral P2, etc..). This makes it easier to assemble mass/stiffness matrices, and provides an interface for point-location and plotting.

In this documentation, the variable name `ctx` is exclusively used for `gridContext` objects.

See also [Constructing Grids](@ref) in the [FEM-API](@ref) section.

### Node ordering and dof ordering

Finite Element methods work with degrees of freedom (dof), which are elements of some dual space.
For nodal finite elements, these correspond to evaluation functionals at the nodes of the grid.

The nodes of the grid can be obtained in the following way `[n.x for n in ctx.grid.nodes]`.
However, most of the methods of this package do _not_ return results in this order, but instead
use `JuAFEM.jl`'s dof-ordering.

See also the documentation in [`dof2node`](@ref) and [`CoherentStructures.gridContext`](@ref)

When working with (non-natural) [Boundary Conditions](@ref), the ordering is further changed, due to there being fewer degrees of freedom in total.

## Assembly

See [Stiffness and Mass Matrices](@ref) from the [API](@ref) section.

## Evaluating Functions in the Approximation Space

given a series of coefficients that represent a function in the approximation space, to evaluate a function at a point, use the `evaluate_function_from_nodevals` or `evaluate_function_from_dofvals` functions.
```@example 4
ctx = regularP2TriangularGrid((10,10))
u = zeros(ctx.n)
u[45] = 1.0
Plots.heatmap(linspace(0,1,200),linspace(0,1,200), (x,y)->evaluate_function_from_nodevals(ctx,u,[x,y]))
```
For more details, consult the API: [`evaluate_function_from_dofvals`](@ref), [`evaluate_function_from_nodevals`](@ref)

## Nodal Interpolation

To perform nodal interpolation of a grid, use the [`nodal_interpolation`](@ref) function.

## Boundary Conditions

To use something other than the natural homogeneous von Neumann boundary conditions, the `CoherentStructures.boundaryData` type can be used. This currently supports combinations of homogeneous Dirichlet and periodic boundary conditions.
 - Homogeneous Dirichlet BCs require rows and columns of the stiffness/mass matrices to be deleted
 - Periodic boundary conditions require rows and columns of the stiffness/mass matrices to be added to each other.

 This means that the coefficient vectors for elements of the approximation space that satisfy the boundary conditions are potentially smaller and in a different order. Given a `bdata` argument, functions like `plot_u` will take this into account.

### Constructing Boundary Conditions

Natural von-Neumann boundary conditions can be constructed with:
`boundaryData()` and are generally the default

Homogeneous Dirichlet boundary conditions can be constructed with the `getHomDBCS(ctx,[which="all"])` function. The optional `which` parameter is a vector of strings, corresponding to `JuAFEM` face-sets, e.g. `getHomDBCS(ctx,which=["left","right"])`

Periodic boundary conditions are constructed by calling `boundaryData(ctx,predicate,[which_dbc=[]])`. The argument `predicate` is a function that should return `true` if and only if two points should be identified. Due to floating-point rounding errors, note that using exact comparisons (`==`) should be avoided. Only points that are in `JuAFEM.jl` boundary facesets are considered. If this is too restrictive, use the `boundaryData(dbc_dofs, periodic_dofs_from,periodic_dofs_to)` constructor.

For details, see [`boundaryData`](@ref)


### Example

Here we apply Homogeneous DBC to top and bottom, and identify the left and right side:
```@example 4

ctx = regularQuadrilateralGrid((10,10))
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && (abs((p1[1] - p2[1])%1.0) < 1e-10)
bdata = boundaryData(ctx,predicate,["top","bottom"])
u = ones(nDofs(ctx,bdata))
u[20] = 2.0; u[38] = 3.0; u[56] = 4.0
plot_u(ctx,u,200,200,bdata=bdata,colorbar=:none)
```

To apply boundary conditions to a stiffness/mass matrix, use the `applyBCS` function. Note that `assembleStiffnessMatrix` and `assembleMassMatrix` take a `bdata` argument that does this internally.

## Plotting and Videos

There are some helper functions that exist for making plots and videos of functions on grids. These rely on the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) library. Plotting recipes are unfortunately not implemented.

The simplest way to plot is using the [`plot_u`](@ref) function. Plots and videos of eulerian plots like `` f \circ \Phi^0_t `` can be made with the [`plot_u_eulerian`](@ref) and  [`eulerian_videos`](@ref) functions.

## Parallelisation

Many of the plotting functions support parallelism internally.
Tensor fields can be constructed in parallel, and then passed to [`assembleStiffnessMatrix`](@ref). For an example that does this, see
TODO: Add this example
