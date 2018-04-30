# FEM-based Methods

These methods rely on the theory outlined by Froyland's [*Dynamical Laplacian*]
(http://arxiv.org/pdf/1411.7186v4.pdf)
and the [*Geometric Heat Flow*](https://www.researchgate.net/publication/306291640_A_geometric_heat-flow_theory_of_Lagrangian_coherent_structures) of Karrasch & Keller.

This involves the discretization of an averaged heat_flow operator of the form:

$\Delta^{dyn} := \sum_{t \in \mathcal T} P^*_t \Delta P_t$

for a finite series of times $\mathcal T$, where $P_t$ is the transfer-operator for the flow at time $t$

The resulting operator is both symmetric and uniformly elliptic, and can be discretize using FEM-based methods. Eigenfunctions of $\Delta^{dyn}$ can be used to find Lagrangian Coherent Structures.

## Example

Here is an example of how one might use these methods.
```@example 1
using CoherentStructures
LL = [0.0,0.0]; UR = [1.0,1.0];
ctx = regularTriangularGrid((50,50),LL,UR)

A = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-10,tolerance= 1.e-3)
K = assembleStiffnessMatrix(ctx,A)
M = assembleMassMatrix(ctx)
λ, v = eigs(-K,M,which=:SM);
```
Here we have a time-dependent velocity field that describes the rotating double gyre dynamical system, this velocity field is given by the `rot_double_gyre` function. The second argument to `mean_diff_tensor` are the times at which we average the pullback diffusion tensors. The third parameter is the δ used for the finite-difference scheme, `tolerance` is passed to the ODE solver from [DifferentialEquations.jl](http://juliadiffeq.org/). In the above, `A(x)` approximates the mean diffusion tensor given by

$A(x) = \sum_{t \in \mathcal T}(D\Phi^t(x))^{-1} (D\Phi^t x)^{-T}$

The eigenfunctions saved in `v` approximate those of $\Delta^{dyn}$
```@example 1
import Plots
res = [plot_u(ctx, v[:,i],colorbar=:none,clim=(-3,3)) for i in 1:6];
Plots.plot(res...,margin=-10Plots.px)
```
Looking at the spectrum, there seems to be a gap after the third eigenvalue:
```@example 1
Plots.scatter(range(1,6), real.(λ))
```
We can use the [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) package to compute coherent structures from the first two non-trivial eigenvalues:
```@example 1
using Clustering
numclusters=2
res = kmeans(v[:,2:numclusters+1]',numclusters+1)
u = kmeansresult2LCS(res)
Plots.plot([plot_u(ctx,u[:,i],200,200,color=:viridis) for i in [1,2,3]]...)

```
## The `gridContext` type

The FEM-based methods of `CoherentStructures.jl` rely heavily on the [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) package.
This package is very low-level and does not provide point-location/plotting functions.
To be able to more conveniently work with the specific types of grids that we need, all necessary variables for a single grid can be stored in an object of type `gridContext`, which also provides an interface for point-location and plotting.

In this documentation, the variable name `ctx` is exclusively used for objects of this type.

Details regarding the internals of this type can ben found in the API section, it is in general easier not to worry about these but to simply treat the type as an abstraction representing a grid.

### Node ordering and dof ordering

Finite Element methods work with degrees of freedom (dof), which are elements of some dual space.
For nodal finite elements, these correspond to evaluation functionals at the nodes of the grid.

The nodes of the grid can be obtained in the following way `[n.x for n in ctx.grid.nodes]`.
However, most of the methods of this package do _not_ return results in this order, but instead
use `JuAFEM.jl`'s dof-ordering.

TODO: describe how to convert between the orderings


## Boundary Conditions


TODO: Finish this

##Plotting
