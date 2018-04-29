# FEM-based Methods

These methods rely on the theory outlined by Froyland's [*Dynamical Laplacian*]
(http://arxiv.org/pdf/1411.7186v4.pdf)
and the [*Geometric Heat Flow*](https://www.researchgate.net/publication/306291640_A_geometric_heat-flow_theory_of_Lagrangian_coherent_structures) of Karrash & Keller.

This involves the discretization of an averaged heat_flow operator of the form:

$\Delta^{dyn} := \sum_{t \in \mathcal T} P^*_t \Delta P_t$

for a finite series of times $\mathcal T$, where $P_t$ is the transfer-operator for the flow at time $t$

The resulting operator is both symmetric and uniformly elliptic, and can be discretize using FEM-based methods. Eigenfunctions of $\Delta^{dyn}$ can be used to find Lagrangian Coherent Structures.

### The `gridContext` type

The FEM-based methods of `CoherentStructures.jl` rely heavily on the [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) package.
This package is very low-level and does not provide point-location/plotting functions.
To be able to more conveniently work with the specific types of grids that we need, all necessary variables for a single grid can be stored in an object of type `gridContext`, which also provides an interface for point-location and plotting.

In this documentation, the variable name `ctx` is exclusively used for objects of this type.

Details regarding the internals of this type can ben found in the API section, it is in general easier not to worry about these but to simply treat the type as an abstraction representing a grid.

#### Node ordering and dof ordering

Finite Element methods work with degrees of freedom (dof), which are elements of some dual space.
For nodal finite elements, these correspond to evaluation functionals at the nodes of the grid.

The nodes of the grid can be obtained in the following way `[n.x for n in ctx.grid.nodes]`.
However, most of the methods of this package do _not_ return results in this order, but instead
use `JuAFEM.jl`'s dof-ordering.

TODO: describe how to convert between the orderings


#### Boundary Conditions


TODO: Finish this
