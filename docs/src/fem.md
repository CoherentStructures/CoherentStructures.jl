# FEM-based Methods

These methods rely on the theory outlined by Froyland's [*Dynamical Laplacian*]
(http://arxiv.org/pdf/1411.7186v4.pdf)
and the [*Geometric Heat Flow*](https://www.researchgate.net/publication/306291640_A_geometric_heat-flow_theory_of_Lagrangian_coherent_structures) of Karrash & Keller.

This involves the discretization of an averaged heat_flow operator of the form:

$\Delta^{dyn} := \sum_{t \in \mathcal T} P^*_t \Delta P_t$

for a finite series of times $\mathcal T$, where $P_t$ is the transfer-operator for the flow at time $t$

The resulting operator is both symmetric and uniformly elliptic, and can be discretize using FEM-based methods. Eigenfunctions of $\Delta^{dyn}$ can be used to find Lagrangian Coherent Structures.

### The `gridContext` type
