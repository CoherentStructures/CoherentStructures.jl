# Graph Laplacian/diffusion maps-based LCS methods

```@meta
CurrentModule = CoherentStructures
```

The LCS approaches implemented and described in this section are strongly influenced
by ideas developed in the spectral clustering/diffusion maps communities. The
general references include:
   * [Shi & Malik, Normalized cuts and image segmentation, 2000](https://dx.doi.org/10.1109/34.868688)
   * [Coifman & Lafon, Diffusion maps, 2006](https://dx.doi.org/10.1016/j.acha.2006.04.006)
   * [Marshall & Hirn, Time coupled diffusion maps, 2018](https://dx.doi.org/10.1016/j.acha.2017.11.003)
In the LCS context, these ideas have been adopted in the following works:
   * somewhat related [Froyland & Padberg-Gehle, 2015](https://dx.doi.org/10.1063/1.4926372)
   * [Hadjighasem et al., 2016](http://dx.doi.org/10.1103/PhysRevE.93.063107)
   * [Banisch & Koltai, 2017](https://dx.doi.org/10.1063/1.4971788)
   * [Rypina et al., 2017](https://dx.doi.org/10.5194/npg-24-189-2017)/[Padberg-Gehle & Schneide, 2018](https://dx.doi.org/10.5194/npg-24-661-2017)
   <!-- * De Diego et al., 2018 -->

For demonstrations on example cases, please consult the page on
[Working with trajectories](@ref).

## Function documentation

### Sparsification methods

Three commonly used sparsification methods are implemented for use with various
graph Laplacian methods.
```@docs
KNN
MutualKNN
Neighborhood
```

Other sparsification methods can be implemented by defining a
`SparsificationMethod` type and a corresponding [`spdist`](@ref) method.

### Diffusion-maps-type/adjancency-matrix-based graph Laplacian methods

Since the Euclidean heat kernel is ubiquitous in diffusion maps-based computations,
we provide it for convenience.
```@docs
gaussian
gaussiancutoff
```
To compute a sparse distance matrix (or adjacency matrix, depending on the
[sparsification method](@Sparsification methods)), use [`spdist`](@ref).
```@docs
spdist
```
The main high-level functions for the above-listed methods are the following.
```@docs
sparse_diff_op_family
sparse_diff_op
```

### Normalization functions

In the diffusion maps framework, there are two commonly used normalization steps:
1. kernel-density estimate normalization ([`kde_normalize!`](@ref)), and
2. row normalization ([`row_normalize!`](@ref)), to obtain a diffusion/Markov
   operator (w.r.t. right- and left-action, respectively).
```@docs
kde_normalize!
row_normalize!
```

### Diffusion-coordinate-like functions

```@docs
diffusion_coordinates
diffusion_distance
```
