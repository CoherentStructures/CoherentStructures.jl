# Graph Laplacian/diffusion maps-based LCS methods

```@meta
CurrentModule = CoherentStructures
```

Cite a couple of important papers:
   * [Shi & Malik, Normalized cuts and image segmentation, 2000](https://dx.doi.org/10.1109/34.868688)
   * [Coifman & Lafon, Diffusion maps, 2006](https://dx.doi.org/10.1016/j.acha.2006.04.006)
   * [Marshall & Hirn, Time coupled diffusion maps, 2018](https://dx.doi.org/10.1016/j.acha.2017.11.003)
In the LCS context, we have
   * somewhat related [Froyland & Padberg-Gehle, 2015](https://dx.doi.org/10.1063/1.4926372)
   * [Hadjighasem et al., 2016](http://dx.doi.org/10.1103/PhysRevE.93.063107)
   * [Banisch & Koltai, 2017](https://dx.doi.org/10.1063/1.4971788)
   * [Rypina et al., 2017](https://dx.doi.org/10.5194/npg-24-189-2017)/[Padberg-Gehle & Schneide, 2018](https://dx.doi.org/10.5194/npg-24-661-2017)
   * De Diego et al., 2018

## Function documentation

### Sparsification methods

Two commonly used sparsification methods are implemented for use with various
graph Laplacian methods, see below.

```@docs
KNN
mutualKNN
neighborhood
```

Other sparsification methods can be implemented by defining a corresponding [`sparseaffinitykernel`](@ref) instance.

### Diffusion-maps type graph Laplacian methods

```@docs
diff_op
sparse_diff_op_family
sparse_diff_op
sparseaffinitykernel
α_normalize!
wLap_normalize!
```

### Adjancency-matrix-based graph Laplacian methods

```@docs
sparse_adjacency
sparse_adjacency_list
```

### Diffusion-coordinate-like functions

```@docs
diffusion_coordinates
diffusion_distance
```
