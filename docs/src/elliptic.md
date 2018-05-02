# Geodesic elliptic material vortices

```@meta
CurrentModule = CoherentStructures
```
The following functions implement an LCS methodology developed in the following papers:
   * [Haller & Beron-Vera, 2012](https://dx.doi.org/10.1016/j.physd.2012.06.012)
   * [Haller & Beron-Vera, 2013](https://dx.doi.org/10.1017/jfm.2013.391)
   * [Karrasch, Huhn, and Haller, 2015](https://dx.doi.org/10.1098/rspa.2014.0639)
The present code is structurally inspired--albeit partially significantly
improved--by Alireza Hadjighasem's MATLAB implementation [github project](https://github.com/Hadjighasem/Elliptic_LCS_2D),
which was written in the context of the [SIAM Review paper](https://doi.org/10.1137/140983665). Depending on the indefinite metric
tensor field used, the following functions directly yield the following types of
coherent structures:
   * black-hole/Lagrangian coherent vortices ([Haller & Beron-Vera, 2012](https://doi.org/10.1017/jfm.2013.391))
   * elliptic objective Eulerian coherent structures (OECSs) ([Serra & Haller, 2016](https://dx.doi.org/10.1063/1.4951720))
   * material diffusive transport barriers (Haller, Karrasch, and Kogelbauer, 2018)
The general procedure is the following:
   * locate tensor singularities, where the tensor field is the Cauchy-Green strain tensor field, the rate-of-strain tensor field, and the averaged diffusion-weighted Cauchy-Green tensor field, resp.;
   * determine their singularity type (only non-degenerate types like wedges and trisectors are detected);
   * look for wedge pairs that are reasonably isolated;
   * place an eastwards oriented Poincaré section at the pair center;
   * for each point on the discretized Poincaré section, scan through the parameter space such that the corresponding η-orbit closes at that point;
   * take the outermost closed orbit as the coherent vortex barrier.

## Example

## Function documentation

The fully automated meta-function is the following:

```@docs
ellipticLCS
```

Essentially, it calls sequentially the following functions.

```@docs
singularity_location_detection
```

```@docs
singularity_type_detection
```

```@docs
detect_elliptic_region
```

```@docs
set_Poincaré_section
```

```@docs
compute_outermost_closed_orbit
```
