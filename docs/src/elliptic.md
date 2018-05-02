# Geodesic elliptic material vortices

```@meta
CurrentModule = CoherentStructures
```
The following functions implement an LCS methodology developed by George Haller
and collaborators, which can be found in the following papers:
   * Haller & Beron-Vera, 2012
   * Haller & Beron-Vera, 2013
   * Karrasch, Huhn, and Haller, 2015
The present code is structurally inspired by Alireza Hadjighasem's MATLAB
implementation [github project](https://github.com/Hadjighasem/Elliptic_LCS_2D),
which was written in the context of the [SIAM Review paper](https://doi.org/10.1137/140983665). Depending on the indefinite metric
tensor field used, the following functions directly yield the following types of
coherent structures:
   * black-hole/Lagrangian coherent vortices [Haller & Beron-Vera, 2012](https://doi.org/10.1017/jfm.2013.391)
   * elliptic objective Eulerian coherent structures (OECSs) [Serra & Haller, 2016](http://dx.doi.org/10.1063/1.4951720)
   * material diffusive transport barriers ((Haller, Karrasch, and Kogelbauer, 2018)

## Example

## Function documentation

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

```@docs
ellipticLCS
```
