# Geodesic elliptic material vortices

```@meta
CurrentModule = CoherentStructures
```

## Background

The following functions implement an LCS methodology developed in the following papers:

* [Haller & Beron-Vera, 2012](https://doi.org/10.1016/j.physd.2012.06.012)
* [Haller & Beron-Vera, 2013](https://doi.org/10.1017/jfm.2013.391)
* [Karrasch, Huhn, and Haller, 2015](https://doi.org/10.1098/rspa.2014.0639)

Our implementation here follows conceptually [Karrasch, Huhn, and Haller, 2015](https://doi.org/10.1098/rspa.2014.0639),
and is described in detail in the preprint TBD.
Depending on the indefinite metric tensor field used, the functions below yield
the following types of coherent structures:

* black-hole/Lagrangian coherent vortices ([Haller & Beron-Vera, 2012](https://doi.org/10.1017/jfm.2013.391))
* elliptic objective Eulerian coherent structures (OECSs) ([Serra & Haller, 2016](https://doi.org/10.1063/1.4951720))
* material diffusive transport barriers ([Haller, Karrasch, and Kogelbauer, 2018](https://doi.org/10.1073/pnas.1720177115))

The general procedure is the following. Assume $T$ is the symmetric tensor field
of interest, say, (i) the Cauchy-Green strain tensor field $C$, (ii) the
rate-of-strain tensor field $S$, or (iii) the averaged diffusion-weighted
Cauchy-Green tensor field $\bar{C}_D$; cf. the references above. Denote by
$0<\lambda_1\leq\lambda_2$ the eigenvalue and by $\xi_1$ and $\xi_2$ the
corresponding eigenvector fields of $T$. Then the direction fields of interest
are given by

$\eta_{\lambda}^{\pm} := \sqrt{\frac{\lambda_2 - \lambda}{\lambda_2-\lambda_1}}\xi_1 \pm \sqrt{\frac{\lambda - \lambda_1}{\lambda_2-\lambda_1}}\xi_2.$

Tensor singularities are defined as points at which $\lambda_2=\lambda_1$, i.e.,
at which the two characteristic directions $\xi_1$ and $\xi_2$ are not
well-defined. As described and exploited in
[Karrasch et al., 2015](https://doi.org/10.1098/rspa.2014.0639),
non-negligible tensor singularities express themselves by an angle gap when
tracking (the angle of) tensor eigenvector fields along closed paths surrounding
the singularity. Our approach here avoids computing singularities directly, but
rather computes the index for each grid cell and then combines nearby
singularities, i.e., adds non-vanishing indices of nearby grid cells.

In summary, the implementation consists of the following steps:

   1. compute the index for each grid cell and combine nearby singular grid cells
      to "singularity candidates";
   2. look for elliptic singularity candidates (and potentially isolated wedge
      pairs);
   3. place an eastwards oriented Poincaré section at the pair center;
   4. for each point on the discretized Poincaré section, scan through the given
      parameter interval such that the corresponding η-orbit closes at that point;
   5. if desired: for each Poincaré section, take the outermost closed orbit as
      the coherent vortex barrier (if there exist any).

## Function documentation

### The meta-functions `ellipticLCS` and `constrainedLCS`

The fully automated high-level functions are:

```@docs
ellipticLCS
constrainedLCS
materialbarriers
```

One of their arguments is a list of parameters used in the LCS detection. This
list is combined in a data type called `LCSParameters`. The output is a list of `EllipticBarrier`s and a list of `Singularity`s.
There is an option to retrieve all closed barriers (`outermost=false`), in
contrast to extracting only the outermost vortex boundaries (`outermost=true`), which is more efficient.

The meta-functions consist of two steps: first, the index
theory-based determination of where to search for closed orbits,, cf.
[Index theory-based placement of Poincaré sections](@ref); second, the
closed orbit computation, cf. [Closed orbit detection](@ref).

### Specific types

These are the specifically introduced types for elliptic LCS computations.

```@docs
LCSParameters
EllipticBarrier
EllipticVortex
```

Another one is `Singularity`, which comes along with some convenience functions.

```@docs
Singularity
getcoords
getindices
```

### Index theory-based placement of Poincaré sections

This is performed by [`singularity_detection`](@ref) for line fields
(such as eigenvector fields of symmetric positive-definite tensor fields) and by
[`critical_point_detection`](@ref) for classic vector fields.

```@docs
singularity_detection
critical_point_detection
```

This function takes three steps. The first two are:

```@docs
compute_singularities
Combine
```

The third step is a postprocessing step, in which detected singularities are
merged according to different heuristics.

```@docs
Combine20
Combine31
Combine20Aggressive
```

The function [`compute_singularities`](@ref) requires one of two signed distance
functions for angles. These are [`s1dist`](@ref) for vector fields, and
[`p1dist`](@ref) for line fields.

```@docs
s1dist
p1dist
```

From all virtual/merged singularities those with a suitable index are selected.
Around each elliptic singularity the tensor field is localized and passed on for
closed orbit detection.

### Closed orbit detection

```@docs
compute_returning_orbit
compute_closed_orbits
```

### Convenience functions

First of all, [`Singularity`](@ref), [`EllipticBarrier`](@ref) and [`EllipticVortex`](@ref)
objects can be passed as initial conditions to the [`flow`](@ref) function, which returns
a (time-resolved) vector of corresponding objects. Moreover, we have advection with
adaptive point insertion (known in oceanography as "Dritschel advection") and an function
to compute the area of [`EllipticBarrier`](@ref) and [`EllipticVortex`](@ref) objects.

```@docs
flowgrow
area
```
