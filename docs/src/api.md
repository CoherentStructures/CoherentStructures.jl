# API
## Dynamical Systems Utilities

```@meta
CurrentModule = CoherentStructures
```
```@docs
flow
```

```@docs
linearized_flow
```

TODO: add documentation for various pullback diffusion tensor functions, etc...

## FEM-API

### Stiffness and Mass Matrices
```@docs
assembleStiffnessMatrix
assembleMassMatrix
```

### Constructing Grids

There are several helper functions available for constructing grids. The simplest is:
```@docs
regularGrid
```
Supported values for the `gridType` argument are:
```@example
using CoherentStructures #hide
CoherentStructures.regularGridTypes
```
The following functions are conceptually similar:
```@docs
regularTriangularGrid
regularDelaunayGrid
regularP2TriangularGrid
regularP2DelaunayGrid
regularQuadrilateralGrid
regularP2QuadrilateralGrid
```
All of these methods return a `gridContext` object.
```@docs
CoherentStructures.gridContext
```
#### Irregular grids
The constructors for `CoherentStructures.gridContext`, including one for irregular delaunay grids, are not exported by default, the documentation is available through the REPL:

``` #TODO: add @docs here once it works
help?> (::Type{CoherentStructures.gridContext{2}})
```

### Boundary Conditions API
```@docs
boundaryData
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
evaluate_function_from_nodevals
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
```
default_quadrature_order=5
```
