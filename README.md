To install, run `Pkg.clone("git@gitlab.lrz.de:ga24guz/CoherentStructures.git")`
If you do not have a public key registered with gitlab, run:
`Pkg.clone("https://gitlab.lrz.de/ga24guz/CoherentStructures.git")`.

See examples/rot\_double\_gyre.jl for a minimal working example of CG and TO-based methods.

See examples/OceanFlow.jl for a minimal working example with an ocean-flow testcase

See examples/symbolic_differentation.jl and examples/bickley_with_macro.jl for
examples where the vector fields are generated from time-dependent Hamiltonians.

See also examples/Laplace\_\{Bickley,DG\}\_interp.jl files for more examples

*TODO*:
   * Write a function that takes as input a tensor-function and a point structure (vector, matrix, 3d array),
and returns a---if necessary---time-dependent tensor field. (static case done)
   * Speed up the spatiotemporal metric evaluation!!!
