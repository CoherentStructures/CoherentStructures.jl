# CoherentStructures.jl

**CoherentStructures.jl** is a toolbox for computing Lagrangian Coherent Structures
(LCSs) by objective, i.e., observer-independent, methods in aperiodic flows in
Julia.

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][build-img]][build-url] [![][codecov-img]][codecov-url] |


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode.
First, install the [`JuAFEM.jl`](https://github.com/KristofferC/JuAFEM.jl) package by typing in the Julia REPL (console):

    pkg> add https://github.com/KristofferC/JuAFEM.jl.git

In order to run the example cases, please install our companion package
[`StreamMacros.jl`](https://github.com/CoherentStructures/StreamMacros.jl.git) by typing

    pkg> add https://github.com/CoherentStructures/StreamMacros.jl

Finally, install this package by typing

    pkg> add https://github.com/CoherentStructures/CoherentStructures.jl.git

For an introduction to LCS methods, usage examples, and implementation details,
see the documentation. There, you will also find links to executable source
files for some examples.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://coherentstructures.github.io/CoherentStructures.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://coherentstructures.github.io/CoherentStructures.jl/stable

[build-img]: https://github.com/CoherentStructures/CoherentStructures.jl/workflows/CI/badge.svg
[build-url]: https://github.com/CoherentStructures/CoherentStructures.jl/actions?query=workflow%3ACI

[codecov-img]: http://codecov.io/github/CoherentStructures/CoherentStructures.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/CoherentStructures/CoherentStructures.jl?branch=master
