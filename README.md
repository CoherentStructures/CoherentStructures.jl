# CoherentStructures.jl

**CoherentStructures.jl** is a toolbox for computing Lagrangian Coherent Structures
(LCSs) by objective, i.e., observer-independent, methods in aperiodic flows in
Julia.

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![stable docs][docs-stable-img]][docs-stable-url] [![dev docs][docs-dev-img]][docs-dev-url] | [![build status][build-img]][build-url] [![code coverage][codecov-img]][codecov-url] |

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode.

Install this package by typing

    pkg> add https://github.com/CoherentStructures/CoherentStructures.jl.git

In order to run the example cases, please install our companion package
[`StreamMacros.jl`](https://github.com/CoherentStructures/StreamMacros.jl) by typing

    pkg> add https://github.com/CoherentStructures/StreamMacros.jl.git

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
