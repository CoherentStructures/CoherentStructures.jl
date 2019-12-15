# CoherentStructures.jl
*Tools for computing Lagrangian Coherent Structures in Julia*

## Introduction

`CoherentStructures.jl` is a toolbox for computing Lagrangian Coherent Structures (LCSs) by objective, i.e., observer-independent, methods in aperiodic flows in Julia.
It has been developed in [Oliver Junge](https://www-m3.ma.tum.de/Allgemeines/OliverJunge)'s research group at TUM, Germany, by (in alphabetical order)
   * Alvaro de Diego ([@adediego](https://github.com/adediego))
   * [Oliver Junge](https://www-m3.ma.tum.de/Allgemeines/OliverJunge) ([@gaioguy](https://github.com/gaioguy))
   * [Daniel Karrasch](https://www-m3.ma.tum.de/Allgemeines/DanielKarrasch) ([@dkarrasch](https://github.com/dkarrasch))
   * Nathanael Schilling ([@natschil](https://github.com/natschil))
Contributions from colleagues in the field are most welcome via raising issues or, even better, via pull requests.

## Installation

First, install the [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) package by typing in the Julia REPL (console):

    ]add https://github.com/KristofferC/JuAFEM.jl.git

Then, run the following in the Julia REPL:

    ]add https://github.com/CoherentStructures/CoherentStructures.jl.git


## Overview of supported methods

We currently support the following methods:
* [Finite-element discretizations of the dynamic Laplacian](@ref),
* [Geodesic elliptic material vortices](@ref), such as "black-hole vortices", "material barriers" and "OECS", as well as
* [Graph Laplacian/diffusion maps-based LCS methods](@ref) for spectral clustering/diffusion maps inspired LCS approaches.

The graph Laplacian methods and the "TO" forms of the dynamic Laplacian FEM methods work directly on trajectories.

For more information on specific methods, consult the relevant pages of the documentation or the
examples pages.

## Examples
As a quick hands-on introduction, we demonstrate the usage of the
`CoherentStructures.jl` package on some classic flow problems. For references to
the original works in which the methods were developed see the respective help
page.

### List of examples

* [Rotating double gyre](@ref)
* [The standard map](@ref)
* [Bickley jet](@ref)
* [Geostrophic ocean flow](@ref)
* [Working with trajectories](@ref)
