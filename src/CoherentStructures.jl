module CoherentStructures

    using StaticArrays

    using Tensors
    # using Tensors: otimes, dot, ⊗, ⋅

    using DiffEqBase
    using OrdinaryDiffEq
    using Contour
    using Distances
    using NearestNeighbors
    using Interpolations
    using LinearMaps
    using IterativeSolvers
    using LinearAlgebra
    using SparseArrays
    using Distributed
    using SharedArrays
    using Markdown
    using Statistics
    using Arpack
    using GeometricalPredicates
    using VoronoiDelaunay
    using ForwardDiff

    using JuAFEM
    using RecipesBase
    using SymEngine

    #Contains a list of functions being exported
    include("exports.jl")

    ##Diffusion operator related functions
    abstract type SparsificationMethod end

    include("diffusion_operators.jl")

    ##Distances related functions and types
    include("dynamicmetrics.jl")

    ##Some small utility functions that are used throughout
    abstract type abstractGridContext{dim} end
    include("util.jl")

    ##Functions related to pulling back tensors
    include("pullbacktensors.jl")

    ##Functions related to geodesic elliptic LCS detection
    include("ellipticLCS.jl")

    #Vector field from stream function generation
    include("streammacros.jl")

    ##Definitions of velocity fields
    include("velocityfields.jl")

    ##Extensions to JuAFEM dealing with non-curved grids
    ##Support for evaluating functions at grid points, delaunay Triangulations

    #The cellLocator provides an abstract basis class for classes for locating points on grids.
    #A cellLocator should implement a locatePoint function (see below)
    #TODO: Find out the existence of such a function can be enforced by julia

    abstract type cellLocator end
    include("gridfunctions.jl")

    #Creation of Stiffness and Mass-matrices
    include("FEMassembly.jl")

    #TO-approach based methods
    include("TO.jl")


    #Plotting
    include("plotting.jl")

    #Solving Advection/Diffusion Equation
    include("advection_diffusion.jl")
end
