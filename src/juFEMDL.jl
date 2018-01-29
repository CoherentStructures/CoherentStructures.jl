module juFEMDL

    using Tensors
    using DiffEqBase, OrdinaryDiffEq#, ForwardDiff
    using Interpolations


    import GeometricalPredicates
    import VoronoiDelaunay
    import JLD

    using JuAFEM

    using GR

    #Contains a list of functions being exported
    include("exports.jl")


    abstract type abstractGridContext{dim} end

    ##Some small utility functions that are used throughout
    include("util.jl")

    ##Functions related to pulling back tensors
    include("PullbackTensors.jl")

    ##Definitions of velocity fields
    include("velocityFields.jl")

    ##Extensions to JuAFEM dealing with non-curved grids
    ##Support for evaluating functions at grid points, delaunay Triangulations

    #The cellLocator provides an abstract basis class for classes for locating points on grids.
    #A cellLocator should implement a locatePoint function (see below)
    #TODO: Find out the existence of such a function can be enforced by julia

    abstract type cellLocator end

    include("GridFunctions.jl")

    #Creation of Stiffness and Mass-matrices
    include("FEMassembly.jl")

    #TO-approach based methods
    include("TO.jl")


   #Some test cases, similar to velocityFields.jl
   include("numericalExperiments.jl")

   #Plotting
   include("plotting.jl")
end