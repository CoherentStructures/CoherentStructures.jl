module CoherentStructures

import LinearAlgebra
import SparseArrays
using Distributed
import SharedArrays
import Markdown
using Statistics: mean

import StaticArrays
import Tensors
using Tensors: Vec, Tensor, SymmetricTensor

import DiffEqBase
import OrdinaryDiffEq
import Contour
import Distances
import NearestNeighbors
import Interpolations
import LinearMaps
import IterativeSolvers
import Arpack
import GeometricalPredicates
import VoronoiDelaunay
import ForwardDiff

import JuAFEM
import RecipesBase
import SymEngine

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

#The pointLocator provides an abstract basis class for classes for locating points on grids.
#A pointLocator should implement a locatePoint function (see below)
#TODO: Find out the existence of such a function can be enforced by julia

abstract type pointLocator end
include("gridfunctions.jl")
include("pointlocation.jl")
include("boundaryconditions.jl")

#Creation of Stiffness and Mass-matrices
include("FEMassembly.jl")

#TO-approach based methods
include("TO.jl")

#Ulam's method
include("ulam.jl")


#Plotting
include("plotting.jl")

#Solving Advection/Diffusion Equation
include("advection_diffusion.jl")
end
