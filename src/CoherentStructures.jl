module CoherentStructures

# use standard libraries
using LinearAlgebra
using ProgressMeter
using SparseArrays
using Distributed
using SharedArrays: SharedArray
using Statistics: mean

# import data type packages
import StaticArrays
using StaticArrays: SVector, @SVector, SArray, SMatrix, @SMatrix
import Tensors
using Tensors: Vec, Tensor, SymmetricTensor
import AxisArrays
using AxisArrays: AxisArray, ClosedInterval, axisvalues

import DiffEqBase
import OrdinaryDiffEq
const ODE = OrdinaryDiffEq
import Distances
const Dists = Distances
import NearestNeighbors
const NN = NearestNeighbors
import Interpolations
const ITP = Interpolations

# import linear algebra related packages
import LinearMaps
const LMs = LinearMaps
import IterativeSolvers
import Arpack

# import geometry related packages
import GeometricalPredicates
const GP = GeometricalPredicates
import VoronoiDelaunay
const VD = VoronoiDelaunay

import JuAFEM
const JFM = JuAFEM
using RecipesBase
import SymEngine
import ModelingToolkit
using ModelingToolkit: Variable, Differential, simplified_expr,
      expand_derivatives, Expression, Operation, Constant

# contains a list of exported functions
include("exports.jl")

# diffusion operator-related functions
"""
    abstract type SparsificationMethod

Abstract type for sparsification methods.
Concrete subtypes are [`KNN`](@ref), [`MutualKNN`](@ref),
and [`Neighborhood`](@ref).
"""
abstract type SparsificationMethod end
include("diffusion_operators.jl")

# distance-related functions and types
include("dynamicmetrics.jl")

# some utility functions that are used throughout
abstract type AbstractGridContext{dim} end
include("util.jl")

# functions related to pulling back tensors under flow maps
include("pullbacktensors.jl")

# functions related to geodesic elliptic LCS detection
include("ellipticLCS.jl")

# generation of vector fields from stream functions
include("streammacros.jl")

# definitions of velocity fields
include("velocityfields.jl")

#SEBA algorithm
include("seba.jl")

#Functions related to isoperimetry
include("isoperimetry.jl")

##Extensions to JuAFEM dealing with non-curved grids
##Support for evaluating functions at grid points, delaunay Triangulations

#The PointLocator provides an abstract basis class for classes for locating points on grids.
#A PointLocator should implement a locatePoint function (see below)
#TODO: Find out the existence of such a function can be enforced by julia

abstract type PointLocator end
include("gridfunctions.jl")
include("pointlocation.jl")
include("boundaryconditions.jl")

# creation of Stiffness and Mass-matrices
include("FEMassembly.jl")

# solving advection-diffusion equation
include("advection_diffusion.jl")

# odesolvers
include("odesolvers.jl")

# transfer operator-based methods
include("TO.jl")

# Ulam's method
include("ulam.jl")

# plotting functionality
include("plotting.jl")

end
