module CoherentStructures

# use standard libraries
using LinearAlgebra
using LinearAlgebra: checksquare
import ProgressMeter
using SparseArrays
using SparseArrays: nzvalview, dropzeros!
using Distributed
using SharedArrays: SharedArray
using Statistics: mean

# import data type packages
using StaticArrays: SVector, SMatrix
using Tensors: Vec, Tensor, SymmetricTensor, basevec, dott, tdot, otimes, unsafe_symmetric
using AxisArrays: AxisArray, ClosedInterval, axisvalues

# using DiffEqBase: DiffEqBase, initialize!, isconstant, update_coefficients!, @..
using SciMLBase
using OrdinaryDiffEq: ODEProblem, ODEFunction, ContinuousCallback, terminate!, solve, Tsit5
# using OrdinaryDiffEq: OrdinaryDiffEqNewtonAlgorithm, DEFAULT_LINSOLVE,
#     alg_order, OrdinaryDiffEqMutableCache, alg_cache, @muladd, perform_step!, @unpack,
#     unwrap_alg, is_mass_matrix_alg, _unwrap_val

using Distances: Distances, PreMetric, SemiMetric, Metric, Euclidean, PeriodicEuclidean,
    pairwise, pairwise!, colwise, colwise!, result_type

using NearestNeighbors: BallTree, KDTree, inrange, knn, MinkowskiMetric

using Interpolations: Interpolations, LinearInterpolation, CubicSplineInterpolation,
    interpolate, scale, BSpline, Linear, Cubic, Natural, OnGrid, Free

# import linear algebra related packages
using LinearMaps: LinearMap, FunctionMap
using IterativeSolvers: cg
using ArnoldiMethod: partialschur, partialeigen

# import geometry related packages
using GeometricalPredicates: GeometricalPredicates, Point, AbstractPoint2D, Point2D, getx, gety
using VoronoiDelaunay: DelaunayTessellation2D, findindex, isexternal, max_coord, min_coord

# Ferrite
import Ferrite
const FEM = Ferrite

#Other miscallaneous packages
using RecipesBase
import ColorTypes
using ForwardDiff
using Contour

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

# definitions of velocity fields
include("velocityfields.jl")

#SEBA algorithm
include("seba.jl")

#Functions related to isoperimetry
include("isoperimetry.jl")

##Extensions to Ferrite dealing with non-curved grids
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
# include("odesolvers.jl")

# transfer operator-based methods
include("TO.jl")

# Ulam's method
include("ulam.jl")

# plotting functionality
include("plotting.jl")

# linear response related methods
include("linearResponse.jl")

# functions for dynamic isoperimetry
include("dynamicIsoperimetry.jl")

end
