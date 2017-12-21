#(c) 2017 Nathanael Schilling
#This file implements methods for working with JuAFEM grids
#This includes methods for making grids from Delaunay Triangulations based
#on the code in FEMDL.jl
#There are also functions for evaluating functions on the grid

using Tensors, JuAFEM

import GeometricalPredicates
GP = GeometricalPredicates
import VoronoiDelaunay
VD = VoronoiDelaunay

include("util.jl")

#JuAFEM has no functions for determining which cell a point is in.

#The cellLocator provides an abstract basis class for classes for locating points on grids.
#A cellLocator should implement a locatePoint function (see below)
#TODO: Find out the existence of such a function can be enforced by julia

abstract type cellLocator end

const default_quadrature_order=5

#The following type is used in files on which this file depends (as a forward declaration there).
#Redeclaring here for readability
abstract type abstractGridContext{dim} end

#This type is used for storing everything needed as context to work on a grid
#when doing FEM on scalar fields. Currently implemented only for 2D
#mutable so that it is passed by reference and not by value. #TODO: is this sensible?
mutable struct gridContext{dim} <: abstractGridContext{dim}
    grid::JuAFEM.Grid
    ip::JuAFEM.Interpolation
    dh::JuAFEM.DofHandler
    qr::JuAFEM.QuadratureRule
    loc::cellLocator
    node_to_dof::Vector{Int} #dhtable[nodeid] contains the index of the corresponding dof
    dof_to_node::Vector{Int} #dhtable[nodeid] contains the index of the corresponding dof
    n::Int #The number of nodes
    m::Int #The number of cells
    function gridContext{dim}(grid::JuAFEM.Grid,ip::JuAFEM.Interpolation,dh::JuAFEM.DofHandler,qr::JuAFEM.QuadratureRule,loc::cellLocator) where {dim}
        x =new{dim}(grid,ip,dh,qr,loc)
        x.n = JuAFEM.getnnodes(dh.grid)
        x.m = JuAFEM.getncells(dh.grid)

        #TODO: Measure if the sorting below is expensive
        x.node_to_dof = nodeToDHTable(x)
        x.dof_to_node = sortperm(x.node_to_dof)
        return x
    end
end

#Based on JuAFEM's WriteVTK.vtk_point_data
function nodeToDHTable(ctx::abstractGridContext{dim}) where {dim}
    dh::DofHandler = ctx.dh
    const n = ctx.n
    res = fill(0,n)
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        counter = 1
        offset = JuAFEM.field_offset(dh, dh.field_names[1])
        for node in getnodes(cell)
               res[node] = _celldofs[counter + offset]
           counter += 1
        end
    end
    return res
end

#Constructor for grids created with delaunay triangulations.
#It has to be defined like this as otherwise julia complains that 2 is not a type
(::Type{gridContext{2}})(::Type{Triangle},node_list::Vector{Vec{2,Float64}},quadrature_order::Int=default_quadrature_order) = begin
        grid,loc = generate_grid(Triangle,node_list)
        ip = Lagrange{2, RefTetrahedron, 1}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        return gridContext{2}(grid,ip,dh,qr,loc)
end

#Creates a regular uniform grid on a square
function regularDelaunayGrid(numnodes::Tuple{Int,Int}=(25,25),LL::Vec{2}=Vec{2}([0.0,0.0]),UR::Vec{2}=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    m = 50 # number of cell in one direction
    node_list = Vec{2,Float64}[]
    for x1 in linspace(LL[1],UR[1],numnodes[1])
        for x0 in linspace(LL[2],UR[2],numnodes[2])
            push!(node_list,Vec{2}([x0,x1]))
        end
    end
    return gridContext{2}(Triangle,node_list, quadrature_order)
end


#The locatePoint function returns a tuple (coords, [nodes])
#where coords gives the coordinates within the reference shape (e.g. standard simplex)
#And [nodes] is the list of corresponding node ids, ordered in the order of the
#corresponding shape functions from JuAFEM's interpolation.jl file
#Here we call the specialized locatePoint function for this kind of grid
function locatePoint(ctx::gridContext{dim}, x::Vec{dim}) where dim
    return locatePoint(ctx.loc,ctx.grid,x)
end

#TODO: Make this also work for P2-Lagrange
function evaluate_function(grid::gridContext,x::Vec{2},u::Vector{Float64},outside_value=0.0)
    local_coordinates,nodes = try
         locatePoint(ctx,x)
    catch y
        if isa(y,DomainError)
            return outside_value
        end
        throw(y)
    end
    result = 0.0
    for (j,nodeid) in enumerate(nodes)
        result +=  u[nodeid]*JuAFEM.value(ctx.ip, j, local_coordinates)
    end
    return result
end

#Helper type for keeping track of point numbers
struct NumberedPoint2D <: VD.AbstractPoint2D
    x::Float64
    y::Float64
    id::Int64
    NumberedPoint2D(x::Float64,y::Float64,k::Int64) = new(x,y,k)
    NumberedPoint2D(x::Float64,y::Float64) = new(x, y, 0)
    NumberedPoint2D(p::VD.Point2D) = new(p.x, p.y, 0)
    NumberedPoint2D(p::Vec{2,Float64}) = new(p[1], p[2], 0)
 end
 GP.Point(x::Real, y::Real, k::Int64) = NumberedPoint2D(x, y, k)
 GP.Point2D(p::NumberedPoint2D) = Point2D(p.x,p.y)
 GP.gety(p::NumberedPoint2D) = p.y
 GP.getx(p::NumberedPoint2D) = p.x

#More or less equivalent to matlab's delaunay function, based on code from FEMDL.jl
function delaunay2(x::Vector{Vec{2,Float64}})
    width = VD.max_coord - VD.min_coord
    max_x = maximum(map(v->v[1],x))
    min_x = minimum(map(v->v[1],x))
    max_y = maximum(map(v->v[2],x))
    min_y = minimum(map(v->v[2],x))
    scale_x = 0.9*width/(max_x - min_x)
    scale_y = 0.9*width/(max_y - min_y)
    n = length(x)
    a = [NumberedPoint2D(VD.min_coord+x[i][1]*scale_x - min_x,VD.min_coord+x[i][2]*scale_y-min_y,i) for i in 1:n]
    #TODO: Replace below with an assert, or with nothing (as it should never happen)
    for i in 1:n
        if GP.getx(a[i]) < VD.min_coord || GP.gety(a[i]) > VD.max_coord
            ax = GP.getx(a[i])
            ay = GP.gety(a[i])
            print("a = [$ax,$ay]\n")
        end
    end
    tess = VD.DelaunayTessellation2D{NumberedPoint2D}(n)
    push!(tess,a)
    m = 0
    for tri in tess; m += 1; end  # count number of triangles --
    return tess,m,scale_x,scale_y,min_x,min_y
end

#For delaunay triangulations, we can use the tesselation
#object and the locate() function
struct delaunayCellLocator <: cellLocator
    m::Int64
    scale_x::Float64
    scale_y::Float64
    min_x::Float64
    min_y::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
end

function locatePoint(loc::delaunayCellLocator, grid::JuAFEM.Grid, x::Vec{2})
    point_inbounds = NumberedPoint2D(VD.min_coord+x[1]*loc.scale_x-loc.min_x,VD.min_coord+x[2]*loc.scale_y-loc.min_y,1)
    if min(point_inbounds.x, point_inbounds.y) < VD.min_coord || max(point_inbounds.x,point_inbounds.y) > VD.max_coord
        throw(DomainError())
    end
    t = VD.locate(loc.tess, point_inbounds)
    if VD.isexternal(t)
        throw(DomainError())
    end
    v1::Vec{2} = grid.nodes[t._b.id].x - grid.nodes[t._a.id].x
    v2::Vec{2} = grid.nodes[t._c.id].x - grid.nodes[t._a.id].x
    J::Tensor{2,2} = v1 ⊗ e1  + v2 ⊗ e2
    return (inv(J) ⋅ (x - grid.nodes[t._a.id].x)), [t._b.id, t._c.id, t._a.id]
end

#Here N gives the number of nodes and M gives the number of faces
struct regularGridLocator{T} <: cellLocator where {M,N,T <: JuAFEM.Cell{2,M,N}}
    n_x::Int
    n_y::Int
    left::Vec{2}
    right::Vec{2}
end

function locatePoint(loc::regularGridLocator{Triangle},grid::JuAFEM.Grid, x::Vec{2})
    #TODO: Implement this
    return
end

function locatePoint(loc::regularGridLocator{Quadrilateral},grid::JuAFEM.Grid, x::Vec{2})
    #TODO: Implement this
    return
end


function JuAFEM.generate_grid(::Type{Triangle}, nodes_in::Vector{Vec{2,Float64}})
    tess,m,scale_x,scale_y,min_x,min_y = delaunay2(nodes_in)
    nodes = Node{2,Float64}[]
    for node_coords in  nodes_in
        push!(nodes,Node(node_coords))
    end
    cells = Triangle[]
    for tri in tess
        J = (nodes_in[tri._b.id] - nodes_in[tri._a.id]) ⊗ e1
        J +=  (nodes_in[tri._c.id] - nodes_in[tri._a.id]) ⊗ e2
        detJ = det(J)
        @assert det(J) != 0
        if detJ > 0
            new_tri  = Triangle((tri._a.id,tri._b.id,tri._c.id))
        else
            new_tri  = Triangle((tri._a.id,tri._c.id,tri._b.id))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    locator = delaunayCellLocator(m,scale_x,scale_y,min_x,min_y,tess)
    return grid, locator

end
