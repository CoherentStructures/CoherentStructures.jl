
using Tensors
using JuAFEM
import JuAFEM.generate_grid
import VoronoiDelaunay
VD = VoronoiDelaunay
import GeometricalPredicates
GP = GeometricalPredicates
import GR


#Helper struct for keeping track of point numbers
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

#More or less matlab's delaunay function, based on code from FEMDL.jl
function delaunay(x::Vector{Vec{2,Float64}})
    width = VD.max_coord - VD.min_coord
    max_x = maximum(map(v->v[1],x))
    min_x = minimum(map(v->v[1],x))
    max_y = maximum(map(v->v[2],x))
    min_y = minimum(map(v->v[2],x))
    scale_x = 0.9*width/(max_x - min_x)
    scale_y = 0.9*width/(max_y - min_y)
    n = length(x)
    a = [NumberedPoint2D(VD.min_coord+x[i][1]*scale_x,VD.min_coord+x[i][2]*scale_y,i) for i in 1:n]
    tess = VD.DelaunayTessellation2D{NumberedPoint2D}(n)
    push!(tess,a)
    m = 0
    for tri in tess; m += 1; end  # count number of triangles -- TODO
    return tess,m,scale_x,scale_y
end

abstract type cellLocator end

struct delaunayCellLocator <: cellLocator
    m::Int64
    scale_x::Float64
    scale_y::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
end

e1 = basevec(Vec{2},1)
e2 = basevec(Vec{2},2)

function locatePoint(loc::delaunayCellLocator, grid::JuAFEM.Grid, x::Vec{2})
    point_inbounds = NumberedPoint2D(VD.min_coord+x[1]*loc.scale_x,VD.min_coord+x[2]*loc.scale_y,1)
    t = VD.locate(loc.tess, point_inbounds)
    if VD.isexternal(t)
        throw(DomainError())
    end
    v1 = grid.nodes[t._b.id].x - grid.nodes[t._a.id].x
    v2 = grid.nodes[t._c.id].x - grid.nodes[t._a.id].x
    J = v1 ⊗ e1  + v2 ⊗ e2
    return (inv(J) ⋅ (x - grid.nodes[t._a.id].x)), [t._b.id, t._c.id, t._a.id]

end

#TODO: Make this also work for P2-Lagrange
function evaluate_function(grid::JuAFEM.Grid,loc::cellLocator,x::Vec{2},u::Vector{Float64},ip::JuAFEM.Interpolation{2,RefTetrahedron,1},outside_value=0.0)
    local_coordinates,nodes = try
         locatePoint(loc,grid,x)
    catch y
        if isa(y,DomainError)
            return outside_value
        end
        throw(y)
    end
    result = 0.0
    for (j,nodeid) in enumerate(nodes)
        result +=  u[nodeid]*JuAFEM.value(ip, j, local_coordinates)
    end
    return result
end

#TODO: Make this also work for P2-Lagrange
#TODO: Make this much, much more efficient
function plot_u(grid::JuAFEM.Grid,loc::cellLocator, u::Vector{Float64},ip::JuAFEM.Interpolation{2,RefTetrahedron,1},nx=20,ny=20)
    x1 = Float64[]
    x2 = Float64[]
    values = Float64[]
    for x in linspace(0.0,1.0,nx)
        for y in linspace(0.0,1.0,ny)
            push!(x1,x)
            push!(x2,y)
            current_point = Vec{2}([x,y])
            push!(values, evaluate_function(grid,loc, Vec{2}(current_point),u, ip))
        end
    end
    GR.contourf(x1,x2,values,colormap=GR.COLORMAP_JET)
end

using Plots#TODO: Get rid of this...
function plot_tesselation(loc::delaunayCellLocator)
    x, y = VD.getplotxy(VD.delaunayedges(loc.tess))
    Plots.plot((x - VD.min_coord)/loc.scale_x,(y - VD.min_coord)/loc.scale_y)
end


function generate_grid(::Type{Triangle}, nodes_in::Vector{Vec{2,Float64}})
    tess,m,scale_x,scale_y = delaunay(nodes_in)
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
    locator = delaunayCellLocator(m,scale_x,scale_y,tess)
    return grid, locator

end
