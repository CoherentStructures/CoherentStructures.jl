#(c) 2017 Nathanael Schilling
#This file implements methods for working with JuAFEM grids
#This includes methods for making grids from Delaunay Triangulations based
#on the code in FEMDL.jl
#There are also functions for evaluating functions on the grid

GP = GeometricalPredicates
VD = VoronoiDelaunay

#JuAFEM has no functions for determining which cell a point is in.

const default_quadrature_order=5

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

    #The following two fields are only well-defined for regular rectangular grids
    spatialBounds::Vector{AbstractVector} #In 2D, this is {LL,UR} for regular grids
    numberOfPointsInEachDirection::Vector{Int}

    gridType::String #See also makeRegularGrid() function

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

#TODO: is it better for gridType to be of type ::Symbol?
regularGridTypes = ["regular triangular grid", "regular P2 triangular grid", "regular Delaunay grid", "regular P2 Delaunay grid", "regular quadrilateral grid", "regular P2 quadrilateral grid"]
function regularGrid(gridType::String, numnodes::Tuple{Int,Int}, LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    if gridType == "regular triangular grid"
        return regularTriangularGrid(numnodes, LL,UR,quadrature_order)
    elseif gridType == "regular Delaunay grid"
        return regularDelaunayGrid(numnodes, LL,UR,quadrature_order)
    elseif gridType == "regular P2 triangular grid"
        return regularTriangularGrid(numnodes, LL,UR,quadrature_order)
    elseif gridType == "regular P2 Delaunay grid"
        return regularP2DelaunayGrid(numnodes, LL,UR,quadrature_order)
    elseif gridType == "regular quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL,UR,quadrature_order)
    elseif gridType == "regular P2 quadrilateral grid"
        return regularP2QuadrilateralGrid(numnodes, LL,UR,quadrature_order)
    else
        fail("Unsupported grid type '$gridType'")
    end
end

#Based on JuAFEM's WriteVTK.vtk_point_data
function nodeToDHTable(ctx::abstractGridContext{dim}) where {dim}
    dh::DofHandler = ctx.dh
    const n = ctx.n
    res = fill(0,n)
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        ctr = 1
        offset = JuAFEM.field_offset(dh, dh.field_names[1])
        for node in getnodes(cell)
               res[node] = _celldofs[ctr + offset]
               ctr += 1
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
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.gridType="irregular Delaunay grid" #This can be ovewritten by other constructors
        return result
end


#Creates a regular grid on a square with delaunay triangulation
function regularDelaunayGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    node_list = Vec{2,Float64}[]
    for x0 in linspace(LL[1],UR[1],numnodes[1])
        for x1 in linspace(LL[2],UR[2],numnodes[2])
            push!(node_list,Vec{2}([x0,x1]))
        end
    end
    result = gridContext{2}(Triangle,node_list, quadrature_order)
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
    result.gridType = "regular Delaunay grid"
    return result
end

(::Type{gridContext{2}})(::Type{QuadraticTriangle},node_list::Vector{Vec{2,Float64}},quadrature_order::Int=default_quadrature_order) = begin
        grid,loc = generate_grid(QuadraticTriangle,node_list)
        ip = Lagrange{2, RefTetrahedron, 2}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.gridType = "irregular P2 Delaunay grid"
        return result
end

function regularP2DelaunayGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    node_list = Vec{2,Float64}[]
    for x0 in linspace(LL[1],UR[1],numnodes[1])
        for x1 in linspace(LL[2],UR[2],numnodes[2])
            push!(node_list,Vec{2}([x0,x1]))
        end
    end
    result = gridContext{2}(QuadraticTriangle,node_list, quadrature_order)
    #TODO: Think about what values would be sensible for the two variables below
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
    result.gridType = "regular P2 Delaunay grid"
    return result
end


#Constructor for regular 2D triangular grids (without delaunay)
(::Type{gridContext{2}})(::Type{Triangle},
                         numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),
                         quadrature_order::Int=default_quadrature_order) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
        grid = generate_grid(Triangle,(numnodes[1]-1,numnodes[2]-1),Vec{2}(LL),Vec{2}(UR) )
        loc = regularGridLocator{Triangle}(numnodes[1],numnodes[2],Vec{2}(LL),Vec{2}(UR))
        ip = Lagrange{2, RefTetrahedron, 1}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.spatialBounds = [LL,UR]
        result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
        result.gridType = "regular triangular grid"
        return result
end

#Creates a regular grid on a rectangle with Triangles but without delaunay Triangulation
function regularTriangularGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(Triangle,numnodes, LL,UR)
end


#Constructor for regular P2-Lagrange 2D triangular grids (without delaunay)
(::Type{gridContext{2}})(::Type{QuadraticTriangle},
                         numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),
                         quadrature_order::Int=default_quadrature_order) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
        grid = generate_grid(QuadraticTriangle,(numnodes[1]-1,numnodes[2]-1),Vec{2}(LL), Vec{2}(UR ))
        loc = regularGridLocator{QuadraticTriangle}(numnodes[1],numnodes[2],Vec{2}(LL),Vec{2}(UR))
        ip = Lagrange{2, RefTetrahedron, 2}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.spatialBounds = [LL,UR]
        result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
        result.gridType = "regular P2 triangular grid"
        return result
end

function regularP2TriangularGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(QuadraticTriangle,numnodes, LL,UR)
end

#Constructor for regular P2-Lagrange 2D quadrilateral grids
(::Type{gridContext{2}})(::Type{QuadraticQuadrilateral},
                         numnodes::Tuple{Int,Int}=(25,25),LL::Vec{2}=Vec{2}([0.0,0.0]),UR::Vec{2}=Vec{2}([1.0,1.0]),
                         quadrature_order::Int=default_quadrature_order) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
        grid = generate_grid(QuadraticQuadrilateral,(numnodes[1]-1,numnodes[2]-1),Vec{2}(LL), Vec{2}(UR) )
        loc = regularGridLocator{QuadraticQuadrilateral}(numnodes[1],numnodes[2],Vec{2}(LL),Vec{2}(UR))
        ip = Lagrange{2, RefCube, 2}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefCube}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.spatialBounds = [LL,UR]
        result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
        result.gridType = "regular P2 quadrilateral grid"
        return result
end

function regularP2QuadrilateralGrid(numnodes::Tuple{Int,Int}=(25,25),LL::Vec{2}=Vec{2}([0.0,0.0]),UR::Vec{2}=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(QuadraticQuadrilateral,numnodes, LL,UR)
end


#Constructor for regular 2D quadrilateral grids
(::Type{gridContext{2}})(::Type{Quadrilateral},
                         numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),
                         quadrature_order::Int=default_quadrature_order) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
        grid = generate_grid(Quadrilateral,(numnodes[1]-1,numnodes[2]-1),Vec{2}(LL),Vec{2}(UR))
        loc = regularGridLocator{Quadrilateral}(numnodes[1],numnodes[2],Vec{2}(LL),Vec{2}(UR))
        ip = Lagrange{2, RefCube, 1}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefCube}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.spatialBounds = [LL,UR]
        result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
        result.gridType = "regular quadrilateral grid"
        return result
end

#Creates a regular grid on a rectangle with Quadrilateral Elements
function regularQuadrilateralGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Vec{2}([0.0,0.0]),UR::AbstractVector=Vec{2}([1.0,1.0]),quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(Quadrilateral,numnodes, LL,UR)
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
function evaluate_function(ctx::gridContext,x::Vec{2},u::Vector{Float64},outside_value=0.0)
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
    a = [NumberedPoint2D(VD.min_coord+(x[i][1] - min_x)*scale_x,VD.min_coord+(x[i][2]-min_y)*scale_y,i) for i in 1:n]
    for i in 1:n
        assert(!(GP.getx(a[i]) < VD.min_coord || GP.gety(a[i]) > VD.max_coord))
    end
    tess = VD.DelaunayTessellation2D{NumberedPoint2D}(n)
    push!(tess,a)
    m = 0
    for i in tess
        m += 1
    end
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
    point_inbounds = NumberedPoint2D(VD.min_coord+(x[1]-loc.min_x)*loc.scale_x,VD.min_coord+(x[2]-loc.min_y)*loc.scale_y)
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
    #TODO: rewrite this so that we actually find the cell in question and get the ids
    #From there (instead of from the tesselation). Then get rid of the permutation that
    #is implicit below (See also comment below in p2DelaunayCellLocator locatePoint())
    return (inv(J) ⋅ (x - grid.nodes[t._a.id].x)), [t._b.id, t._c.id, t._a.id]
end


#For delaunay triangulations with P2-Lagrange Elements
struct p2DelaunayCellLocator <: cellLocator
    m::Int64
    scale_x::Float64
    scale_y::Float64
    min_x::Float64
    min_y::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
    internal_triangles::Vector{Int}
    inv_internal_triangles::Vector{Int}
    function p2DelaunayCellLocator(m,scale_x,scale_y,min_x,min_y,tess)
        itr = start(tess)
        internal_triangles = []
        inv_internal_triangles = zeros(length(tess._trigs))
        while !done(tess,itr)
            push!(internal_triangles, itr.ix)
            next(tess,itr)
        end
        for (i,j) in enumerate(internal_triangles)
            inv_internal_triangles[j] = i
        end
        res = new(m,scale_x,scale_y,min_x,min_y,tess,internal_triangles,inv_internal_triangles)
        return res
    end
end

function locatePoint(loc::p2DelaunayCellLocator, grid::JuAFEM.Grid, x::Vec{2})
    point_inbounds = NumberedPoint2D(VD.min_coord+(x[1]-loc.min_x)*loc.scale_x,VD.min_coord+(x[2]-loc.min_y)*loc.scale_y)
    if min(point_inbounds.x, point_inbounds.y) < VD.min_coord || max(point_inbounds.x,point_inbounds.y) > VD.max_coord
        throw(DomainError())
    end
    t = VD.findindex(loc.tess, point_inbounds)
    if VD.isexternal(loc.tess._trigs[t])
        throw(DomainError())
    end
    const qTriangle = grid.cells[loc.inv_internal_triangles[t]]
    v1::Vec{2} = grid.nodes[qTriangle.nodes[2]].x - grid.nodes[qTriangle.nodes[1]].x
    v2::Vec{2} = grid.nodes[qTriangle.nodes[3]].x - grid.nodes[qTriangle.nodes[1]].x
    J::Tensor{2,2} = v1 ⊗ e1  + v2 ⊗ e2
    #TODO: Think about whether doing it like this (with the permutation) is sensible
    return (inv(J) ⋅ (x - grid.nodes[qTriangle.nodes[1]].x)), permute!(collect(qTriangle.nodes),[2,3,1,5,6,4])
end

#Here N gives the number of nodes and M gives the number of faces
struct regularGridLocator{T} <: cellLocator where {M,N,T <: JuAFEM.Cell{2,M,N}}
    n_x::Int
    n_y::Int
    LL::Vec{2}
    UR::Vec{2}
end
function locatePoint(loc::regularGridLocator{Triangle},grid::JuAFEM.Grid, x::Vec{2})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.n_x-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.n_y-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.n_x-1) #If we hit the right hand edge
        n1 = loc.n_x-2
        loc1 = 1.0
    end
    if n2 == (loc.n_y-1) #If we hit the top edge
        n2 = loc.n_y-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in:
    ll = n1 + n2*loc.n_x
    lr = ll + 1
    ul = n1 + (n2+1)*loc.n_x
    ur = ul + 1
    assert(ur < (loc.n_x * loc.n_y))
    if loc1 + loc2 <= 1.0 # ◺
        return Vec{2}([loc1,loc2]), [lr+1, ul+1,ll+1]
    else # ◹
        #The transformation that maps ◹ (with bottom node at origin) to ◺ (with ll node at origin)
        #Does [0,1] ↦ [1,0] and [-1,1] ↦ [0,1]
        #So it has representation matrix (columnwise) [ [1,-1] | [1,0] ]
        const tM = Tensor{2,2}([1,-1,1,0])
        return tM⋅Vec{2}([loc1-1,loc2]), [ ur+1, ul+1,lr+1]
    end
    return
end

function locatePoint(loc::regularGridLocator{Quadrilateral},grid::JuAFEM.Grid, x::Vec{2})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.n_x-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.n_y-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.n_x-1) #If we hit the right hand edge
        n1 = loc.n_x-2
        loc1 = 1.0
    end
    if n2 == (loc.n_y-1) #If we hit the top edge
        n2 = loc.n_y-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in:
    ll = n1 + n2*loc.n_x
    lr = ll + 1
    ul = n1 + (n2+1)*loc.n_x
    ur = ul + 1
    assert(ur < (loc.n_x * loc.n_y))
    return Vec{2}([2*loc1-1,2*loc2-1]), [ll+1,lr+1,ur+1,ul+1]
end

#Same principle as for Triangle type above
function locatePoint(loc::regularGridLocator{QuadraticTriangle},grid::JuAFEM.Grid, x::Vec{2})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.n_x-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.n_y-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.n_x-1) #If we hit the right hand edge
        n1 = loc.n_x-2
        loc1 = 1.0
    end
    if n2 == (loc.n_y-1) #If we hit the top edge
        n2 = loc.n_y-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in
    #Zero-indexing of the array here, so we need to +1 for everything being returned
    num_x_with_edge_nodes::Int = loc.n_x  + loc.n_x - 1
    num_y_with_edge_nodes::Int = loc.n_y + loc.n_y -1
    ll = 2*n1 + 2*n2*num_x_with_edge_nodes
    lr = ll + 2
    ul = 2*n1 + 2(n2+1)*num_x_with_edge_nodes
    ur = ul + 2
    middle_left =  2*n1 + (2*n2+1)*num_x_with_edge_nodes
    assert(ur < (num_x_with_edge_nodes*num_y_with_edge_nodes)) #Sanity check
    if loc1 + loc2 <= 1.0 # ◺
        return Vec{2}([loc1,loc2]), [lr+1,ul+1,ll+1, middle_left+2, middle_left+1, ll+2]
    else # ◹

        #The transformation that maps ◹ (with bottom node at origin) to ◺ (with ll node at origin)
        #Does [0,1] ↦ [1,0] and [-1,1] ↦ [0,1]
        #So it has representation matrix (columnwise) [ [1,-1] | [1,0] ]
        const tM = Tensor{2,2}([1,-1,1,0])
        return tM⋅Vec{2}([loc1-1,loc2]), [ ur+1, ul+1,lr+1,ul+2,middle_left+2, middle_left+3]
    end
    return
end


function locatePoint(loc::regularGridLocator{QuadraticQuadrilateral},grid::JuAFEM.Grid, x::Vec{2})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.n_x-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.n_y-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.n_x-1) #If we hit the right hand edge
        n1 = loc.n_x-2
        loc1 = 1.0
    end
    if n2 == (loc.n_y-1) #If we hit the top edge
        n2 = loc.n_y-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in
    #Zero-indexing of the array here, so we need to +1 for everything being returned
    num_x_with_edge_nodes::Int = loc.n_x  + loc.n_x - 1
    num_y_with_edge_nodes::Int = loc.n_y + loc.n_y -1
    ll = 2*n1 + 2*n2*num_x_with_edge_nodes
    lr = ll + 2
    ul = 2*n1 + 2(n2+1)*num_x_with_edge_nodes
    ur = ul + 2
    middle_left =  2*n1 + (2*n2+1)*num_x_with_edge_nodes
    assert(ur < (num_x_with_edge_nodes*num_y_with_edge_nodes)) #Sanity check
    #permute!(collect(qTriangle.nodes),[2,3,1,5,6,4])
    return Vec{2}([2*loc1-1,2*loc2-1]), [ll+1,lr+1,ur+1,ul+1,ll+2,middle_left+3, ul+2, middle_left+1,middle_left+2]
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

function JuAFEM.generate_grid(::Type{QuadraticTriangle}, nodes_in::Vector{Vec{2,Float64}})
    tess,m,scale_x,scale_y,min_x,min_y = delaunay2(nodes_in)
    locator = p2DelaunayCellLocator(m,scale_x,scale_y,min_x,min_y,tess)
    nodes = Node{2,Float64}[]
    #TODO: replace below with map
    for node_coords in  nodes_in
        push!(nodes,Node(node_coords))
    end
    n = length(nodes)
    ctr = n #As we add nodes (for edge vertices), increment the ctr...

    centerNodes = spzeros(n,n)
    cells = QuadraticTriangle[]
    for tri_id in 1:m
        tri = tess._trigs[locator.internal_triangles[tri_id]]

        #Create non-vertex nodes
        ab = centerNodes[tri._a.id, tri._b.id]
        if ab == 0
            ctr+=1
            ab = centerNodes[tri._a.id,tri._b.id] = centerNodes[tri._b.id,tri._a.id] =  ctr
            center = Node(0.5*(nodes[tri._b.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end
        ac = centerNodes[tri._a.id, tri._c.id]
        if ac == 0
            ctr+=1
            ac = centerNodes[tri._a.id,tri._c.id] = centerNodes[tri._c.id,tri._a.id] = ctr
            center = Node(0.5*(nodes[tri._c.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end

        bc = centerNodes[tri._c.id, tri._b.id]
        if bc == 0
            ctr+=1
            bc = centerNodes[tri._b.id,tri._c.id] = centerNodes[tri._c.id,tri._b.id] = ctr
            center = Node(0.5*(nodes[tri._c.id].x + nodes[tri._b.id].x))
            push!(nodes,center)
        end

        J = (nodes_in[tri._b.id] - nodes_in[tri._a.id]) ⊗ e1
        J +=  (nodes_in[tri._c.id] - nodes_in[tri._a.id]) ⊗ e2
        detJ = det(J)

        @assert det(J) != 0
        if detJ > 0
            new_tri  = QuadraticTriangle((tri._a.id,tri._b.id,tri._c.id,ab,bc,ac))
        else
            new_tri  = QuadraticTriangle((tri._a.id,tri._c.id,tri._b.id,ac,bc,ab))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    return grid, locator

end
