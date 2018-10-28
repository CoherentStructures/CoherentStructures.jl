#(c) 2017 Nathanael Schilling
#This file implements methods for working with JuAFEM grids
#This includes methods for making grids from Delaunay Triangulations based
#on the code in FEMDL.jl

const GP = GeometricalPredicates
const VD = VoronoiDelaunay


#JuAFEM has no functions for determining which cell a point is in.
#Amongst other things, this file (and pointlocation.jl) implements an API for doing this.

const default_quadrature_order=5
const default_quadrature_order3D=2


"""
    struct gridContext<dim>

Stores everything needed as "context" to be able to work on a FEM grid based on the `JuAFEM` package.
Adds a point-locator API which facilitates plotting functions defined on the grid within Julia.

# Fields
- `grid::JuAFEM.Grid`, `ip::JuAFEM.Interpolation`,`ip_geom::JuAFEM.Interpolation`, `qr::JuAFEM.QuadratureRule` - See the `JuAFEM` package
- `loc::CellLocator` object used for point-location on the grid.
- `node_to_dof::Vector{Int}`  lookup table for dof index of a node (for Lagrange Elements)
- `dof_to_node::Vector{Int}`  inverse of node_to_dof
- `cell_to_dof::Vector{Int}`  lookup table for dof index of a cell (for piecewise constant elements)
- `dof_to_cell::Vector{Int}`  inverse of cell_to_dof
- `num_nodes::Int` number of nodes on the grid
- `num_cells::Int` number of elements (e.g. triangles,quadrilaterals, ...) on the grid
- `n` number of degrees of freedom (== `num_nodes` for Lagrange Elements, and == `num_cells` for piecewise constant elements)
- `quadrature_points::Vector{Vec{dim,Float64}}` All quadrature points on the grid, in a fixed order.
- `mass_weights::Vector{Float64}` Weighting for mass matrix
- `spatialBounds` If available, the corners of a bounding box of a domain. For regular grids, the bounds are tight.
- `numberOfPointsInEachDirection` For regular grids, how many (non-interior) nodes make up the regular grid.
- `gridType` A string describing what kind of grid this is (e.g. "regular triangular grid")
"""
mutable struct gridContext{dim} <: abstractGridContext{dim} #TODO: Currently set as mutable, is this sensible?
    grid::JuAFEM.Grid
    ip::JuAFEM.Interpolation
    ip_geom::JuAFEM.Interpolation
    dh::JuAFEM.DofHandler
    qr::JuAFEM.QuadratureRule
    loc::pointLocator

    ##The following two fields only make sense for Lagrange-elements##
    node_to_dof::Vector{Int} #node_to_dof[nodeid] contains the index of the corresponding dof
    dof_to_node::Vector{Int} #dof_to_node[dofid] contains the index of the corresponding node

    ##The following two fields only make sense for piecewise constant elements##
    cell_to_dof::Vector{Int} #cell_to_dof[cellid] contains the index of the corresponding dof
    dof_to_cell::Vector{Int} #dof_to_cell[dofid] contains the index of the corresponding cell

    num_nodes::Int #The number of nodes
    num_cells::Int #The number of cells

    n::Int #The number of dofs

    quadrature_points::Vector{Tensors.Vec{dim,Float64}} #All quadrature points, ordered by how they are accessed in assemble routines
    mass_weights::Vector{Float64}

    ##The following two fields are only well-defined for regular rectangular grids
    spatialBounds::Vector{AbstractVector} #This is {LL,UR} for regular grids
    #This is the number of (non-interior) nodes in each direction (not points)
    numberOfPointsInEachDirection::Vector{Int}

    gridType::String

    function gridContext{dim}(
                grid::JuAFEM.Grid,
                ip::JuAFEM.Interpolation,
                ip_geom::JuAFEM.Interpolation,
                dh::JuAFEM.DofHandler,
                qr::JuAFEM.QuadratureRule,
                loc::pointLocator
            ) where {dim}

        x =new{dim}(grid, ip,ip_geom, dh, qr, loc)
        x.num_nodes = JuAFEM.getnnodes(dh.grid)
        x.num_cells = JuAFEM.getncells(dh.grid)
        x.n = JuAFEM.ndofs(dh)

        #TODO: Measure if the sorting below is expensive
        if isa(ip,JuAFEM.Lagrange)
            x.node_to_dof = nodeToDHTable(x)
            x.dof_to_node = sortperm(x.node_to_dof)
        elseif isa(ip,JuAFEM.PiecewiseConstant)
            x.cell_to_dof = cellToDHTable(x)
            x.dof_to_cell = sortperm(x.cell_to_dof)
        else
            throw(AssertionError("Unknown interpolation type"))
        end
        x.quadrature_points = getQuadPoints(x)
        x.mass_weights = ones(length(x.quadrature_points))
        return x
    end
end


#Based on JuAFEM's WriteVTK.vtk_point_data
function nodeToDHTable(ctx::abstractGridContext{dim}) where {dim}
    dh::JuAFEM.DofHandler = ctx.dh
    n = ctx.n
    res = Vector{Int}(undef,n)
    for cell in JuAFEM.CellIterator(dh)
        _celldofs = JuAFEM.celldofs(cell)
        ctr = 1
        offset = JuAFEM.field_offset(dh, dh.field_names[1])
        for node in JuAFEM.getnodes(cell)
               res[node] = _celldofs[ctr + offset]
               ctr += 1
        end
    end
    return res
end

function cellToDHTable(ctx::abstractGridContext{dim}) where {dim}
    dh::JuAFEM.DofHandler = ctx.dh
    n = ctx.n
    res = Vector{Int}(undef,n)
    for (cellindex,cell) in enumerate(JuAFEM.CellIterator(dh))
        _celldofs = JuAFEM.celldofs(cell)
        offset = JuAFEM.field_offset(dh, dh.field_names[1])
        for node in JuAFEM.getnodes(cell)
            mynode = node
        end
        res[cellindex] = _celldofs[offset+1]
    end
    return res
end


function gridContext{1}(::Type{JuAFEM.Line},
                         numnodes::Tuple{Int}=( 25), LL::AbstractVector=[0.0], UR::AbstractVector=[1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JuAFEM.Lagrange{1,JuAFEM.RefCube,1}(),
                         )
    # The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.Line, (numnodes[1]-1,), Tensors.Vec{1}(LL), Tensors.Vec{1}(UR))
    loc = regular1DGridLocator{JuAFEM.Line}(numnodes[1], Tensors.Vec{1}(LL), Tensors.Vec{1}(UR))

    dh = JuAFEM.DofHandler(grid)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)

    qr = JuAFEM.QuadratureRule{1, JuAFEM.RefCube}(quadrature_order)
    result = gridContext{1}(grid, ip,JuAFEM.Lagrange{1,JuAFEM.RefCube,1}(), dh, qr, loc)
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1]]
    result.gridType = "regular 1d grid"

    return result
end


function gridContext{1}(::Type{JuAFEM.QuadraticLine},
                         numnodes::Tuple{Int}=( 25),
                         LL::AbstractVector=[0.0],
                         UR::AbstractVector=[1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JuAFEM.Lagrange{1,JuAFEM.RefCube,2}()
                         )
    # The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.QuadraticLine, (numnodes[1]-1,), Tensors.Vec{1}(LL), Tensors.Vec{1}(UR))
    loc = regular1DGridLocator{JuAFEM.QuadraticLine}(numnodes[1], Tensors.Vec{1}(LL), Tensors.Vec{1}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{1, JuAFEM.RefCube}(quadrature_order)
    push!(dh, :T, 1) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result = gridContext{1}(grid, ip,JuAFEM.Lagrange{1,JuAFEM.RefCube,2}(), dh, qr, loc)
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1]]
    result.gridType = "regular 1d grid"
    return result
end

function regular1dGridPC(numnodes::Int,left=0.0,right=1.0; quadrature_order::Int=default_quadrature_order)
    return gridContext{1}(JuAFEM.Line,(numnodes,),
        [left],[right];
        quadrature_order=quadrature_order,
        ip=JuAFEM.PiecewiseConstant{1,JuAFEM.RefCube,1}()
        )
end

function regular1dGrid(numnodes::Int,left=0.0,right=1.0; quadrature_order::Int=default_quadrature_order)
    return gridContext{1}(JuAFEM.Line,(numnodes,), [left],[right]; quadrature_order=quadrature_order)
end

function regular1dGrid(numnodes::Tuple{Int},args...;kwargs...)
    return regular1dGrid(numnodes[1],args...;kwargs...)
end

function regular1dP2Grid(numnodes::Int,left=0.0,right=1.0; quadrature_order::Int=default_quadrature_order)
    return gridContext{1}(JuAFEM.QuadraticLine,(numnodes,), [left],[right]; quadrature_order=quadrature_order)
end

function regular1dP2Grid(numnodes::Tuple{Int},args...;kwargs...)
    return regular1dP2Grid(numnodes[1],args...;kwargs...)
end

regular2DGridTypes = ["regular triangular grid",
                    "regular P2 triangular grid",
                    "regular Delaunay grid",
                    "regular P2 Delaunay grid",
                    "regular quadrilateral grid",
                    "regular P2 quadrilateral grid"]

"""
    regular2DGrid(gridType, numnodes, LL=[0.0,0.0],UR=[1.0,1.0];quadrature_order=default_quadrature_order)

Constructs a regular grid. `gridType` should be from `CoherentStructures.regular2DGridTypes`
"""
function regular2DGrid(
            gridType::String,
            numnodes::Tuple{Int,Int},
            LL::AbstractVector=[0.0,0.0],
            UR::AbstractVector=[1.0,1.0];
            kwargs...
        )

    if gridType == "regular triangular grid"
        return regularTriangularGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular Delaunay grid"
        return regularDelaunayGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 triangular grid"
        return regularP2TriangularGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 Delaunay grid"
        return regularP2DelaunayGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 quadrilateral grid"
        return regularP2QuadrilateralGrid(numnodes, LL, UR; kwargs...)
    else
        fail("Unsupported grid type '$gridType'")
    end
end


#= #TODO 1.0
"""
    gridContext{2}(JuAFEM.Triangle, node_list, [quadrature_order=default_quadrature_order])

Create a P1-Lagrange grid based on Delaunay Triangulation.
Uses `DelaunayVoronoi.jl` internally.
"""
=#
function gridContext{2}(
            ::Type{JuAFEM.Triangle},
            node_list::Vector{Tensors.Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order,
            on_torus=false,
            LL=nothing,
            UR=nothing,
            ip=JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,1}(),
            )
    grid, loc = JuAFEM.generate_grid(JuAFEM.Triangle, node_list;on_torus=on_torus,LL=LL,UR=UR)
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{2, JuAFEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result =  gridContext{2}(grid, ip,JuAFEM.Lagrange{2,RefTetrahedron,1}(), dh, qr, loc)
    result.gridType = "irregular Delaunay grid" #This can be ovewritten by other constructors
    if LL == nothing
        LL = [minimum([x[i] for x in node_list]) for i in 1:2]
    end
    if UR == nothing
        UR = [maximum([x[i] for x in node_list]) for i in 1:2]
    end
    result.spatialBounds = [LL,UR]
    return result
end

function aperiodicDelaunayGrid(nodes_in::Vector{Vec{2,Float64}})
    ctx = CoherentStructures.gridContext{2}(
        JuAFEM.Triangle,nodes_in,on_torus=false)
        return ctx
end

function periodicDelaunayGrid(
                    nodes_in::Vector{Vec{2,Float64}},
                    LL::AbstractVector=[0.0,0.0],
                    UR::AbstractVector=[1.0,1.0]
    )

    ctx = CoherentStructures.gridContext{2}(
        JuAFEM.Triangle,nodes_in,on_torus=true,LL=LL,UR=UR)

    metric = PEuclidean(UR-LL)

    bdata = boundaryData(ctx,metric)
    return ctx,bdata
end


"""
    regularDelaunayGrid(numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular grid on a square.
Internally uses Delauny Triangulation.
"""
function regularDelaunayGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order
        )
    X = range(LL[1],stop= UR[1],length= numnodes[1])
    Y = range(LL[2], stop=UR[2], length=numnodes[2])
    node_list = vec([Tensors.Vec{2}((x, y)) for y in Y, x in X])
    result = CoherentStructures.gridContext{2}(JuAFEM.Triangle,
         node_list;
         quadrature_order=quadrature_order
         )
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular Delaunay grid"
    return result
end


function regularDelaunayGridPC(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order
        )
    X = range(LL[1],stop= UR[1],length= numnodes[1])
    Y = range(LL[2], stop=UR[2], length=numnodes[2])
    node_list = vec([Tensors.Vec{2}((x, y)) for y in Y, x in X])
    result = CoherentStructures.gridContext{2}(
            JuAFEM.Triangle,
            node_list,
            quadrature_order=quadrature_order,
            ip=JuAFEM.PiecewiseConstant{2,RefTetrahedron,1}()
            )
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular Delaunay grid"
    return result
end

#= TODO 1.0
"""
    gridContext{2}(JuAFEM.QuadraticTriangle, node_list, quadrature_order=default_quadrature_order)

Create a P2 grid given a set of (non-interior) nodes using Delaunay Triangulation.
"""
=#
function gridContext{2}(
            ::Type{JuAFEM.QuadraticTriangle},
            node_list::Vector{Tensors.Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order,
            ip=JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()
            )
    grid, loc = JuAFEM.generate_grid(JuAFEM.QuadraticTriangle, node_list)
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{2, JuAFEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result = gridContext{2}(grid, ip,JuAFEM.Lagrange{2,RefTetrahedron,2}(), dh, qr, loc)
    result.gridType = "irregular P2 Delaunay grid"
    return result
end

"""
    regularP2DelaunayGrid(numnodes=(25,25),LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P2 triangular grid with `numnodes` being the number of (non-interior) nodes in each direction.
"""
function regularP2DelaunayGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order
        )

    X = range(LL[1], stop=UR[1], length=numnodes[1])
    Y = range(LL[2], stop=UR[2], length=numnodes[2])
    node_list = vec([Tensors.Vec{2}((x, y)) for y in Y, x in X])
    result = gridContext{2}(JuAFEM.QuadraticTriangle, node_list, quadrature_order=quadrature_order)
    #TODO: Think about what values would be sensible for the two variables below
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 Delaunay grid"
    return result
end

#=
"""
    gridContext{2}(JuAFEM.Triangle, numnodes=(25,25),LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular triangular grid. Does not use Delaunay triangulation internally.
"""
=#

function gridContext{2}(::Type{JuAFEM.Triangle},
                         numnodes::Tuple{Int,Int}=(25, 25),
                         LL::AbstractVector=[0.0, 0.0],
                         UR::AbstractVector=[1.0, 1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,1}(),
                         )
    # The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.Triangle, (numnodes[1]-1,numnodes[2]-1), Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    loc = regular2DGridLocator{JuAFEM.Triangle}(numnodes[1], numnodes[2], Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{2, JuAFEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result = gridContext{2}(grid, ip,JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,1}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular triangular grid"
    return result
end

"""
    regularTriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0], quadrature_order=default_quadrature_order)

Create a regular P1 triangular grid on a rectangle; it does not use Delaunay triangulation internally.
"""
function regularTriangularGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=[0.0,0.0], UR::AbstractVector=[1.0,1.0];
                                quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(JuAFEM.Triangle, numnodes, LL, UR,quadrature_order=quadrature_order)
end


"""
    regularTriangularGridPC(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0], quadrature_order=default_quadrature_order)

Create a regular triangular grid on a rectangle with piecewise constant dofs; it does not use Delaunay triangulation internally.
"""
function regularTriangularGridPC(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=[0.0,0.0], UR::AbstractVector=[1.0,1.0];
                                quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(
        JuAFEM.Triangle,
        numnodes,
        LL,UR;
        quadrature_order=quadrature_order,ip=JuAFEM.PiecewiseConstant{2,RefTetrahedron,1}()
        )
end


#= TODO 1.0
"""
    gridContext{2}(JUAFEM.QuadraticTriangle, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 triangular grids. Does not use Delaunay triangulation internally.
"""
=#
function gridContext{2}(::Type{JuAFEM.QuadraticTriangle},
                         numnodes::Tuple{Int,Int}=(25, 25),
                         LL::AbstractVector=[0.0,0.0],
                         UR::AbstractVector=[1.0,1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()
                         )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.QuadraticTriangle, (numnodes[1]-1,numnodes[2]-1), Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    loc = regular2DGridLocator{JuAFEM.QuadraticTriangle}(numnodes[1], numnodes[2], Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{2, JuAFEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result =  gridContext{2}(grid, ip,JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 triangular grid"
    return result
end


"""
    regularP2TriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P2 triangular grid on a Rectangle. Does not use Delaunay triangulation internally.
"""
function regularP2TriangularGrid(
            numnodes::Tuple{Int,Int}=(25, 25), LL::AbstractVector=[0.0, 0.0], UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order
        )
    return gridContext{2}(JuAFEM.QuadraticTriangle, numnodes, LL, UR, quadrature_order=quadrature_order)
end

#= TODO 1.0
"""
    gridContext{2}(JUAFEM.Quadrilateral, numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0], quadrature_order=default_quadrature_order)

Constructor for regular P1 quadrilateral grids.
"""
=#
function gridContext{2}(
            ::Type{JuAFEM.Quadrilateral},
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=[0.0,0.0],
            UR::AbstractVector=[1.0,1.0];
            quadrature_order::Int=default_quadrature_order,
            ip=JuAFEM.Lagrange{2, JuAFEM.RefCube, 1}()
            )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.Quadrilateral, (numnodes[1]-1,numnodes[2]-1), Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    loc = regular2DGridLocator{JuAFEM.Quadrilateral}(numnodes[1], numnodes[2], Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{2, JuAFEM.RefCube}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result =  gridContext{2}(grid, ip,JuAFEM.Lagrange{2,JuAFEM.RefCube,1}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular quadrilateral grid"
    return result
end

"""
    regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P1 quadrilateral grid on a Rectangle.
"""
function regularQuadrilateralGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order
        )
    return gridContext{2}(JuAFEM.Quadrilateral, numnodes, LL, UR, quadrature_order=quadrature_order)
end


#= TODO 1.0
"""
    gridContext{2}(JUAFEM.QuadraticQuadrilateral, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 quadrilateral grids.
"""
=#
function gridContext{2}(
            ::Type{JuAFEM.QuadraticQuadrilateral},
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order,
            ip=JuAFEM.Lagrange{2, JuAFEM.RefCube, 2}()
            )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.QuadraticQuadrilateral, (numnodes[1]-1,numnodes[2]-1), Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    loc = regular2DGridLocator{JuAFEM.QuadraticQuadrilateral}(numnodes[1], numnodes[2], Tensors.Vec{2}(LL), Tensors.Vec{2}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{2, JuAFEM.RefCube}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result =  gridContext{2}(grid, ip,JuAFEM.Lagrange{2,JuAFEM.RefCube,2}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 quadrilateral grid"
    return result
end


"""
    regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0], quadrature_order=default_quadrature_order)

Create a regular P2 quadrilateral grid on a rectangle.
"""
function regularP2QuadrilateralGrid(
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order
        )
    return gridContext{2}(JuAFEM.QuadraticQuadrilateral, numnodes, LL, UR, quadrature_order=quadrature_order)
end
#=TODO 1.0
"""
    gridContext{3}(JuAFEM.Tetrahedron, numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular Tetrahedral Grid. Does not use Delaunay triangulation internally.
"""
=#
function gridContext{3}(::Type{JuAFEM.Tetrahedron},
                         numnodes::Tuple{Int,Int,Int}=(10,10,10),
                         LL::AbstractVector=[0.0,0.0,0.0],
                         UR::AbstractVector=[1.0,1.0,1.0];
                         quadrature_order::Int=default_quadrature_order3D,
                         ip=JuAFEM.Lagrange{3, JuAFEM.RefTetrahedron, 1}()
                         )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.Tetrahedron, (numnodes[1]-1, numnodes[2]-1, numnodes[3] -1), Tensors.Vec{3}(LL), Tensors.Vec{3}(UR))
    loc = regular3DGridLocator{JuAFEM.Tetrahedron}(numnodes[1], numnodes[2], numnodes[3], Tensors.Vec{3}(LL), Tensors.Vec{3}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{3, JuAFEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result =  gridContext{3}(grid, ip,JuAFEM.Lagrange{3,JuAFEM.RefCube,1}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2], numnodes[3]]
    result.gridType = "3D regular triangular grid"
    return result
end


#=TODO 1.0
"""
    gridContext{3}(JuAFEM.QuadraticTetrahedron, numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P2 Tetrahedral Grid. Does not use Delaunay triangulation internally.
"""
=#
function gridContext{3}(::Type{JuAFEM.QuadraticTetrahedron},
                         numnodes::Tuple{Int,Int,Int}=(10,10,10),
                         LL::AbstractVector=[0.0,0.0,0.0],
                         UR::AbstractVector=[1.0,1.0,1.0];
                         quadrature_order::Int=default_quadrature_order3D,
                         ip=JuAFEM.Lagrange{3, JuAFEM.RefTetrahedron, 2}()
                         )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JuAFEM.generate_grid(JuAFEM.QuadraticTetrahedron, (numnodes[1]-1, numnodes[2]-1, numnodes[3] -1), Tensors.Vec{3}(LL), Tensors.Vec{3}(UR))
    loc = regular3DGridLocator{JuAFEM.QuadraticTetrahedron}(numnodes[1], numnodes[2], numnodes[3], Tensors.Vec{3}(LL), Tensors.Vec{3}(UR))
    dh = JuAFEM.DofHandler(grid)
    qr = JuAFEM.QuadratureRule{3, JuAFEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JuAFEM.close!(dh)
    result =  gridContext{3}(grid, ip,JuAFEM.Lagrange{3,RefCube,2}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2], numnodes[3]]
    result.gridType = "3D regular triangular grid"
    return result
end

"""
    regularTetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P1 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularTetrahedralGrid(numnodes::Tuple{Int,Int,Int}=(10,10,10), LL::AbstractVector=[0.0,0.0,0.0], UR::AbstractVector=[1.0,1.0,1.0];
                                    quadrature_order::Int=default_quadrature_order3D)
    return gridContext{3}(JuAFEM.Tetrahedron, numnodes, LL, UR, quadrature_order=quadrature_order)
end

"""
    regularP2TetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P2 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularP2TetrahedralGrid(numnodes::Tuple{Int,Int,Int}=(10,10,10), LL::AbstractVector=[0.0,0.0,0.0], UR::AbstractVector=[1.0,1.0,1.0];
                                    quadrature_order::Int=default_quadrature_order3D)
    return gridContext{3}(JuAFEM.QuadraticTetrahedron, numnodes, LL, UR, quadrature_order=quadrature_order)
end


function mollify_xin(
    ctx::gridContext{dim}, x_in::AbstractVector{T},project_in) where {dim,T}

    if !project_in
        if dim == 1
            x = Tensors.Vec{dim,T}((x_in[1],))
        elseif dim == 2
            x = Tensors.Vec{dim,T}((x_in[1], x_in[2]))
        elseif dim == 3
            x = Tensors.Vec{dim,T}((x_in[1], x_in[2], x_in[3]))
        else
            error("dim = $dim not supported")
        end
    else
        if dim == 1
            x = Tensors.Vec{dim,T}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1], x_in[1])),
                ))
        elseif dim == 2
            #TODO: replace this with a macro maybe
            x = Tensors.Vec{dim,T}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1], x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2], x_in[2]))
                ))
        elseif dim == 3
            #TODO: replace this with a macro maybe
            x = Tensors.Vec{dim,T}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1], x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2], x_in[2]))
                ,max(ctx.spatialBounds[1][3], min(ctx.spatialBounds[2][3], x_in[3]))
                ))
        else
            error("dim = $dim not supported")
        end
    end
    return x
end

"""
    evaluate_function_from_node_or_cellvals(ctx,vals,x_in; [outside_value=0, project_in=false])

Like `evaluate_function_from_dofvals`, but the coefficients from `vals` are assumed to be in node order.
This is more efficient than `evaluate_function_from_dofvals`
"""
function evaluate_function_from_node_or_cellvals(
    ctx::gridContext{dim}, vals::AbstractVector{S},
    x_in::Vec{dim,W}; outside_value=0.0, project_in=false,is_diag=false
    )::W where {
        dim,S,W,
        }

    x::Vec{dim,W} = mollify_xin(ctx,x_in,project_in)

    @assert length(vals) == ctx.n

    local_coordinates::Tensors.Vec{dim,Float64}, nodes::Vector{Int}, cellid::Int = try
         locatePoint(ctx, x)
    catch y
        if isa(y,DomainError)
            return spzeros(T,size(vals)[2]) .+ outside_value
        end
        print("Unexpected error for $x")
        throw(y)
    end

    result::S = zero(S)

    if isa(ctx.ip, JuAFEM.Lagrange)
        for (j, nodeid) in enumerate(nodes)
            val::Float64 = JuAFEM.value(ctx.ip, j, local_coordinates)
            result += vals[nodeid]*val
        end
    elseif isa(ctx.ip,JuAFEM.PiecewiseConstant)
        #TODO: is hardcoding the 1 good here?
        val = JuAFEM.value(ctx.ip, 1, local_coordinates)
        result += vals[cellid]*val
    else
        throw(AssertionError("Unknown interpolation"))
    end
    return result
end

#TODO: this may not work, fix it.
function evaluate_function_from_node_or_cellvals_multiple(
    ctx::gridContext{dim}, vals::AbstractMatrix{S},
    x_in::AbstractVector{Tensors.Vec{dim,W}}; outside_value=0.0, project_in=false,is_diag=false
    )::SparseMatrixCSC{S,Int64} where{dim,S,W}

    x::Vector{Vec{dim,W}} = [mollify_xin(ctx,x_cur,project_in) for x_cur in x_in]

    @assert size(vals)[1] == ctx.n
    npoints = length(x_in)
    result::SparseMatrixCSC{S,Int64} = spzeros(S,size(vals)[2],npoints)
    for current_point in 1:npoints
        try
            local_coordinates::Tensors.Vec{dim,Float64}, nodes::Vector{Int},cellid::Int = locatePoint(ctx,x[current_point])

            if isa(ctx.ip, JuAFEM.Lagrange)
                for (j, nodeid) in enumerate(nodes)
                    val::Float64 = JuAFEM.value(ctx.ip, j, local_coordinates)
                    if !is_diag
                        for i in 1:size(vals)[2]
                            result[i,current_point] += vals[nodeid,i]*val
                        end
                    else
                        result[nodeid,current_point] += vals[nodeid,nodeid]*val
                    end
                end
            elseif isa(ctx.ip,JuAFEM.PiecewiseConstant)
                val = JuAFEM.value(ctx.ip, 1, local_coordinates)
                if !is_diag
                    for i in 1:size(vals)[2]
                        result[i,current_point] += vals[cellid,i]*val
                    end
                else
                    result[cellid,current_point] += vals[cellid,cellid]*val
                end
            else
                throw(AssertionError("Unknown interpolation"))
            end
        catch y
                if isa(y,DomainError)
                    result[:, current_point] .= outside_value
                else
                    print("Unexpected error for $(x[current_point])")
                    rethrow(y)
                end
        end
    end
    return result
end


"""
    evaluate_function_from_dofvals(ctx,dofvals,x_in; [outside_value=0,project_in=false])

Evaluate a function in the approximation space at the point (or Vector of points) `x_in`. If `x_in` is out of bounds, return `outside_value`.
If `project_in` is `true`, points not within `ctx.spatialBounds` are first projected into the domain.

The coefficients in `dofvals` are interpreted to be in dof order. If a matrix of coefficients is passed as an argument,
then the function is evaluated for each column.
"""
function evaluate_function_from_dofvals_multiple(
    ctx::gridContext{dim}, dofvals::AbstractMatrix{S},
    x_in::AbstractVector{Tensors.Vec{dim,W}};
    outside_value=0.0, project_in=false
    ) where { dim,S,W, }
    u_vals = zeros(S,size(dofvals))
    if isa(ctx.ip, JuAFEM.Lagrange)
        for i in 1:ctx.n
            u_vals[i,:] = dofvals[ctx.node_to_dof[i],:]
        end
    elseif isa(ctx.ip, JuAFEM.PiecewiseConstant)
        for i in 1:ctx.n
            u_vals[i,:] = dofvals[ctx.cell_to_dof[i],:]
        end
    end
    return evaluate_function_from_node_or_cellvals_multiple(ctx,u_vals, x_in,outside_value=outside_value,project_in=project_in)
end



"""
    sample_to(u::Vector{T},ctx_old,ctx_new)

Perform nodal_interpolation of a function onto a different grid.
"""
function sample_to(u::Vector{T}, ctx_old::CoherentStructures.gridContext, ctx_new::CoherentStructures.gridContext;
    bdata=boundaryData(),project_in=true
    ) where {T}
    u_undoBCS = undoBCS(ctx_old,u,bdata)
    u_new::Vector{T} = zeros(T,ctx_new.n)*NaN
    for i in 1:ctx_new.n
        u_new[ctx_new.node_to_dof[i]] = evaluate_function_from_dofvals(ctx_old,u_undoBCS,ctx_new.grid.nodes[i].x;
            outside_value=NaN,project_in=project_in)
    end
    return u_new
end

"""
    sample_to(u::AbstractArray{2,T},ctx_old,ctx_new)

Perform nodal_interpolation of a function onto a different grid for a set of columns of a matrix.
Returns a matrix
"""

function sample_to(u::AbstractArray{T,2},
        ctx_old::CoherentStructures.gridContext,
        ctx_new::CoherentStructures.gridContext;
        bdata=boundaryData(),
        project_in=true
        ) where {T}
    ncols = size(u)[2]
    u_undoBCS = zeros(T, ctx_old.n,ncols)
    for j in 1:ncols
        u_undoBCS[:,j] = undoBCS(ctx_old,u[:,j],bdata)
    end
    u_new::Array{T,2} = zeros(T,ctx_new.n,ncols)*NaN
    for i in 1:ctx_new.n
        for j in 1:ncols
            u_new[ctx_new.node_to_dof[i],j] = evaluate_function_from_dofvals(
            ctx_old,u_undoBCS[:,j],ctx_new.grid.nodes[i].x, outside_value=NaN,project_in=true)
        end
    end
    return u_new
end


#Helper type for keeping track of point numbers
struct NumberedPoint2D <: VD.AbstractPoint2D
    x::Float64
    y::Float64
    id::Int64
    NumberedPoint2D(x::Float64,y::Float64,k::Int64) = new(x,y,k)
    NumberedPoint2D(x::Float64,y::Float64) = new(x, y, 0)
    NumberedPoint2D(p::VD.Point2D) = new(p.x, p.y, 0)
    NumberedPoint2D(p::Tensors.Vec{2,Float64}) = new(p[1], p[2], 0)
end
GP.Point(x::Real, y::Real, k::Int64) = NumberedPoint2D(x, y, k)
GP.Point2D(p::NumberedPoint2D) = Point2D(p.x,p.y)
GP.gety(p::NumberedPoint2D) = p.y
GP.getx(p::NumberedPoint2D) = p.x

#More or less equivalent to matlab's delaunay function, based on code from FEMDL.jl

function delaunay2(x::Vector{Tensors.Vec{2,Float64}})
    width = VD.max_coord - VD.min_coord
    max_x = maximum(map(v->v[1], x))
    min_x = minimum(map(v->v[1], x))
    max_y = maximum(map(v->v[2], x))
    min_y = minimum(map(v->v[2], x))
    scale_x = 0.9*width/(max_x - min_x)
    scale_y = 0.9*width/(max_y - min_y)
    n = length(x)
    a = [NumberedPoint2D(VD.min_coord+(x[i][1] - min_x)*scale_x, VD.min_coord+(x[i][2]-min_y)*scale_y, i) for i in 1:n]
    for i in 1:n
        @assert !(GP.getx(a[i]) < VD.min_coord || GP.gety(a[i]) > VD.max_coord)
    end
    tess = VD.DelaunayTessellation2D{NumberedPoint2D}(n)
    push!(tess, a)
    m = 0
    for i in tess
        m += 1
    end
    return tess, m, scale_x, scale_y, min_x, min_y
end


function JuAFEM.generate_grid(::Type{JuAFEM.Triangle}, nodes_in::Vector{Tensors.Vec{2,Float64}};on_torus=false,LL=[0.0,0.0],UR=[1.0,1.0])
    num_nodes_in = length(nodes_in)

    nodes_to_triangulate = Vector{Tensors.Vec{2,Float64}}()

    points_mapping = Vector{Int}()
    if on_torus
        dx = UR[1] - LL[1]
        dy = UR[2] - LL[2]
        for i in [0,1,-1]
            for j in [0,1,-1]
                for (index,node) in enumerate(nodes_in)
                    new_point = node .+ (i*dx,j*dy)
                    push!(nodes_to_triangulate,Tensors.Vec{2}((new_point[1],new_point[2])))
                    push!(points_mapping,index)
                end
            end
        end
    else
        for (index,node) in enumerate(nodes_in)
            push!(nodes_to_triangulate,node)
            push!(points_mapping,index)
        end
    end


    tess, m, scale_x, scale_y, min_x, min_y = delaunay2(nodes_to_triangulate)

    nodes = map(JuAFEM.Node, nodes_in)
    additional_nodes = Dict{Int,Int}()#which nodes outside of the torus do we need?
    cells_used = Dict{Tuple{Int,Int,Int},Int}()
    #See if this cell, or vertical and horizontal translations
    #of it have already been added to the triangulation.
    function in_cells_used(element::Tuple{Int,Int,Int})
        a,b,c = element
        function moveby(i,direction)
            return min(max(1,i + num_nodes_in*direction),9*num_nodes_in)
        end
        for direction in -9:9
            #TODO: optimize this for speed
            moved_cell = Tuple{Int,Int,Int}(sort(collect(moveby.(element,direction))))
            if moved_cell ∈ keys(cells_used)
                return cells_used[moved_cell]
            end
        end
        return 0
    end

    cells = JuAFEM.Triangle[]
    cell_number_table = zeros(Int, length(tess._trigs))

    tri_iterator = Base.iterate(tess)
    while tri_iterator != nothing
        (tri,triindex) = tri_iterator
        J = Tensors.otimes((nodes_to_triangulate[tri._b.id] - nodes_to_triangulate[tri._a.id]), e1)
        J += Tensors.otimes((nodes_to_triangulate[tri._c.id] - nodes_to_triangulate[tri._a.id]), e2)
        detJ = det(J)
        @assert detJ != 0
        if detJ > 0
            new_tri = JuAFEM.Triangle((tri._a.id, tri._b.id, tri._c.id))
        else
            new_tri = JuAFEM.Triangle((tri._a.id, tri._c.id, tri._b.id))
        end
        if !on_torus
            push!(cells, new_tri)
            cell_number_table[triindex.ix-1] = length(cells)
        else
            tri_nodes = [new_tri.nodes[i] for i in 1:3]
            #Are any of the vertices actually inside?
            if any(x -> x <= num_nodes_in, tri_nodes)
                thiscell = in_cells_used(new_tri.nodes)
                if thiscell != 0
                    cell_number_table[triindex.ix-1] = thiscell
                else
                    for (index,cur) in enumerate(tri_nodes)
                        if cur > num_nodes_in
                            if cur ∈ keys(additional_nodes)
                                tri_nodes[index] = additional_nodes[cur]
                            else
                                push!(nodes, JuAFEM.Node(nodes_to_triangulate[cur]))
                                additional_nodes[cur] = length(nodes)
                                tri_nodes[index] = length(nodes)
                            end
                        end
                    end
                    new_tri = JuAFEM.Triangle((tri_nodes[1],tri_nodes[2],tri_nodes[3]))
                    push!(cells, new_tri)
                    cell_number_table[triindex.ix-1] = length(cells)
                    cells_used[ Tuple{Int,Int,Int}(sort(tri_nodes)) ] = length(cells)
                end
            end
        end
        tri_iterator = Base.iterate(tess,triindex)
    end

    facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = JuAFEM.Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    locator = delaunayCellLocator(m, scale_x, scale_y, min_x, min_y, tess,nodes_to_triangulate,points_mapping,cell_number_table)
    return grid, locator
end

function JuAFEM.generate_grid(::Type{JuAFEM.QuadraticTriangle}, nodes_in::Vector{Tensors.Vec{2,Float64}})

    nodes_to_triangulate = Vector{Tensors.Vec{2,Float64}}[]
    dx = UR[1] - LL[1]
    dy = UR[2] - LL[2]

    points_mapping = Vector{Int}[]

    for (index,node) in enumerate(nodes_in)
        if on_torus
            for i in -1:1
                for j in -1:1
                    new_point = node .+ (i*dx,j*dy)
                    push!(nodes_to_triangulate,new_point)
                    push!(points_mapping,index)
                end
            end
        else
            push!(nodes_to_triangulate)
            push!(points_mapping,index)
        end
    end
    for (index,node) in enumerate(nodes_in)
        if on_torus
            for i in -1:1
                for j in -1:1
                    new_point = node .+ (i*dx,j*dy)
                    push!(new_points,new_point)
                    push!(points_mapping,index)
                end
            end
        else
        push!(new_points,node)
        push!(points_mapping,index)
        end
    end
    tess, m, scale_x, scale_y, minx, miny = delaunay2(nodes_to_triangulate)
    locator = p2DelaunayCellLocator(m, scale_x, scale_y, minx, miny, tess,points_mapping)
    nodes = map(JuAFEM.Node, nodes_in)
    n = length(nodes)
    ctr = n #As we add nodes (for edge vertices), increment the ctr...

    centerNodes = spzeros(n,n)
    cells = JuAFEM.QuadraticTriangle[]
    for tri_id in 1:m
        tri = tess._trigs[locator.internal_triangles[tri_id]]

        #Create non-vertex nodes
        ab = centerNodes[tri._a.id, tri._b.id]
        if ab == 0
            ctr += 1
            ab = centerNodes[tri._a.id,tri._b.id] = centerNodes[tri._b.id,tri._a.id] =  ctr
            center = JuAFEM.Node(0.5*(nodes[tri._b.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end
        ac = centerNodes[tri._a.id, tri._c.id]
        if ac == 0
            ctr += 1
            ac = centerNodes[tri._a.id,tri._c.id] = centerNodes[tri._c.id,tri._a.id] = ctr
            center = JuAFEM.Node(0.5*(nodes[tri._c.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end

        bc = centerNodes[tri._c.id, tri._b.id]
        if bc == 0
            ctr += 1
            bc = centerNodes[tri._b.id,tri._c.id] = centerNodes[tri._c.id,tri._b.id] = ctr
            center = JuAFEM.Node(0.5*(nodes[tri._c.id].x + nodes[tri._b.id].x))
            push!(nodes,center)
        end

        J = Tensors.otimes((nodes_in[tri._b.id] - nodes_in[tri._a.id]) , e1)
        J +=  Tensors.otimes((nodes_in[tri._c.id] - nodes_in[tri._a.id]) , e2)
        detJ = det(J)

        @assert det(J) != 0
        if detJ > 0
            new_tri = JuAFEM.QuadraticTriangle((tri._a.id,tri._b.id,tri._c.id,ab,bc,ac))
        else
            new_tri = JuAFEM.QuadraticTriangle((tri._a.id,tri._c.id,tri._b.id,ac,bc,ab))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = JuAFEM.Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    return grid, locator

end

"""
    nodal_interpolation(ctx,f)

Perform nodal interpolation of a function. Returns a vector of coefficients in dof order
"""
function nodal_interpolation(ctx::gridContext, f::Function)
    nodal_values = [f(ctx.grid.nodes[ctx.dof_to_node[j]].x) for j in 1:ctx.n]
end



###P2 Grids in 3D:
#TODO: See if this can be moved upstream

#Based on JuAFEM's generate_grid(Tetrahedron, ...) function
function JuAFEM.generate_grid(::Type{JuAFEM.QuadraticTetrahedron}, cells_per_dim::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
    nodes_per_dim = (2 .* cells_per_dim) .+ 1

    cells_per_cube = 6
    total_nodes = prod(nodes_per_dim)
    total_elements = cells_per_cube * prod(cells_per_dim)

    n_nodes_x, n_nodes_y, n_nodes_z = nodes_per_dim
    n_cells_x, n_cells_y, n_cells_z = cells_per_dim

    # Generate nodes
    coords_x = range(left[1], stop=right[1], length=n_nodes_x)
    coords_y = range(left[2], stop=right[2], length=n_nodes_y)
    coords_z = range(left[3], stop=right[3], length=n_nodes_z)
    numbering = reshape(1:total_nodes, nodes_per_dim)

    # Pre-allocate the nodes & cells
    nodes = Vector{JuAFEM.Node{3,T}}(undef,total_nodes)
    cells = Vector{JuAFEM.QuadraticTetrahedron}(undef,total_elements)

    # Generate nodes
    node_idx = 1
    @inbounds for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        nodes[node_idx] = JuAFEM.Node((coords_x[i], coords_y[j], coords_z[k]))
        node_idx += 1
    end

    # Generate cells, case 1 from: http://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
    # cube = (1, 2, 3, 4, 5, 6, 7, 8)
    # left = (1, 4, 5, 8), right = (2, 3, 6, 7)
    # front = (1, 2, 5, 6), back = (3, 4, 7, 8)
    # bottom = (1, 2, 3, 4), top = (5, 6, 7, 8)
    cell_idx = 0
    #TODO add @inbounds once this works...
    for k in 1:n_cells_z, j in 1:n_cells_y, i in 1:n_cells_x
        cube = numbering[(2*(i-1) + 1):(2*i + 1), (2*(j-1)+1): 2*j + 1, (2*(k-1) +1): (2*k +1)]

        localnodes = [  ((1,1,1),(3,1,1),(1,3,1),(1,3,3)),
                        ((1,1,1),(1,1,3),(3,1,1),(1,3,3)),
                        ((3,1,1),(3,3,1),(1,3,1),(1,3,3)),
                        ((3,1,1),(3,3,3),(3,3,1),(1,3,3)),
                        ((3,1,1),(1,1,3),(3,1,3),(1,3,3)),
                        ((3,1,1),(3,1,3),(3,3,3),(1,3,3))
                        ]
        avg(x,y) = (x == 1 && y == 3) || (x == 3 && y == 1) ? 2 : x
        indexavg(x,y) = CartesianIndex(avg.(Tuple(x),Tuple(y)))
        for (idx, p1vertices) in enumerate(localnodes)
            v1,v2,v3,v4 = map(CartesianIndex,p1vertices)
            cells[cell_idx + idx] = JuAFEM.QuadraticTetrahedron((cube[v1],cube[v2],cube[v3],cube[v4],
                        cube[indexavg(v1,v2)],cube[indexavg(v2,v3)],cube[indexavg(v1,v3)],cube[indexavg(v1,v4)],
                        cube[indexavg(v2,v4)],cube[indexavg(v3,v4)]))
        end
        cell_idx += cells_per_cube
    end

    # Order the cells as c_nxyz[n, x, y, z] such that we can look up boundary cells
    c_nxyz = reshape(1:total_elements, (cells_per_cube, cells_per_dim...))

    @views le = [map(x -> (x,4), c_nxyz[1, 1, :, :][:])   ; map(x -> (x,2), c_nxyz[2, 1, :, :][:])]
    @views ri = [map(x -> (x,1), c_nxyz[4, end, :, :][:]) ; map(x -> (x,1), c_nxyz[6, end, :, :][:])]
    @views fr = [map(x -> (x,1), c_nxyz[2, :, 1, :][:])   ; map(x -> (x,1), c_nxyz[5, :, 1, :][:])]
    @views ba = [map(x -> (x,3), c_nxyz[3, :, end, :][:]) ; map(x -> (x,3), c_nxyz[4, :, end, :][:])]
    @views bo = [map(x -> (x,1), c_nxyz[1, :, :, 1][:])   ; map(x -> (x,1), c_nxyz[3, :, :, 1][:])]
    @views to = [map(x -> (x,3), c_nxyz[5, :, :, end][:]) ; map(x -> (x,3), c_nxyz[6, :, :, end][:])]

    boundary_matrix = JuAFEM.boundaries_to_sparse([le; ri; bo; to; fr; ba])

    facesets = Dict(
        "left" => Set(le),
        "right" => Set(ri),
        "front" => Set(fr),
        "back" => Set(ba),
        "bottom" => Set(bo),
        "top" => Set(to),
    )
    return JuAFEM.Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end
