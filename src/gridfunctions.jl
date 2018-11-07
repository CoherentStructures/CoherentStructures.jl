#(c) 2017 Nathanael Schilling
#This file implements methods for working with JuAFEM grids
#This includes methods for making grids from Delaunay Triangulations based
#on the code in FEMDL.jl

#JuAFEM has no functions for determining which cell a point is in.
#Amongst other things, this file (and pointlocation.jl) implements an API for doing this.

const default_quadrature_order = 5
const default_quadrature_order3D = 2


"""
    struct gridContext<dim>

Stores everything needed as "context" to be able to work on a FEM grid based on the `JuAFEM` package.
Adds a point-locator API which facilitates plotting functions defined on the grid within Julia.

# Fields
- `grid::JuAFEM.Grid`, `ip::JuAFEM.Interpolation`,`ip_geom::JuAFEM.Interpolation`, `qr::JuAFEM.QuadratureRule` - See the `JuAFEM` package
- `loc::CellLocator` object used for point-location on the grid.

- `node_to_dof::Vector{Int}`  lookup table for dof index of a node (for Lagrange elements)
- `dof_to_node::Vector{Int}`  inverse of node_to_dof

- `cell_to_dof::Vector{Int}`  lookup table for dof index of a cell (for piecewise constant elements)
- `dof_to_cell::Vector{Int}`  inverse of cell_to_dof

- `num_nodes::Int` number of nodes on the grid
- `num_cells::Int` number of elements (e.g. triangles,quadrilaterals, ...) on the grid
- `n` number of degrees of freedom (== `num_nodes` for Lagrange Elements, and == `num_cells` for piecewise constant elements)

- `quadrature_points::Vector{Vec{dim,Float64}}` All quadrature points on the grid, in a fixed order.
- `mass_weights::Vector{Float64}` Weighting for stiffness/mass matrices

- `spatialBounds` If available, the corners of a bounding box of a domain. For regular grids, the bounds are tight.
- `numberOfPointsInEachDirection` For regular grids, how many (non-interior) nodes make up the regular grid.
- `gridType` A string describing what kind of grid this is (e.g. "regular triangular grid")
"""
mutable struct gridContext{dim} <: abstractGridContext{dim} #TODO: Currently set as mutable, is this sensible?
    grid::JFM.Grid
    ip::JFM.Interpolation
    ip_geom::JFM.Interpolation
    dh::JFM.DofHandler
    qr::JFM.QuadratureRule
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

    quadrature_points::Vector{Vec{dim,Float64}} #All quadrature points, ordered by how they are accessed in assemble routines
    mass_weights::Vector{Float64}

    ##The following two fields are only well-defined for regular rectangular grids
    spatialBounds::Vector{AbstractVector} #This is {LL,UR} for regular grids
    #This is the number of (non-interior) nodes in each direction (not points)
    numberOfPointsInEachDirection::Vector{Int}

    gridType::String

    function gridContext{dim}(
                grid::JFM.Grid,
                ip::JFM.Interpolation,
                ip_geom::JFM.Interpolation,
                dh::JFM.DofHandler,
                qr::JFM.QuadratureRule,
                loc::pointLocator
            ) where {dim}

        x =new{dim}(grid, ip,ip_geom, dh, qr, loc)
        x.num_nodes = JFM.getnnodes(dh.grid)
        x.num_cells = JFM.getncells(dh.grid)
        x.n = JFM.ndofs(dh)

        #TODO: Measure if the sorting below is expensive
        if isa(ip,JFM.Lagrange)
            x.node_to_dof = nodeToDHTable(x)
            x.dof_to_node = sortperm(x.node_to_dof)
        elseif isa(ip,JFM.PiecewiseConstant)
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
    dh::JFM.DofHandler = ctx.dh
    n = ctx.n
    res = Vector{Int}(undef,n)
    for cell in JFM.CellIterator(dh)
        _celldofs = JFM.celldofs(cell)
        ctr = 1
        offset = JFM.field_offset(dh, dh.field_names[1])
        for node in JFM.getnodes(cell)
               res[node] = _celldofs[ctr + offset]
               ctr += 1
        end
    end
    return res
end

function cellToDHTable(ctx::abstractGridContext{dim}) where {dim}
    dh::JFM.DofHandler = ctx.dh
    n = ctx.n
    res = Vector{Int}(undef,n)
    for (cellindex,cell) in enumerate(JFM.CellIterator(dh))
        _celldofs = JFM.celldofs(cell)
        offset = JFM.field_offset(dh, dh.field_names[1])
        for node in JFM.getnodes(cell)
            mynode = node
        end
        res[cellindex] = _celldofs[offset+1]
    end
    return res
end
#=
"""
    gridContext{1}(JFM.Line, [numnodes, LL, UR; ip,quadrature_order,ip])

Constructor for a 1d regular mesh with `numnodes[1]` node on the interval `[LL[1],UR[1]]`.
The default for `ip` is P1-Lagrange elements, but piecewise-constant elements can also be used.
"""
=#
function gridContext{1}(::Type{JFM.Line},
                         numnodes::Tuple{Int}=( 25), LL::AbstractVector=[0.0], UR::AbstractVector=[1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JFM.Lagrange{1,JFM.RefCube,1}(),
                         )
    # The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.Line, (numnodes[1]-1,), Vec{1}(LL), Vec{1}(UR))
    loc = regular1dGridLocator{JFM.Line}(numnodes[1], Vec{1}(LL), Vec{1}(UR))

    dh = JFM.DofHandler(grid)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)

    qr = JFM.QuadratureRule{1, JFM.RefCube}(quadrature_order)
    result = gridContext{1}(grid, ip,JFM.Lagrange{1,JFM.RefCube,1}(), dh, qr, loc)
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1]]
    if isa(ip, JFM.Lagrange)
        result.gridType = "regular P1 1d grid"
    else
        result.gridType = "regular PC 1d grid"
    end

    return result
end

#=
"""
    gridContext{1}(JuAFEM.QuadraticLine, numnodes, LL, UR, quadrature_order)
"""
=#
function gridContext{1}(::Type{JFM.QuadraticLine},
                         numnodes::Tuple{Int}=( 25),
                         LL::AbstractVector=[0.0],
                         UR::AbstractVector=[1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JFM.Lagrange{1,JFM.RefCube,2}()
                         )
    # The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.QuadraticLine, (numnodes[1]-1,), Vec{1}(LL), Vec{1}(UR))
    loc = regular1dGridLocator{JFM.QuadraticLine}(numnodes[1], Vec{1}(LL), Vec{1}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{1, JFM.RefCube}(quadrature_order)
    push!(dh, :T, 1) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result = gridContext{1}(grid, ip,JFM.Lagrange{1,JFM.RefCube,2}(), dh, qr, loc)
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1]]
    if !isa(ip, JFM.Lagrange)
        @warn("Using non-Lagrange shape functions may or may not work on this mesh")
    end
    result.gridType = "regular P2 1d grid"
    return result
end

const regular1dGridTypes = [
    "regular PC 1d grid",
    "regular P1 1d grid",
    "regular P2 1d grid"
    ]


"""
    regular1dGrid(numnodes,left=0.0,right=1.0; [quadrature_order, PC=false])

Create a regular grid with `numnodes` nodes on the interval `[left,right]` in 1d.
If `PC==false`, uses P1-Lagrange basis functions. If `PC=true`, uses piecewise-constant
basis functions.
"""
function regular1dGrid(
    numnodes::Int,left=0.0,right=1.0;
    quadrature_order::Int=default_quadrature_order,
    PC=false
    )
    if !PC
        result = gridContext{1}(JFM.Line,(numnodes,), [left],[right]; quadrature_order=quadrature_order)
    else
        result = gridContext{1}(JFM.Line,(numnodes,),
        [left],[right];
        quadrature_order=quadrature_order,
        ip=JFM.PiecewiseConstant{1,JFM.RefCube,1}()
        )
    end
    return result, boundaryData()
end

function regular1dGrid(numnodes::Tuple{Int},args...;kwargs...)
    return regular1dGrid(numnodes[1],args...;kwargs...)
end

"""
    regular1dP2Grid(numnodes, [left,right; quadrature_order])

Create a regular grid with `numnodes` non-interior nodes on the interval `[left,right]`.
Uses P2-Lagrange elements.
"""
function regular1dP2Grid(
    numnodes::Int,left=0.0,right=1.0;
    quadrature_order::Int=default_quadrature_order
    )
    result = gridContext{1}(
        JFM.QuadraticLine,
        (numnodes,),
         [left],[right];
         quadrature_order=quadrature_order
         )
    return result, boundaryData()
end

function regular1dP2Grid(numnodes::Tuple{Int},args...;kwargs...)
    return regular1dP2Grid(numnodes[1],args...;kwargs...)
end

const regular2dGridTypes = [
                    "regular PC triangular grid",
                    "regular P1 triangular grid",
                    "regular P2 triangular grid",
                    "regular PC Delaunay grid",
                    "regular P1 Delaunay grid",
                    "regular P2 Delaunay grid",
                    "regular PC quadrilateral grid",
                    "regular P1 quadrilateral grid",
                    "regular P2 quadrilateral grid"
                    ]

"""
    regular2dGrid(gridType, numnodes, LL=[0.0,0.0],UR=[1.0,1.0];quadrature_order=default_quadrature_order)

Constructs a regular grid. `gridType` should be from `CoherentStructures.regular2dGridTypes`
"""
function regular2dGrid(
            gridType::String,
            numnodes::Tuple{Int,Int},
            LL::AbstractVector=[0.0,0.0],
            UR::AbstractVector=[1.0,1.0];
            kwargs...
        )

    if gridType == "regular PC triangular grid"
        return regularTriangularGrid(numnodes, LL, UR;PC=true, kwargs...)
    elseif gridType == "regular P1 triangular grid"
        return regularTriangularGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular PC Delaunay grid"
        return regularDelaunayGrid(numnodes, LL, UR;PC=true, kwargs...)
    elseif gridType == "regular P1 Delaunay grid"
        return regularDelaunayGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 triangular grid"
        return regularP2TriangularGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 Delaunay grid"
        return regularP2DelaunayGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular PC quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL, UR;PC=true, kwargs...)
    elseif gridType == "regular P1 quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 quadrilateral grid"
        return regularP2QuadrilateralGrid(numnodes, LL, UR; kwargs...)
    else
        fail("Unsupported grid type '$gridType'")
    end
end


#= #TODO 1.0
"""
    gridContext{2}(JuAFEM.Triangle, node_list; [on_torus=false,LL,UR,quadrature_order=default_quadrature_order,ip])

Create a P1-Lagrange grid based on Delaunay Triangulation.
If `on_torus==true`, triangulates on a periodic domain (in both directions)
defined by `LL` (lower-left corner) and `UR` (upper-right corner).
The parameter `ip` defines what kind of shape functions to use, the default is P1-Lagrange (can also be piecewise constant).
i
Uses `DelaunayVoronoi.jl` internally.
"""
=#
function gridContext{2}(
            ::Type{JFM.Triangle},
            node_list::Vector{Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order,
            on_torus=false,
            LL=nothing,
            UR=nothing,
            ip=JFM.Lagrange{2,JFM.RefTetrahedron,1}(),
            )
    if on_torus
        @assert !( LL == nothing || UR == nothing)
    end
    grid, loc = JFM.generate_grid(
            JFM.Triangle, node_list;
            on_torus=on_torus,LL=LL,UR=UR
            )
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{2, JFM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result =  gridContext{2}(grid, ip,JFM.Lagrange{2,JFM.RefTetrahedron,1}(), dh, qr, loc)
    if isa(ip, JFM.Lagrange)
        result.gridType = "irregular P1 Delaunay grid"
    else
        result.gridType = "irregular PC Delaunay grid"
    end
    if LL == nothing
        LL = [minimum([x[i] for x in node_list]) for i in 1:2]
    end
    if UR == nothing
        UR = [maximum([x[i] for x in node_list]) for i in 1:2]
    end
    result.spatialBounds = [LL,UR]
    return result
end

"""
    irregularDelaunayGrid(nodes_in; [on_torus=true,LL,UR,PC=false,...])

Triangulate the nodes `nodes_in` and return a `gridContext` and `bdata` for them.
If `on_torus==true`, the triangulation is done on a torus.
If `PC==true`, return a mesh with piecewise constant shape-functions, else P1 Lagrange.
"""
function irregularDelaunayGrid(nodes_in::Vector{Vec{2,Float64}};
    on_torus=false, LL=nothing, UR=nothing,
    PC=false,
    kwargs...
    )
    if on_torus
        @assert !(LL == nothing || UR == nothing)
    end
    if !PC
        ip = JFM.Lagrange{2,JFM.RefTetrahedron,1}()
    else
        ip = JFM.PiecewiseConstant{2,JFM.RefTetrahedron,1}()
    end
    ctx = CoherentStructures.gridContext{2}(JFM.Triangle,
        nodes_in,on_torus=on_torus,LL=LL,UR=UR,ip=ip,
        kwargs...
        )
    if !on_torus
        bdata = boundaryData()
    else
        bdata = boundaryData(ctx,PEuclidean(UR .- LL))
    end
    return ctx, bdata
end

"""
    randomDelaunayGrid(npoints; LL, UR,...)
Create a delaunay grid in 2d from `npoints` random points on the box with lower
left corner `LL` and upper right corner `UR`.
Extra keyword arguments are passed down to `irregularDelaunayGrid`.
"""
function randomDelaunayGrid(
                    npoints::Int;
                    LL=[0.0,0.0],
                    UR=[1.0,1.0],
                    kwargs...
    )

    nodes_in::Vector{Vec{2,Float64}} = Vec{2}.(zip(rand(npoints).*(UR[1]-LL[1]) .+ LL[1],rand(npoints).*(UR[2]-LL[2]) .+ LL[2]))
    return irregularDelaunayGrid(nodes_in; LL=LL,UR=UR,kwargs...)
end


"""
    regularDelaunayGrid(numnodes=(25,25), LL, UR; [quadrature_order,on_torus=false, nudge_epsilon=1e-5,PC=false])

Create a regular grid on a square with lower left corner `LL` and upper-right corner `UR`.
Internally uses Delauny Triangulation.
If `on_torus==true`, uses a periodic Delaunay triangulation. To avoid degenerate special cases,
all nodes are given a random `nudge`, the strength of which depends on `numnodes` and `nudge_epsilon`.
If `PC==true`, returns a piecewise constant grid. Else returns a P1-Lagrange grid.
"""
function regularDelaunayGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order,
            on_torus=false,
            nudge_epsilon::Float64=1e-5,
            PC=false
        )
    X = range(LL[1],stop= UR[1],length= numnodes[1])
    Y = range(LL[2], stop=UR[2], length=numnodes[2])
    if PC
        ip=JFM.PiecewiseConstant{2,JFM.RefTetrahedron,1}()
    else
        ip=JFM.Lagrange{2,JFM.RefTetrahedron,1}()
    end
    node_list = vec([Vec{2}((x, y)) for y in Y, x in X])
    if on_torus
        function nudge(point)
            nudgefactor = (UR .- LL)  .* nudge_epsilon ./ numnodes
            return Vec{2}(
                max.(LL .+ 0.1*nudgefactor, min.(UR .- 0.1*nudgefactor,
                    point .+ nudgefactor .* rand(2)
                    )))
        end
        node_list = nudge.(filter(x -> minimum(abs.(x .- UR)) > 1e-8, node_list))
    end
    result = CoherentStructures.gridContext{2}(JFM.Triangle,
         node_list;
         quadrature_order=quadrature_order,
         on_torus=on_torus,
         LL=LL,
         UR=UR,
         ip=ip
         )
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    if !PC
        result.gridType = "regular P1 Delaunay grid"
    else
        result.gridType = "regular PC Delaunay grid"
    end
    if !PC
        bdata = boundaryData(result,PEuclidean(UR .- LL))
    else
        bdata = boundaryData()
    end
    return result, bdata
end


#= TODO 1.0
"""
    gridContext{2}(JuAFEM.QuadraticTriangle, node_list, quadrature_order=default_quadrature_order)

Create a P2 grid given a set of (non-interior) nodes using Delaunay Triangulation.
"""
=#
function gridContext{2}(
            ::Type{JFM.QuadraticTriangle},
            node_list::Vector{Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order,
            ip=JFM.Lagrange{2,JFM.RefTetrahedron,2}()
            )
    grid, loc = JFM.generate_grid(JFM.QuadraticTriangle, node_list)
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{2, JFM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result = gridContext{2}(grid, ip,JFM.Lagrange{2,JFM.RefTetrahedron,2}(), dh, qr, loc)
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
    node_list = vec([Vec{2}((x, y)) for y in Y, x in X])
    result = gridContext{2}(JFM.QuadraticTriangle, node_list, quadrature_order=quadrature_order)
    #TODO: Think about what values would be sensible for the two variables below
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 Delaunay grid"
    return result, boundaryData()
end

#=
"""
    gridContext{2}(JuAFEM.Triangle, numnodes, LL,UR; [quadrature_order])

Create a regular triangular grid. Does not use Delaunay triangulation internally.
"""
=#

function gridContext{2}(::Type{JFM.Triangle},
                         numnodes::Tuple{Int,Int}=(25, 25),
                         LL::AbstractVector=[0.0, 0.0],
                         UR::AbstractVector=[1.0, 1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JFM.Lagrange{2,JFM.RefTetrahedron,1}(),
                         )
    # The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.Triangle, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = regular2DGridLocator{JFM.Triangle}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{2, JFM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result = gridContext{2}(grid, ip,JFM.Lagrange{2,JFM.RefTetrahedron,1}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular triangular grid"
    return result
end

"""
    regularTriangularGrid(numnodes, LL,UR;[quadrature_order, PC=false])

Create a regular triangular grid on a rectangle; does not use Delaunay triangulation internally.
If
"""
function regularTriangularGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=[0.0,0.0], UR::AbstractVector=[1.0,1.0];
                                quadrature_order::Int=default_quadrature_order, PC=false)
    if PC == false
        ip = JFM.Lagrange{2,JFM.RefTetrahedron,1}
    else
        ip = JFM.PiecewiseConstant{2,JuAFEMRefTetrahedron,1}
    end
    ctx =  gridContext{2}(JFM.Triangle,
            numnodes, LL, UR;
            quadrature_order=quadrature_order,ip=ip
            )
    return ctx, boundaryData()
end




#= TODO 1.0
"""
    gridContext{2}(JuAFEM.QuadraticTriangle, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 triangular grids. Does not use Delaunay triangulation internally.
"""
=#
function gridContext{2}(::Type{JFM.QuadraticTriangle},
                         numnodes::Tuple{Int,Int}=(25, 25),
                         LL::AbstractVector=[0.0,0.0],
                         UR::AbstractVector=[1.0,1.0];
                         quadrature_order::Int=default_quadrature_order,
                         ip=JFM.Lagrange{2,JFM.RefTetrahedron,2}()
                         )
    if !isa(ip,JFM.Lagrange)
        @warn "Using non-Lagrange interpolation with P2 elements may or may not work"
    end
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.QuadraticTriangle, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = regular2DGridLocator{JFM.QuadraticTriangle}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{2, JFM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result =  gridContext{2}(grid, ip,JFM.Lagrange{2,JFM.RefTetrahedron,2}(), dh, qr, loc)
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
    ctx = gridContext{2}(JFM.QuadraticTriangle, numnodes, LL, UR, quadrature_order=quadrature_order)
    return ctx, boundaryData()
end

#= TODO 1.0
"""
    gridContext{2}(JuAFEM.Quadrilateral, numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0], quadrature_order=default_quadrature_order)

Constructor for regular P1 quadrilateral grids.
"""
=#
function gridContext{2}(
            ::Type{JFM.Quadrilateral},
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=[0.0,0.0],
            UR::AbstractVector=[1.0,1.0];
            quadrature_order::Int=default_quadrature_order,
            ip=JFM.Lagrange{2, JFM.RefCube, 1}()
            )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.Quadrilateral, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = regular2DGridLocator{JFM.Quadrilateral}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{2, JFM.RefCube}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result =  gridContext{2}(grid, ip,JFM.Lagrange{2,JFM.RefCube,1}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    if isa(ip, JFM.Lagrange)
        result.gridType = "regular P1 quadrilateral grid"
    else
        result.gridType = "regular PC quadrilateral grid"
    end
    return result
end

"""
    regularP2QuadrilateralGrid(numnodes, LL,UR;[quadrature_order, PC=false]

Create a regular P1 quadrilateral grid on a Rectangle. If `PC==true`, use
piecewise constant shape functions. Else use P1 Lagrange.
"""
function regularQuadrilateralGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order,
            PC=false
        )
    if !PC
        ip = JFM.Lagrange{2,JFM.RefCube,1}
    else
        ip = JFM.PiecewiseConstant{2,JFM.RefCube,1}
    end
    ctx = gridContext{2}(JFM.Quadrilateral,
         numnodes, LL, UR;
         quadrature_order=quadrature_order,
         ip=ip
     )
     return ctx, boundaryData()
end


#= TODO 1.0
"""
    gridContext{2}(JUAFEM.QuadraticQuadrilateral, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 quadrilateral grids.
"""
=#
function gridContext{2}(
            ::Type{JFM.QuadraticQuadrilateral},
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::AbstractVector=[0.0, 0.0],
            UR::AbstractVector=[1.0, 1.0];
            quadrature_order::Int=default_quadrature_order,
            ip=JFM.Lagrange{2, JFM.RefCube, 2}()
            )
    if !isa(ip,JFM.Lagrange)
        @warn "Non-Lagrange interpolation with P2 elements may or may not work"
    end
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.QuadraticQuadrilateral, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = regular2DGridLocator{JFM.QuadraticQuadrilateral}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{2, JFM.RefCube}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result =  gridContext{2}(grid, ip,JFM.Lagrange{2,JFM.RefCube,2}(), dh, qr, loc)
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
    ctx = gridContext{2}(JFM.QuadraticQuadrilateral, numnodes, LL, UR, quadrature_order=quadrature_order)
    return ctx, boundaryData()
end
#=TODO 1.0
"""
    gridContext{3}(JuAFEM.Tetrahedron, numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P1 Tetrahedral Grid in 3D.
"""
=#
function gridContext{3}(::Type{JFM.Tetrahedron},
                         numnodes::Tuple{Int,Int,Int}=(10,10,10),
                         LL::AbstractVector=[0.0,0.0,0.0],
                         UR::AbstractVector=[1.0,1.0,1.0];
                         quadrature_order::Int=default_quadrature_order3D,
                         ip=JFM.Lagrange{3, JFM.RefTetrahedron, 1}()
                         )
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.Tetrahedron, (numnodes[1]-1, numnodes[2]-1, numnodes[3] -1), Vec{3}(LL), Vec{3}(UR))
    loc = regular3DGridLocator{JFM.Tetrahedron}(numnodes[1], numnodes[2], numnodes[3], Vec{3}(LL), Vec{3}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{3, JFM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result =  gridContext{3}(grid, ip,JFM.Lagrange{3,JFM.RefCube,1}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2], numnodes[3]]
    if isa(ip, JFM.Lagrange)
        result.gridType = "3D P1 regular tetrahedral grid"
    else
       result.gridType = "3D PC regular tetrahedral grid"
    end
    return result
end


#=TODO 1.0
"""
    gridContext{3}(JuAFEM.QuadraticTetrahedron, numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P2 Tetrahedral Grid in 3D.
"""
=#
function gridContext{3}(::Type{JFM.QuadraticTetrahedron},
                         numnodes::Tuple{Int,Int,Int}=(10,10,10),
                         LL::AbstractVector=[0.0,0.0,0.0],
                         UR::AbstractVector=[1.0,1.0,1.0];
                         quadrature_order::Int=default_quadrature_order3D,
                         ip=JFM.Lagrange{3, JFM.RefTetrahedron, 2}()
                         )
    if !isa(ip, JUAFEM.Lagrange)
        @warn "Using non-Lagrange interpolation with P2 Elements may or may not work"
    end
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = JFM.generate_grid(JFM.QuadraticTetrahedron, (numnodes[1]-1, numnodes[2]-1, numnodes[3] -1), Vec{3}(LL), Vec{3}(UR))
    loc = regular3DGridLocator{JFM.QuadraticTetrahedron}(numnodes[1], numnodes[2], numnodes[3], Vec{3}(LL), Vec{3}(UR))
    dh = JFM.DofHandler(grid)
    qr = JFM.QuadratureRule{3, JFM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    JFM.close!(dh)
    result =  gridContext{3}(grid, ip,JFM.Lagrange{3,RefCube,2}(), dh, qr, loc)
    result.spatialBounds = [LL, UR]
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2], numnodes[3]]
    result.gridType = "3D regular P2 tetrahedral grid"
    return result
end

"""
    regularTetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P1 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularTetrahedralGrid(
        numnodes::Tuple{Int,Int,Int}=(10,10,10),
        LL::AbstractVector=[0.0,0.0,0.0],
        UR::AbstractVector=[1.0,1.0,1.0];
        quadrature_order::Int=default_quadrature_order3D, PC=false
        )
    if !PC
        ip = JFM.Lagrange{3,JFM.RefTetrahedron,1}()
    else
        ip = JFM.PiecewiseConstant{3,JFM.RefTetrahedron,1}()
    end
    ctx =  gridContext{3}(JFM.Tetrahedron,
        numnodes, LL, UR;
        quadrature_order=quadrature_order,ip=ip
        )
    return ctx, boundaryData()
end

"""
    regularP2TetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P2 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularP2TetrahedralGrid(numnodes::Tuple{Int,Int,Int}=(10,10,10), LL::AbstractVector=[0.0,0.0,0.0], UR::AbstractVector=[1.0,1.0,1.0];
                                    quadrature_order::Int=default_quadrature_order3D)
    ctx =  gridContext{3}(JFM.QuadraticTetrahedron,
        numnodes, LL, UR;
        quadrature_order=quadrature_order
        )
    return ctx, boundaryData()
end


"""
    project_in_xin(ctx,x_in,project_in)

Converts `x_in` to `Vec{dim}`, possibly taking pointwise maxima/minima to make
sure it is within `ctx.spatialBounds` (if `project_in==true`).
Helper function.
"""
function project_in_xin(
    ctx::gridContext{dim}, x_in::AbstractVector{T},project_in) where {dim,T}

    if !project_in
        if dim == 1
            x = Vec{dim,T}((x_in[1],))
        elseif dim == 2
            x = Vec{dim,T}((x_in[1], x_in[2]))
        elseif dim == 3
            x = Vec{dim,T}((x_in[1], x_in[2], x_in[3]))
        else
            error("dim = $dim not supported")
        end
    else
        if dim == 1
            x = Vec{dim,T}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1], x_in[1])),
                ))
        elseif dim == 2
            #TODO: replace this with a macro maybe
            x = Vec{dim,T}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1], x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2], x_in[2]))
                ))
        elseif dim == 3
            #TODO: replace this with a macro maybe
            x = Vec{dim,T}(
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
    x_in::Vec{dim,W}; outside_value=0.0, project_in=false
    )::W where {
        dim,S,W,
        }

    x::Vec{dim,W} = project_in_xin(ctx,x_in,project_in)

    @assert length(vals) == ctx.n

    local_coordinates::Vec{dim,Float64}, nodes::Vector{Int}, cellid::Int = try
         locatePoint(ctx, x)
    catch y
        if isa(y,DomainError)
            return spzeros(T,size(vals)[2]) .+ outside_value
        end
        if throw_errors
            print("Unexpected error for $x")
            rethrow(y)
        else
            return spzeros(T,size(vals)[2]) .+ outside_value
        end
    end

    result::S = zero(S)

    if isa(ctx.ip, JFM.Lagrange)
        for (j, nodeid) in enumerate(nodes)
            val::Float64 = JFM.value(ctx.ip, j, local_coordinates)
            result += vals[nodeid]*val
        end
    elseif isa(ctx.ip,JFM.PiecewiseConstant)
        val = JFM.value(ctx.ip, 1, local_coordinates)
        result += vals[cellid]*val
    else
        throw(AssertionError("Unknown interpolation"))
    end
    return result
end

"""
    evaluate_function_from_dofvals(ctx, dofvals, x_in; outside_value=0.0,project_in=fals)

Evaluate the function at point x_in with coefficients of dofs given by `dofvals` (in dof-order).
Return `outside_value` if point is out of bounds.
Project the point into the domain if `project_in==true`.
For evaluation at many points, or for many dofvals, the function `evaluate_function_from_dofvals_multiple`
is more efficient.
"""
function evaluate_function_from_dofvals(
    ctx::gridContext{dim}, vals::AbstractVector{S},
    x_in::Vec{dim,W}; outside_value=0.0, project_in=false
    )::W where {
        dim,S,W,
        }
    if isa(ctx.ip, JFM.Lagrange)
        vals_reorder = vals[ctx.node_to_dof]
    else
        vals_reorder = vals[ctx.cell_to_dof]
    end
    return evaluate_function_from_node_or_cellvals(
            ctx,vals_reorder, x_in;
            outside_value=outside_value,
            project_in=project_in
            )
end


"""
    evaluate_function_from_node_or_cellvals_multiple(ctx,vals,xin;is_diag=false,kwargs...)

Like `evaluate_function_from_dofvals_multiple` but uses node- (or cell- if piecewise constant interpolation)
ordering for `vals`, which makes it slightly more efficient.
If vals is a diagonal matrix, set `is_diag` to `true` for much faster evaluation.
"""
function evaluate_function_from_node_or_cellvals_multiple(
    ctx::gridContext{dim}, vals::AbstractMatrix{S},
    x_in::AbstractVector{Vec{dim,W}};
    outside_value=NaN, project_in=false,is_diag=false,throw_errors=false
    )::SparseMatrixCSC{S,Int64} where{dim,S,W}


    x::Vector{Vec{dim,W}} = [project_in_xin(ctx,x_cur,project_in) for x_cur in x_in]

    @assert size(vals)[1] == ctx.n
    npoints = length(x_in)

    ctr::Int = 1
    result_colptr=Int64[ctr]
    result_rows=Int64[]
    result_vals = S[]

    for current_point in 1:npoints
        rows_tmp = Int[]
        vals_tmp = S[]
        try
            local_coordinates::Vec{dim,Float64}, nodes::Vector{Int},cellid::Int = locatePoint(ctx,x[current_point])

            if isa(ctx.ip, JFM.Lagrange)
                if !is_diag
                    for i in 1:(size(vals)[2])
                        summed_value = 0.0
                        for (j, nodeid) in enumerate(nodes)
                            val::Float64 = JFM.value(ctx.ip, j, local_coordinates)
                            summed_value += vals[nodeid,i]*val
                        end
                        push!(rows_tmp,i)
                        push!(vals_tmp, summed_value)
                    end
                else
                    for (j, nodeid) in enumerate(nodes)
                        val = JFM.value(ctx.ip, j, local_coordinates)
                        push!(rows_tmp,nodeid)
                        push!(vals_tmp,vals[nodeid,nodeid]*val)
                    end
                end
            elseif isa(ctx.ip,JFM.PiecewiseConstant)
                val = JFM.value(ctx.ip, 1, local_coordinates)
                if !is_diag
                    for i in 1:(size(vals)[2])
                        push!(rows_tmp,i)
                        push!(vals_tmp, vals[cellid,i]*val)
                    end
                else
                    push!(rows_tmp,cellid)
                    push!(vals_tmp, vals[cellid,cellid]*val)
                end
            else
                throw(AssertionError("Unknown interpolation"))
            end
        catch y
                if isa(y,DomainError)
                    if outside_value != 0.0
                        rows_tmp=collect(1:(size(vals)[2]))
                        vals_tmp=[outside_value for i in 1:(size(vals)[2])]
                    end
                else
                    if throw_errors
                        print("Unexpected error for $(x[current_point])")
                        rethrow(y)
                    else
                        if outside_value != 0.0
                            rows_tmp=collect(1:size(vals)[2])
                            vals_tmp=[outside_value for i in 1:(size(vals)[2])]
                        end
                    end
                end
        end
        ordering = sortperm(rows_tmp)
        append!(result_rows,rows_tmp[ordering])
        append!(result_vals,vals_tmp[ordering])
        ctr += length(rows_tmp)
        push!(result_colptr,ctr)
    end
    return SparseMatrixCSC(size(vals)[2],npoints,result_colptr,result_rows,result_vals)
end


"""
    evaluate_function_from_dofvals_multiple(ctx,dofvals,x_in; [outside_value=0,project_in=false,throw_errors=false])

Evaluate a function in the approximation space at the vector of points `x_in`. If `x_in` is out of bounds, return `outside_value`.
If `project_in` is `true`, points not within `ctx.spatialBounds` are first projected into the domain.

Each column of the the matrix of coefficients in `dofvals` is interpreted to be in dof order.
The function is evaluated for each column.

If caught exceptions should be rethrown, set `throw_errors=true`
"""
function evaluate_function_from_dofvals_multiple(
    ctx::gridContext{dim}, dofvals::AbstractMatrix{S},
    x_in::AbstractVector{Vec{dim,W}};
    outside_value=0.0, project_in=false,throw_errors=false
    ) where { dim,S,W, }
    u_vals = zeros(S,size(dofvals))
    if isa(ctx.ip, JFM.Lagrange)
        for i in 1:ctx.n
            u_vals[i,:] = dofvals[ctx.node_to_dof[i],:]
        end
    elseif isa(ctx.ip, JFM.PiecewiseConstant)
        for i in 1:ctx.n
            u_vals[i,:] = dofvals[ctx.cell_to_dof[i],:]
        end
    end
    return evaluate_function_from_node_or_cellvals_multiple(ctx,u_vals, x_in,outside_value=outside_value,project_in=project_in,throw_errors=throw_errors)
end



"""
    sample_to(u::Vector{T},ctx_old,ctx_new)

Perform nodal_interpolation of a function onto a different grid.
"""
function sample_to(u::Vector{T}, ctx_old::CoherentStructures.gridContext, ctx_new::CoherentStructures.gridContext;
    bdata=boundaryData(),project_in=true,outside_value=NaN,
    ) where {T}
    if !isa(ctx_new.ip,JFM.Lagrange)
        throw(AssertionError("Nodal interpolation only defined for Lagrange elements"))
    end
    if isa(ctx_old.ip,JFM.Lagrange)
        u_node_or_cellvals = undoBCS(ctx_old, u,bdata)[ctx_old.node_to_dof]
    else
        u_node_or_cellvals = undoBCS(ctx_old, u,bdata)[ctx_old.cell_to_dof]
    end
    return nodal_interpolation(ctx_new,
                x -> evaluate_function_from_node_or_cellvals(ctx_old,u_node_or_cellvals,x;
                        outside_value=outside_value, project_in=project_in)
                    )
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
        project_in=true,
        outside_value=NaN
        ) where {T}

    if !isa(ctx_new.ip,JFM.Lagrange)
        throw(AssertionError("Nodal interpolation only defined for Lagrange elements"))
    end

    ncols = size(u)[2]
    u_node_or_cellvals = zeros(T, ctx_old.n,ncols)
    for j in 1:ncols
        if isa(ctx_old.ip, JFM.Lagrange)
            u_node_or_cellvals[:,j] = undoBCS(ctx_old,u[:,j],bdata)[ctx.node_to_dof]
        else
            u_node_or_cellvals[:,j] = undoBCS(ctx_old,u[:,j],bdata)[ctx.node_to_dof]
        end
    end

    #TODO: Maybe make this more efficient by calling evaluate_function_from_node_or_cellvals_multiple
    u_new::Array{T,2} = zeros(T,ctx_new.n,ncols)*NaN
    for i in 1:ctx_new.n
        for j in 1:ncols
            u_new[ctx_new.node_to_dof[i],j] = evaluate_function_from_node_or_cellvals(
                ctx_old,u_node_or_cellvals[:,j],
                ctx_new.grid.nodes[i].x, outside_value=outside_value,project_in=project_in
                )
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
    NumberedPoint2D(p::Vec{2,Float64}) = new(p[1], p[2], 0)
end
GP.Point(x::Real, y::Real, k::Int64) = NumberedPoint2D(x, y, k)
GP.Point2D(p::NumberedPoint2D) = Point2D(p.x,p.y)
GP.gety(p::NumberedPoint2D) = p.y
GP.getx(p::NumberedPoint2D) = p.x

#More or less equivalent to matlab's delaunay function, based on code from FEMDL.jl

function delaunay2(x::Vector{Vec{2,Float64}})
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


#See if this cell, or vertical and horizontal translations
#of it have already been added to the triangulation.
#This is a helper function
function in_cells_used(cells_used_arg,element::Tuple{Int,Int,Int},num_nodes_in)::Int
    function moveby(i::Int,xdir::Int,ydir::Int)::Int
        xpos = div((i-1),3*num_nodes_in)
        ypos = rem(div((i-1),num_nodes_in),3)
        remainder = rem((i-1),num_nodes_in)

        #In order to keep the original nodes in the first copy
        #of the tiling, we resorted to use using a "creative" ordering
        #of the indices. This comes back to bite us here.
        function addToPosition(pos,what)
            @assert what ∈ [0,-1,1] #TODO: get rid of this
            if what == 0
                return pos
            end
            if pos == 0 #Corresponds to 0 position
                if what == +1
                    return 1
                else
                    return 2
                end
            elseif pos == 1 #Corresponds to +1 position
                if what == -1
                    return 0
                else
                    return 1
                end
            elseif pos == 2#Corresponds to -1 position
                if what == +1
                    return 0
                else
                    return 2
                end
            else
                throw(AssertionError("Should never reach here, pos=$pos"))
            end
        end

        xpos = addToPosition(xpos,xdir)
        ypos = addToPosition(ypos,ydir)
        return remainder + 3*num_nodes_in*xpos + ypos*num_nodes_in + 1
    end
    for directionx in -1:1
        for directiony in -1:1
            #TODO: optimize this for speed
            moved_cell = Tuple{Int,Int,Int}(sort(collect(moveby.(element,directionx,directiony))))
            if moved_cell ∈ keys(cells_used_arg)
                result = cells_used_arg[moved_cell]
                return result
            end
        end
    end
    return 0
end


"""
Function for creating a grid from scattered nodes.
Calls VoronoiDelaunay for delaunay triangulation. Makes a periodic triangulation
if `on_torus` is set to `true`.
"""
function JuAFEM.generate_grid(::Type{JFM.Triangle},
     nodes_in::Vector{Vec{2,Float64}};
     on_torus=false,LL=nothing,UR=nothing
     )
    if on_torus
        @assert !(LL == nothing || UR == nothing)
    end

    #How many nodes were supplied
    num_nodes_in = length(nodes_in)

    #What nodes to triangulate. If we are working on the torus,
    #this includes copies of the nodes on the covering space.
    nodes_to_triangulate = Vector{Vec{2,Float64}}()

    #points_mapping[i] = j means that nodes_to_triangulate[i]
    #Should be identified with nodes_in[j]
    points_mapping = Vector{Int}()
    if on_torus
        dx = UR[1] - LL[1]
        dy = UR[2] - LL[2]
        for i in [0,1,-1]
            for j in [0,1,-1]
                for (index,node) in enumerate(nodes_in)
                    new_point = node .+ (i*dx,j*dy)
                    push!(nodes_to_triangulate,Vec{2}((new_point[1],new_point[2])))
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

    #Delaunay Triangulation
    tess, m, scale_x, scale_y, min_x, min_y = delaunay2(nodes_to_triangulate)

    nodes = map(JFM.Node, nodes_in)

    #Our grid will require some nodes twice.
    #additional_nodes[i] = j means that we require the node i
    #and that it has been added at index j to the grid's list of nodes
    additional_nodes = Dict{Int,Int}()


    #Cells that are part of the grid later
    cells = JFM.Triangle[]

    #Here we keep track of which cells we've already added
    #We assume that the mesh is fine enough that a cell is uniquely specified
    #by its vertices modulo periodicity, so the the tuple acting as a key is
    #assumed to be sorted.
    cells_used = Dict{Tuple{Int,Int,Int},Int}()
    #Our periodic tesselation produced many copies of the same cell
    #In order to do point location later, cell_number_table[i] = j means
    #that cell i in the periodic tesselation corresponds to cell j of the grid
    cell_number_table = zeros(Int, length(tess._trigs))
    #In order to build cell_number_table, we are interested in *all* cells
    #of the triangulation. But we only consider cells to be added to the grid
    #that have a node inside the original grid. cells_to_deal_with contains other
    #cells, and their indices in the tesselation
    cells_to_deal_with = Dict{Tuple{Int,Int,Int},Int}()

    #We may need to switch the position of two nodes that are the same (modulo periodicity)
    #if one of them gets added first
    switched_nodes_table = collect(1:length(nodes_to_triangulate))
    nodes_used = zeros(Bool, num_nodes_in)


    tri_iterator = Base.iterate(tess)
    while tri_iterator != nothing
        (tri,triindex) = tri_iterator
        #It could be the the triangle in question is oriented the wrong way
        #We test this, and flip it if neccessary
        J = Tensors.otimes((nodes_to_triangulate[switched_nodes_table[tri._b.id]] - nodes_to_triangulate[switched_nodes_table[tri._a.id]]), e1)
        J += Tensors.otimes((nodes_to_triangulate[switched_nodes_table[tri._c.id]] - nodes_to_triangulate[switched_nodes_table[tri._a.id]]), e2)
        detJ = det(J)
        @assert detJ != 0
        if detJ > 0
            new_tri_nodes_from_tess = (tri._a.id, tri._b.id, tri._c.id)
        else
            new_tri_nodes_from_tess = (tri._a.id, tri._c.id, tri._b.id)
        end

        if !on_torus
            push!(cells, JFM.Triangle(new_tri_nodes_from_tess))
            cell_number_table[triindex.ix-1] = length(cells)
        else
            #Get the nodes in question.
            tri_nodes = switched_nodes_table[collect(new_tri_nodes_from_tess)]

            #Are any of the vertices actually inside?
            if any(x -> x <= num_nodes_in, tri_nodes)
                #Let's find out if we've already added a (modulo periodicity)
                #version of this cell
                thiscell = in_cells_used(cells_used,
                                    Tuple{Int,Int,Int}(new_tri_nodes_from_tess),
                                    num_nodes_in)
                if thiscell != 0
                    #We have, so nothing to do here except record this.
                   cell_number_table[triindex.ix-1] = thiscell
                else
                    #Now iterate over the nodes of the cell
                    for (index,cur) in enumerate(tri_nodes)
                        #Is this one of the "original" nodes?
                        if cur <= num_nodes_in
                            #Let's record that we've seen it
                            nodes_used[cur] = true
                        else
                            #Have we never seen the corresponding original node?
                            original_node = points_mapping[cur]
                            if nodes_used[original_node] == false

                                #This node get  s to be the "original" node now
                                tmpNode = nodes[original_node].x
                                nodes[original_node] = JFM.Node(nodes_to_triangulate[cur])
                                nodes_to_triangulate[original_node] = nodes_to_triangulate[cur]
                                nodes_to_triangulate[cur] = tmpNode

                                switched_nodes_table[cur] = original_node
                                switched_nodes_table[original_node] = cur

                                tri_nodes[index] = original_node
                                nodes_used[original_node] = true

                            #Have we added this node already?
                            elseif cur ∈ keys(additional_nodes)
                                tri_nodes[index] = additional_nodes[cur]
                            else
                                #Need to add this node to the grid
                                push!(nodes, JFM.Node(nodes_to_triangulate[cur]))
                                additional_nodes[cur] = length(nodes)
                                tri_nodes[index] = length(nodes)
                            end
                        end
                    end
                    #Phew. We can now add the triangle to the triangulation.
                    new_tri = JFM.Triangle(Tuple{Int,Int,Int}(tri_nodes))
                    push!(cells, new_tri)

                    #We now remember that we've used this cell
                    cell_number_table[triindex.ix-1] = length(cells)
                    cells_used[ Tuple{Int,Int,Int}(sort(collect(new_tri_nodes_from_tess))) ] = length(cells)
                end
            else #Deal with this cell later
                cells_to_deal_with[Tuple{Int,Int,Int}(tri_nodes)] = triindex.ix-1
            end
        end
        tri_iterator = Base.iterate(tess,triindex)
    end
    #Write down location of remining cells
    if on_torus
        for (c,index) in cells_to_deal_with
            thiscell = in_cells_used(cells_used,c,num_nodes_in)
            cell_number_table[index] = thiscell
        end
    end

    #Check if everything worked out fine
    used_nodes = zeros(Bool, length(nodes))
    for c in cells
        used_nodes[collect(c.nodes)] .= true
    end
    if any(x -> x == false, used_nodes)
        @warn "Some nodes added that might cause problems with JuAFEM. Proceed at your own risk."
    end

    facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = JFM.Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    locator = delaunayCellLocator(m, scale_x, scale_y, min_x, min_y, tess,nodes_to_triangulate[switched_nodes_table],points_mapping,cell_number_table)
    return grid, locator
end

function JuAFEM.generate_grid(::Type{JFM.QuadraticTriangle}, nodes_in::Vector{Vec{2,Float64}})

    nodes_to_triangulate = Vector{Vec{2,Float64}}[]
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
    nodes = map(JFM.Node, nodes_in)
    n = length(nodes)
    ctr = n #As we add nodes (for edge vertices), increment the ctr...

    centerNodes = spzeros(n,n)
    cells = JFM.QuadraticTriangle[]
    for tri_id in 1:m
        tri = tess._trigs[locator.internal_triangles[tri_id]]

        #Create non-vertex nodes
        ab = centerNodes[tri._a.id, tri._b.id]
        if ab == 0
            ctr += 1
            ab = centerNodes[tri._a.id,tri._b.id] = centerNodes[tri._b.id,tri._a.id] =  ctr
            center = JFM.Node(0.5*(nodes[tri._b.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end
        ac = centerNodes[tri._a.id, tri._c.id]
        if ac == 0
            ctr += 1
            ac = centerNodes[tri._a.id,tri._c.id] = centerNodes[tri._c.id,tri._a.id] = ctr
            center = JFM.Node(0.5*(nodes[tri._c.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end

        bc = centerNodes[tri._c.id, tri._b.id]
        if bc == 0
            ctr += 1
            bc = centerNodes[tri._b.id,tri._c.id] = centerNodes[tri._c.id,tri._b.id] = ctr
            center = JFM.Node(0.5*(nodes[tri._c.id].x + nodes[tri._b.id].x))
            push!(nodes,center)
        end

        J = Tensors.otimes((nodes_in[tri._b.id] - nodes_in[tri._a.id]) , e1)
        J +=  Tensors.otimes((nodes_in[tri._c.id] - nodes_in[tri._a.id]) , e2)
        detJ = det(J)

        @assert det(J) != 0
        if detJ > 0
            new_tri = JFM.QuadraticTriangle((tri._a.id,tri._b.id,tri._c.id,ab,bc,ac))
        else
            new_tri = JFM.QuadraticTriangle((tri._a.id,tri._c.id,tri._b.id,ac,bc,ab))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = JFM.Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
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
function JuAFEM.generate_grid(::Type{JFM.QuadraticTetrahedron}, cells_per_dim::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
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
    nodes = Vector{JFM.Node{3,T}}(undef,total_nodes)
    cells = Vector{JFM.QuadraticTetrahedron}(undef,total_elements)

    # Generate nodes
    node_idx = 1
    @inbounds for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        nodes[node_idx] = JFM.Node((coords_x[i], coords_y[j], coords_z[k]))
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
            cells[cell_idx + idx] = JFM.QuadraticTetrahedron((cube[v1],cube[v2],cube[v3],cube[v4],
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

    boundary_matrix = JFM.boundaries_to_sparse([le; ri; bo; to; fr; ba])

    facesets = Dict(
        "left" => Set(le),
        "right" => Set(ri),
        "front" => Set(fr),
        "back" => Set(ba),
        "bottom" => Set(bo),
        "top" => Set(to),
    )
    return JFM.Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end
