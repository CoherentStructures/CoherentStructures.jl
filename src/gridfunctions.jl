#(c) 2017 Nathanael Schilling
#This file implements methods for working with Ferrite grids
#This includes methods for making grids from Delaunay Triangulations based
#on the code in FEMDL.jl

#Ferrite has no functions for determining which cell a point is in.
#Amongst other things, this file (and pointlocation.jl) implements an API for doing this.

const default_quadrature_order = 5
const default_quadrature_order3D = 2

"""
    mutable struct GridContext

Stores everything needed as "context" to be able to work on a FEM grid based on the `Ferrite` package.
Adds a point-locator API which facilitates plotting functions defined on the grid within Julia.

# Fields
- `grid::FEM.Grid`, `ip::FEM.Interpolation`, `ip_geom::FEM.Interpolation`, `qr::FEM.QuadratureRule`;
  see the [`Ferrite`](https://github.com/Ferrite-FEM/FEM.jl) package.
- `loc::PointLocator`: object used for point location on the grid.

- `node_to_dof::Vector{Int}`: lookup table for dof index of a node (for Lagrange elements)
- `dof_to_node::Vector{Int}`: inverse of node_to_dof

- `cell_to_dof::Vector{Int}`: lookup table for dof index of a cell (for piecewise constant elements)
- `dof_to_cell::Vector{Int}`: inverse of cell_to_dof

- `num_nodes`: number of nodes on the grid
- `num_cells`: number of elements (e.g. triangles,quadrilaterals, ...) on the grid
- `n`: number of degrees of freedom (== `num_nodes` for Lagrange Elements, and == `num_cells` for piecewise constant elements)

- `quadrature_points::Vector{Vec{dim,Float64}}`: lists of all quadrature points on the grid, in a specific order.
- `mass_weights::Vector{Float64}`: weighting for stiffness/mass matrices.

- `spatialBounds`: if available, the corners of a bounding box of a domain. For regular grids, the bounds are tight.
- `numberOfPointsInEachDirection`: for regular grids, how many (non-interior) nodes make up the regular grid.
- `gridType`: a string describing what kind of grid this is (e.g. "regular triangular grid")
- `no_precompute=false`: whether to precompute objects like quadrature points. Only enable this if you know what you are doing.
"""
mutable struct GridContext{dim,G<:FEM.Grid,ITP<:FEM.Interpolation,ITPG<:FEM.Interpolation,DH<:FEM.DofHandler,QR<:FEM.QuadratureRule,PL<:PointLocator} <: AbstractGridContext{dim} #TODO: Currently set as mutable, is this sensible?
    grid::G
    ip::ITP
    ip_geom::ITPG
    dh::DH
    qr::QR
    loc::PL

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
    spatialBounds::NTuple{2,NTuple{dim,Float64}} #This is {LL,UR} for regular grids
    #This is the number of (non-interior) nodes in each direction (not points)
    numberOfPointsInEachDirection::Vector{Int}

    gridType::String

    function GridContext{dim}(
                grid::FEM.Grid,
                ip::FEM.Interpolation,
                ip_geom::FEM.Interpolation,
                dh::FEM.DofHandler,
                qr::FEM.QuadratureRule,
                loc::PointLocator;
                no_precompute=false
            ) where {dim}

        x = new{dim,typeof(grid),typeof(ip),typeof(ip_geom),typeof(dh),typeof(qr),typeof(loc)}(grid, ip, ip_geom, dh, qr, loc)
        x.num_nodes = FEM.getnnodes(dh.grid)
        x.num_cells = FEM.getncells(dh.grid)
        x.n = FEM.ndofs(dh)

        #TODO: Measure if the sorting below is expensive
        if ip isa FEM.Lagrange
            x.node_to_dof = nodeToDHTable(x)
            x.dof_to_node = sortperm(x.node_to_dof)
        elseif ip isa FEM.PiecewiseConstant
            x.cell_to_dof = cellToDHTable(x)
            x.dof_to_cell = sortperm(x.cell_to_dof)
        else
            throw(AssertionError("Unknown interpolation type"))
        end
        if no_precompute
            x.quadrature_points = Vec{dim,Float64}[]
        else
            x.quadrature_points = getQuadPoints(x)
        end
        x.mass_weights = ones(length(x.quadrature_points))
        return x
    end
end

#Based on Ferrite's WriteVTK.vtk_point_data
function nodeToDHTable(ctx::AbstractGridContext{dim}) where {dim}
    dh = ctx.dh
    n = ctx.n
    res = Vector{Int}(undef,n)
    for cell in FEM.CellIterator(dh)
        _celldofs = FEM.celldofs(cell)
        ctr = 1
        offset = FEM.field_offset(dh, dh.field_names[1])
        for node in FEM.getnodes(cell)
            res[node] = _celldofs[ctr + offset]
            ctr += 1
        end
    end
    return res
end

function cellToDHTable(ctx::AbstractGridContext{dim}) where {dim}
    dh = ctx.dh
    n = ctx.n
    res = Vector{Int}(undef,n)
    for (cellindex,cell) in enumerate(FEM.CellIterator(dh))
        _celldofs = FEM.celldofs(cell)
        offset = FEM.field_offset(dh, dh.field_names[1])
        for node in FEM.getnodes(cell)
            mynode = node
        end
        res[cellindex] = _celldofs[offset+1]
    end
    return res
end
#=
"""
    GridContext{1}(FEM.Line, [numnodes, LL, UR; ip,quadrature_order,ip])

Constructor for a 1d regular mesh with `numnodes[1]` node on the interval `[LL[1],UR[1]]`.
The default for `ip` is P1-Lagrange elements, but piecewise-constant elements can also be used.
"""
=#
function GridContext{1}(::Type{FEM.Line},
                         numnodes::Tuple{Int}=(25,), LL::Tuple{<:Real}=(0.0,), UR::Tuple{<:Real}=(1.0,);
                         quadrature_order::Int=default_quadrature_order,
                         ip=FEM.Lagrange{1,FEM.RefCube,1}(), kwargs...)
    # The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.Line, (numnodes[1]-1,), Vec{1}(LL), Vec{1}(UR))
    loc = Regular1dGridLocator{FEM.Line}(numnodes[1], Vec{1}(LL), Vec{1}(UR))

    dh = FEM.DofHandler(grid)
    push!(dh, :T, 1, ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)

    qr = FEM.QuadratureRule{1, FEM.RefCube}(quadrature_order)
    result = GridContext{1}(grid, ip,FEM.Lagrange{1,FEM.RefCube,1}(), dh, qr, loc; kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1]]
    if ip isa FEM.Lagrange
        result.gridType = "regular P1 1d grid"
    else
        result.gridType = "regular PC 1d grid"
    end

    return result
end

#=
"""
    GridContext{1}(FEM.QuadraticLine, numnodes, LL, UR, quadrature_order)
"""
=#
function GridContext{1}(::Type{FEM.QuadraticLine},
                         numnodes::Tuple{Int}=(25,),
                         LL::Tuple{<:Real}=(0.0,),
                         UR::Tuple{<:Real}=(1.0,);
                         quadrature_order::Int=default_quadrature_order,
                         ip=FEM.Lagrange{1,FEM.RefCube,2}(), kwargs...
                         )
    # The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.QuadraticLine, (numnodes[1]-1,), Vec{1}(LL), Vec{1}(UR))
    loc = Regular1dGridLocator{FEM.QuadraticLine}(numnodes[1], Vec{1}(LL), Vec{1}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{1, FEM.RefCube}(quadrature_order)
    push!(dh, :T, 1) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result = GridContext{1}(grid, ip, FEM.Lagrange{1,FEM.RefCube,2}(), dh, qr, loc; kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1]]
    if !isa(ip, FEM.Lagrange)
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
    regular1dPCGrid(numnodes, left=0.0, right=1.0; [quadrature_order])

Create a regular grid with `numnodes` nodes on the interval `[left, right]` in 1d, with
piecewise-constant basis functions.
"""
function regular1dPCGrid(numnodes::Int, args...; kwargs...)
    return regular1dPCGrid((numnodes,), args...; kwargs...)
end
function regular1dPCGrid(numnodes::Tuple{Int}, left::Real=0.0, right::Real=1.0;
                    quadrature_order::Int=default_quadrature_order, kwargs...)
    result = GridContext{1}(FEM.Line, numnodes, (left,), (right,);
                            quadrature_order=quadrature_order,
                            ip=FEM.PiecewiseConstant{1,FEM.RefCube,1}(),
                            kwargs...)
    return result, BoundaryData()
end

"""
    regular1dP1Grid(numnodes, left=0.0, right=1.0; [quadrature_order])

Create a regular grid with `numnodes` nodes on the interval `[left, right]` in 1d, with
P1-Lagrange basis functions.
"""
function regular1dP1Grid(numnodes::Int, args...; kwargs...)
    return regular1dP1Grid((numnodes,), args...; kwargs...)
end
function regular1dP1Grid(numnodes::Tuple{Int}, left::Real=0.0, right::Real=1.0;
                    quadrature_order::Int=default_quadrature_order, kwargs...)
    result = GridContext{1}(FEM.Line, numnodes, (left,), (right,);
                            quadrature_order=quadrature_order, kwargs...)
    return result, BoundaryData()
end

"""
    regular1dP2Grid(numnodes, left=0.0, right=1.0; [quadrature_order])

Create a regular grid with `numnodes` non-interior nodes on the interval `[left, right]`,
with P2-Lagrange elements.
"""
function regular1dP2Grid(numnodes::Int, args...; kwargs...)
    return regular1dP2Grid((numnodes,), args...; kwargs...)
end
function regular1dP2Grid(numnodes::Tuple{Int}, left=0.0, right=1.0;
                            quadrature_order::Int=default_quadrature_order, kwargs...)
    result = GridContext{1}(FEM.QuadraticLine, numnodes, (left,), (right,);
                                quadrature_order=quadrature_order, kwargs...)
    return result, BoundaryData()
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

# TODO: unclear what this function is good for... completely type unstable!
"""
    regular2dGrid(gridType, numnodes, LL=(0.0,0.0), UR=(1.0,1.0); quadrature_order=default_quadrature_order)

Constructs a regular grid. `gridType` should be one from `regular2dGridTypes`.
"""
function regular2dGrid(gridType::String, numnodes::NTuple{2,Int},
        LL::NTuple{2,<:Real}=(0.0, 0.0), UR::NTuple{2,<:Real}=(1.0, 1.0); kwargs...)
    if gridType == "regular PC triangular grid"
        return regularTriangularGrid(numnodes, LL, UR; PC=true, kwargs...)
    elseif gridType == "regular P1 triangular grid"
        return regularTriangularGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular PC Delaunay grid"
        return regularDelaunayGrid(numnodes, LL, UR; PC=true, kwargs...)
    elseif gridType == "regular P1 Delaunay grid"
        return regularDelaunayGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 triangular grid"
        return regularP2TriangularGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 Delaunay grid"
        return regularP2DelaunayGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular PC quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL, UR; PC=true, kwargs...)
    elseif gridType == "regular P1 quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL, UR; kwargs...)
    elseif gridType == "regular P2 quadrilateral grid"
        return regularP2QuadrilateralGrid(numnodes, LL, UR; kwargs...)
    else
        fail("Unsupported grid type '$gridType'")
    end
end

@deprecate regular2dGrid(gridType::String, numnodes::NTuple{2,Int}, LL::Vector{<:Real}, UR::Vector{<:Real}; kwargs...) regular2dGrid(gridType, numnodes, (LL...,), (UR...,); kwargs...)

#= #TODO 1.0
"""
    GridContext{2}(FEM.Triangle, node_list; [on_torus=false,on_cylinder=false,LL,UR,quadrature_order=default_quadrature_order,ip])

Create a P1-Lagrange grid based on Delaunay Triangulation.
If `on_torus==true`, triangulates on a periodic domain (in both directions)
If `on_cylinder==true`, triangulates on a periodic domain in x-direction
defined by `LL` (lower-left corner) and `UR` (upper-right corner).
The parameter `ip` defines what kind of shape functions to use, the default is P1-Lagrange (can also be piecewise constant).
i
Uses `DelaunayVoronoi.jl` internally.
"""
=#
function GridContext{2}(
            ::Type{FEM.Triangle},
            node_list::Vector{Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order,
            on_torus=false,
            on_cylinder=false,
            LL=nothing,
            UR=nothing,
            ip=FEM.Lagrange{2,FEM.RefTetrahedron,1}(),kwargs...
            )
    if on_torus || on_cylinder
        @assert !(LL === nothing || UR === nothing)
    end
    grid, loc = FEM.generate_grid(FEM.Triangle, node_list;
                    on_torus=on_torus, on_cylinder=on_cylinder, LL=LL, UR=UR)
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{2, FEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1, ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result =  GridContext{2}(grid, ip, FEM.Lagrange{2,FEM.RefTetrahedron,1}(), dh, qr, loc; kwargs...)
    if ip isa FEM.Lagrange
        result.gridType = "irregular P1 Delaunay grid"
    else
        result.gridType = "irregular PC Delaunay grid"
    end
    if LL === nothing
        LL = map(i -> minimum([x[i] for x in node_list]), (1, 2))
    end
    if UR === nothing
        UR = map(i -> maximum([x[i] for x in node_list]), (1, 2))
    end
    result.spatialBounds = (LL, UR)
    return result
end

"""
    irregularDelaunayGrid(nodes_in; [on_torus=false, on_cylinder=false, LL, UR, PC=false, ...])

Triangulate the nodes `nodes_in` and return a `GridContext` and `bdata` for them.
If `on_torus==true`, the triangulation is done on a torus.
If `PC==true`, return a mesh with piecewise constant shape-functions, else P1 Lagrange.
"""
function irregularDelaunayGrid(nodes_in::AbstractVector{Vec{2,Float64}};
        on_torus=false,
        on_cylinder=false,
        LL=nothing,
        UR=nothing,
        PC=false,
        kwargs...)
    if on_torus || on_cylinder
        @assert (LL !== nothing && UR !== nothing)
    end
    if !PC
        ip = FEM.Lagrange{2,FEM.RefTetrahedron,1}()
    else
        ip = FEM.PiecewiseConstant{2,FEM.RefTetrahedron,1}()
    end
    ctx = GridContext{2}(FEM.Triangle, nodes_in;
            on_torus=on_torus, on_cylinder=on_cylinder, LL=LL, UR=UR, ip=ip, kwargs...)
    if on_torus
        bdata = BoundaryData(ctx, Dists.PeriodicEuclidean([UR .- LL...;]))
    elseif on_cylinder
        bdata = BoundaryData(ctx, Dists.PeriodicEuclidean([UR[1] - LL[1], Inf]))
    else
        bdata = BoundaryData()
    end
    return ctx, bdata
end
irregularDelaunayGrid(nodes_in; kwargs...) =
    irregularDelaunayGrid(Vec{2}.(nodes_in); kwargs...)

"""
    randomDelaunayGrid(npoints, LL, UR; kwargs...)

Create a delaunay grid in 2d from `npoints` random points on the box with lower
left corner `LL` and upper right corner `UR`.
Extra keyword arguments are passed to `irregularDelaunayGrid`.
"""
function randomDelaunayGrid(npoints::Int, LL::NTuple{2,<:Real}=(0.0,0.0), UR::NTuple{2,<:Real}=(1.0,1.0);
                            kwargs...)
    width = UR .- LL
    nodes_in = map(_ -> Vec{2}((rand()*width[1] + LL[1], rand()*width[2] + LL[2])), 1:npoints)
    return irregularDelaunayGrid(nodes_in; LL=LL, UR=UR, kwargs...)
end

# deprecation doesn't work somehow
# @deprecate randomDelaunayGrid(npoints::Int; LL::AbstractVector{<:Real}=[0.0,0.0], UR::AbstractVector{<:Real}=[1.0,1.0], kwargs...) randomDelaunayGrid(npoints, (LL...,), (UR...,); kwargs...)

"""
    regularDelaunayGrid(numnodes=(25,25), LL=(0.0,0.0), UR=(1.0,1.0); [quadrature_order, on_torus=false, on_cylinder=false, nudge_epsilon=1e-5, PC=false])

Create a regular grid on a square with lower left corner `LL` and upper-right corner `UR`.
Uses Delaunay triangulation internally.
If `on_torus==true`, uses a periodic Delaunay triangulation in both directions.
If `on_cylinder==true` uses a periodic Delaunay triangulatin in x direction only.
 To avoid degenerate special cases,
all nodes are given a random `nudge`, the strength of which depends on `numnodes` and `nudge_epsilon`.
If `PC==true`, returns a piecewise constant grid. Else returns a P1-Lagrange grid.
"""
function regularDelaunayGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::NTuple{2,<:Real}=(0.0, 0.0),
            UR::NTuple{2,<:Real}=(1.0, 1.0);
            quadrature_order::Int=default_quadrature_order,
            on_torus::Bool=false,
            on_cylinder::Bool=false,
            nudge_epsilon::Real=1e-5,
            PC::Bool=false,kwargs...
        )
    X = range(LL[1], stop=UR[1], length=numnodes[1])
    Y = range(LL[2], stop=UR[2], length=numnodes[2])
    if PC
        ip=FEM.PiecewiseConstant{2,FEM.RefTetrahedron,1}()
    else
        ip=FEM.Lagrange{2,FEM.RefTetrahedron,1}()
    end
    node_list = vec([Vec{2}((x, y)) for y in Y, x in X])
    if on_torus || on_cylinder
        function nudge(point)
            nudgefactor = (UR .- LL)  .* nudge_epsilon ./ numnodes
            return Vec{2}(
                max.(LL .+ 0.1*nudgefactor, min.(UR .- 0.1*nudgefactor,
                    point .+ nudgefactor .* rand(2)
                    )))
        end
        node_list = nudge.(filter(x -> minimum(abs.(x .- UR)) > 1e-8, node_list))
    end
    result = GridContext{2}(FEM.Triangle, node_list;
         quadrature_order=quadrature_order,
         on_torus=on_torus,
         on_cylinder=on_cylinder,
         LL=LL,
         UR=UR,
         ip=ip, kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    if !PC
        result.gridType = "regular P1 Delaunay grid"
    else
        result.gridType = "regular PC Delaunay grid"
    end
    if !PC
        bdata = BoundaryData(result, Dists.PeriodicEuclidean([(UR .- LL)...]))
    else
        bdata = BoundaryData()
    end
    return result, bdata
end


#= TODO 1.0
"""
    GridContext{2}(FEM.QuadraticTriangle, node_list, quadrature_order=default_quadrature_order)

Create a P2 grid given a set of (non-interior) nodes using Delaunay Triangulation.
"""
=#
function GridContext{2}(
            ::Type{FEM.QuadraticTriangle},
            node_list::Vector{Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order,
            ip=FEM.Lagrange{2,FEM.RefTetrahedron,2}(),kwargs...
            )
    grid, loc = FEM.generate_grid(FEM.QuadraticTriangle, node_list)
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{2, FEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result = GridContext{2}(grid, ip,FEM.Lagrange{2,FEM.RefTetrahedron,2}(), dh, qr, loc;kwargs...)
    result.gridType = "irregular P2 Delaunay grid"
    return result
end

"""
    regularP2DelaunayGrid(numnodes=(25,25), LL=(0.0,0.0), UR=(1.0,1.0), quadrature_order=default_quadrature_order)

Create a regular P2 triangular grid with `numnodes` being the number of (non-interior) nodes in each direction.
"""
function regularP2DelaunayGrid(
            numnodes::Tuple{Int,Int}=(25, 25),
            LL::NTuple{2,<:Real}=(0.0, 0.0),
            UR::NTuple{2,<:Real}=(1.0, 1.0);
            quadrature_order::Int=default_quadrature_order, kwargs...)
    X = range(LL[1], stop=UR[1], length=numnodes[1])
    Y = range(LL[2], stop=UR[2], length=numnodes[2])
    node_list = vec([Vec{2}((x, y)) for y in Y, x in X])
    result = GridContext{2}(FEM.QuadraticTriangle, node_list, quadrature_order=quadrature_order;kwargs...)
    #TODO: Think about what values would be sensible for the two variables below
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 Delaunay grid"
    return result, BoundaryData()
end

#=
"""
    GridContext{2}(FEM.Triangle, numnodes, LL,UR; [quadrature_order])

Create a regular triangular grid. Does not use Delaunay triangulation internally.
"""
=#

function GridContext{2}(::Type{FEM.Triangle},
                         numnodes::Tuple{Int,Int}=(25, 25),
                         LL::NTuple{2,<:Real}=(0.0, 0.0),
                         UR::NTuple{2,<:Real}=(1.0, 1.0);
                         quadrature_order::Int=default_quadrature_order,
                         ip=FEM.Lagrange{2,FEM.RefTetrahedron,1}(),kwargs...
                         )
    # The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.Triangle, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = Regular2DGridLocator{FEM.Triangle}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{2, FEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1, ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result = GridContext{2}(grid, ip,FEM.Lagrange{2,FEM.RefTetrahedron,1}(), dh, qr, loc;kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular triangular grid"
    return result
end

"""
    regularTriangularGrid(numnodes, LL,UR;[quadrature_order, PC=false])

Create a regular triangular grid on a rectangle; does not use Delaunay triangulation internally.
If
"""
function regularTriangularGrid(numnodes::Tuple{Int,Int}=(25,25),
                                LL::NTuple{2,<:Real}=(0.0,0.0),
                                UR::NTuple{2,<:Real}=(1.0,1.0);
                                quadrature_order::Int=default_quadrature_order,
                                PC=false, kwargs...)
    if PC == false
        ip = FEM.Lagrange{2,FEM.RefTetrahedron,1}()
    else
        ip = FEM.PiecewiseConstant{2,FEM.RefTetrahedron,1}()
    end
    ctx = GridContext{2}(FEM.Triangle, numnodes, LL, UR;
            quadrature_order=quadrature_order,ip=ip,kwargs...
            )
    return ctx, BoundaryData()
end

@deprecate regularTriangularGrid(numnodes::Tuple{Int,Int}, LL::AbstractVector{<:Real}, UR::AbstractVector{<:Real}; kwargs...) regularTriangularGrid(numnodes, (LL...,), (UR...,); kwargs...)

#= TODO 1.0
"""
    GridContext{2}(FEM.QuadraticTriangle, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 triangular grids. Does not use Delaunay triangulation internally.
"""
=#
function GridContext{2}(::Type{FEM.QuadraticTriangle},
                         numnodes::Tuple{Int,Int}=(25, 25),
                         LL::NTuple{2,<:Real}=(0.0,0.0),
                         UR::NTuple{2,<:Real}=(1.0,1.0);
                         quadrature_order::Int=default_quadrature_order,
                         ip=FEM.Lagrange{2,FEM.RefTetrahedron,2}(), kwargs...
                         )
    if !isa(ip,FEM.Lagrange)
        @warn "Using non-Lagrange interpolation with P2 elements may or may not work"
    end
    #The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.QuadraticTriangle, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = Regular2DGridLocator{FEM.QuadraticTriangle}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{2, FEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result =  GridContext{2}(grid, ip,FEM.Lagrange{2,FEM.RefTetrahedron,2}(), dh, qr, loc;kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 triangular grid"
    return result
end


"""
    regularP2TriangularGrid(numnodes=(25,25), LL=(0.0,0.0), UR=(1.0,1.0), quadrature_order=default_quadrature_order)

Create a regular P2 triangular grid on a rectangle. Does not use Delaunay triangulation internally.
"""
function regularP2TriangularGrid(numnodes::Tuple{Int,Int}=(25, 25),
                                    LL::NTuple{2,<:Real}=(0.0,0.0),
                                    UR::NTuple{2,<:Real}=(1.0,1.0);
                                    quadrature_order::Int=default_quadrature_order, kwargs...)
    ctx = GridContext{2}(FEM.QuadraticTriangle, numnodes, LL, UR, quadrature_order=quadrature_order;kwargs...)
    return ctx, BoundaryData()
end

@deprecate regularP2TriangularGrid(numnodes::Tuple{Int,Int}, LL::AbstractVector{<:Real}, UR::AbstractVector{<:Real};
                                    kwargs...) regularP2TriangularGrid(numnodes, (LL...,), (UR...,); kwargs...)
#= TODO 1.0
"""
    GridContext{2}(FEM.Quadrilateral, numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0], quadrature_order=default_quadrature_order)

Constructor for regular P1 quadrilateral grids.
"""
=#
function GridContext{2}(::Type{FEM.Quadrilateral},
                        numnodes::Tuple{Int,Int}=(25,25),
                        LL::NTuple{2,<:Real}=(0.0,0.0),
                        UR::NTuple{2,<:Real}=(1.0,1.0);
                        quadrature_order::Int=default_quadrature_order,
                        ip=FEM.Lagrange{2, FEM.RefCube, 1}(), kwargs...)
    #The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.Quadrilateral, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = Regular2DGridLocator{FEM.Quadrilateral}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{2, FEM.RefCube}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result =  GridContext{2}(grid, ip,FEM.Lagrange{2,FEM.RefCube,1}(), dh, qr, loc; kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    if ip isa FEM.Lagrange
        result.gridType = "regular P1 quadrilateral grid"
    else
        result.gridType = "regular PC quadrilateral grid"
    end
    return result
end

"""
    regularP2QuadrilateralGrid(numnodes, LL, UR; [quadrature_order, PC=false])

Create a regular P1 quadrilateral grid on a rectangle. If `PC==true`, use
piecewise constant shape functions, otherwise use P1 Lagrange.
"""
function regularQuadrilateralGrid(numnodes::Tuple{Int,Int}=(25, 25),
                                    LL::NTuple{2,<:Real}=(0.0,0.0),
                                    UR::NTuple{2,<:Real}=(1.0,1.0);
                                    quadrature_order::Int=default_quadrature_order,
                                    PC=false, kwargs...)
    if !PC
        ip = FEM.Lagrange{2,FEM.RefCube,1}()
    else
        ip = FEM.PiecewiseConstant{2,FEM.RefCube,1}()
    end
    ctx = GridContext{2}(FEM.Quadrilateral,
         numnodes, LL, UR;
         quadrature_order=quadrature_order,
         ip=ip,kwargs...
     )
     return ctx, BoundaryData()
end


#= TODO 1.0
"""
    GridContext{2}(FEM.QuadraticQuadrilateral, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 quadrilateral grids.
"""
=#
function GridContext{2}(::Type{FEM.QuadraticQuadrilateral},
                        numnodes::Tuple{Int,Int}=(25, 25),
                        LL::NTuple{2,<:Real}=(0.0,0.0),
                        UR::NTuple{2,<:Real}=(1.0,1.0);
                        quadrature_order::Int=default_quadrature_order,
                        ip=FEM.Lagrange{2, FEM.RefCube, 2}(), kwargs...)
    if !isa(ip, FEM.Lagrange)
        @warn "Non-Lagrange interpolation with P2 elements may or may not work"
    end
    #The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.QuadraticQuadrilateral, (numnodes[1]-1,numnodes[2]-1), Vec{2}(LL), Vec{2}(UR))
    loc = Regular2DGridLocator{FEM.QuadraticQuadrilateral}(numnodes[1], numnodes[2], Vec{2}(LL), Vec{2}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{2, FEM.RefCube}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result =  GridContext{2}(grid, ip,FEM.Lagrange{2,FEM.RefCube,2}(), dh, qr, loc;kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2]]
    result.gridType = "regular P2 quadrilateral grid"
    return result
end


"""
    regularP2QuadrilateralGrid(numnodes=(25,25), LL=(0.0,0.0), UR=(1.0,1.0), quadrature_order=default_quadrature_order)

Create a regular P2 quadrilateral grid on a rectangle.
"""
function regularP2QuadrilateralGrid(numnodes::Tuple{Int,Int}=(25,25),
                                    LL::NTuple{2,<:Real}=(0.0,0.0),
                                    UR::NTuple{2,<:Real}=(1.0,1.0);
                                    quadrature_order::Int=default_quadrature_order, kwargs...)
    ctx = GridContext{2}(FEM.QuadraticQuadrilateral, numnodes, LL, UR, quadrature_order=quadrature_order,kwargs...)
    return ctx, BoundaryData()
end

#=TODO 1.0
"""
    GridContext{3}(FEM.Tetrahedron, numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P1 Tetrahedral Grid in 3D.
"""
=#
function GridContext{3}(::Type{FEM.Tetrahedron},
                         numnodes::NTuple{3,Int}=(10,10,10),
                         LL::NTuple{3,<:Real}=(0.0,0.0,0.0),
                         UR::NTuple{3,<:Real}=(1.0,1.0,1.0);
                         quadrature_order::Int=default_quadrature_order3D,
                         ip=FEM.Lagrange{3, FEM.RefTetrahedron, 1}(), kwargs...)
    #The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.Tetrahedron, (numnodes[1]-1, numnodes[2]-1, numnodes[3] -1), Vec{3}(LL), Vec{3}(UR))
    loc = Regular3DGridLocator{FEM.Tetrahedron}(numnodes[1], numnodes[2], numnodes[3], Vec{3}(LL), Vec{3}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{3, FEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result =  GridContext{3}(grid, ip,FEM.Lagrange{3,FEM.RefTetrahedron,1}(), dh, qr, loc;kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2], numnodes[3]]
    if ip isa FEM.Lagrange
        result.gridType = "3D P1 regular tetrahedral grid"
    else
       result.gridType = "3D PC regular tetrahedral grid"
    end
    return result
end


#=TODO 1.0
"""
    GridContext{3}(FEM.QuadraticTetrahedron, numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)

Create a regular P2 Tetrahedral Grid in 3D.
"""
=#
function GridContext{3}(::Type{FEM.QuadraticTetrahedron},
                         numnodes::NTuple{3,Int}=(10,10,10),
                         LL::NTuple{3,<:Real}=(0.0,0.0,0.0),
                         UR::NTuple{3,<:Real}=(1.0,1.0,1.0);
                         quadrature_order::Int=default_quadrature_order3D,
                         ip=FEM.Lagrange{3, FEM.RefTetrahedron, 2}(), kwargs...)
    if !isa(ip, FEM.Lagrange)
        @warn "Using non-Lagrange interpolation with P2 Elements may or may not work"
    end
    #The -1 below is needed because Ferrite internally then goes on to increment it
    grid = FEM.generate_grid(FEM.QuadraticTetrahedron, (numnodes[1]-1, numnodes[2]-1, numnodes[3] -1), Vec{3}(LL), Vec{3}(UR))
    loc = Regular3DGridLocator{FEM.QuadraticTetrahedron}(numnodes[1], numnodes[2], numnodes[3], Vec{3}(LL), Vec{3}(UR))
    dh = FEM.DofHandler(grid)
    qr = FEM.QuadratureRule{3, FEM.RefTetrahedron}(quadrature_order)
    push!(dh, :T, 1,ip) #The :T is just a generic name for the scalar field
    FEM.close!(dh)
    result =  GridContext{3}(grid, ip,FEM.Lagrange{3,FEM.RefTetrahedron,2}(), dh, qr, loc;kwargs...)
    result.spatialBounds = (LL, UR)
    result.numberOfPointsInEachDirection = [numnodes[1], numnodes[2], numnodes[3]]
    result.gridType = "3D regular P2 tetrahedral grid"
    return result
end

"""
    regularTetrahedralGrid(numnodes=(10,10,10), LL=(0.0,0.0,0.0), UR=(1.0,1.0,1.0), quadrature_order=default_quadrature_order3D)

Create a regular P1 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularTetrahedralGrid(
        numnodes::NTuple{3,Int}=(10,10,10),
        LL::NTuple{3,<:Real}=(0.0,0.0,0.0),
        UR::NTuple{3,<:Real}=(1.0,1.0,1.0);
        quadrature_order::Int=default_quadrature_order3D, PC=false, kwargs...)
    if !PC
        ip = FEM.Lagrange{3,FEM.RefTetrahedron,1}()
    else
        ip = FEM.PiecewiseConstant{3,FEM.RefTetrahedron,1}()
    end
    ctx =  GridContext{3}(FEM.Tetrahedron,
        numnodes, LL, UR;
        quadrature_order=quadrature_order,ip=ip, kwargs...
        )
    return ctx, BoundaryData()
end

"""
    regularP2TetrahedralGrid(numnodes=(10,10,10), LL=(0.0,0.0,0.0), UR=(1.0,1.0,1.0), quadrature_order=default_quadrature_order3D)

Create a regular P2 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularP2TetrahedralGrid(numnodes::NTuple{3,Int}=(10,10,10),
                                    LL::NTuple{3,<:Real}=(0.0,0.0,0.0),
                                    UR::NTuple{3,<:Real}=(1.0,1.0,1.0);
                                    quadrature_order::Int=default_quadrature_order3D,
                                    kwargs...)
    ctx = GridContext{3}(FEM.QuadraticTetrahedron,
        numnodes, LL, UR;
        quadrature_order=quadrature_order, kwargs...)
    return ctx, BoundaryData()
end


"""
    project_in_xin(ctx,x_in,project_in)

Converts `x_in` to `Vec{dim}`, possibly taking pointwise maxima/minima to make
sure it is within `ctx.spatialBounds` (if `project_in==true`).
Helper function.
"""
function project_in_xin(
    ctx::GridContext{dim}, x_in::AbstractVector{T}, project_in
    )::Vec{dim,T} where {dim,T}

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
    evaluate_function_from_node_or_cellvals(ctx, vals, x_in; outside_value=0, project_in=false)

Like `evaluate_function_from_dofvals`, but the coefficients from `vals` are
assumed to be in node order. This is more efficient than
`evaluate_function_from_dofvals`.
"""
function evaluate_function_from_node_or_cellvals(
    ctx::GridContext{dim}, vals::AbstractVector{S}, x_in::Vec{dim,W};
    outside_value=NaN, project_in=false, throw_errors=true)::S where {dim,S,W}

    x::Vec{dim,W} = project_in_xin(ctx, x_in, project_in)

    @assert length(vals) == ctx.n

    local_coordinates::Vec{dim,W}, nodes::Vector{Int}, cellid::Int = try
         locatePoint(ctx, x)
    catch y
        if y isa DomainError
            return outside_value
        end
        if throw_errors
            print("Unexpected error for $x")
            rethrow(y)
        else
            return outside_value
        end
    end

    result::S = zero(S)

    if ctx.ip isa FEM.Lagrange
        for (j, nodeid) in enumerate(nodes)
            val = FEM.value(ctx.ip, j, local_coordinates)
            result += vals[nodeid]*val
        end
    elseif ctx.ip isa FEM.PiecewiseConstant
        val = FEM.value(ctx.ip, 1, local_coordinates)
        result += vals[cellid]*val
    else
        throw(AssertionError("Unknown interpolation"))
    end
    return result
end

"""
    evaluate_function_from_node_or_cellvalsFDiff

Like `evaluate_function_from_node_or_cellvals` but with more type-restrictions and ForwardDiff compatibility.
"""
function evaluate_function_from_node_or_cellvalsFDiff(
    ctx::GridContext{dim}, vals::Vector{Float64}, x_in::Vec{dim,W};
    outside_value=NaN, project_in=false, throw_errors=true)::W where {dim,W}

    x::Vec{dim,W} = project_in_xin(ctx,x_in,project_in)

    @assert length(vals) == ctx.n

    local_coordinates::Vec{dim,W}, nodes::Vector{Int}, cellid::Int = try
         locatePoint(ctx, x)
    catch y
        if y isa DomainError
            return outside_value
        end
        if throw_errors
            print("Unexpected error for $x")
            rethrow(y)
        else
            return outside_value
        end
    end

    result::W = zero(W)

    if ctx.ip isa FEM.Lagrange
        for (j, nodeid) in enumerate(nodes)
            val::W = FEM.value(ctx.ip, j, local_coordinates)
            result += vals[nodeid]*val
        end
    elseif ctx.ip isa FEM.PiecewiseConstant
        val = FEM.value(ctx.ip, 1, local_coordinates)
        result += vals[cellid]*val
    else
        throw(AssertionError("Unknown interpolation"))
    end
    return result
end

"""
    evaluate_function_from_dofvals(ctx, dofvals, x_in; outside_value=NaN,project_in=false)

Evaluate the function at point `x_in` with coefficients of dofs given by `dofvals` (in dof-order).
Return `outside_value` if point is out of bounds.
Project the point into the domain if `project_in==true`.
For evaluation at many points, or for many dofvals, the function `evaluate_function_from_dofvals_multiple`
is more efficient.
"""
function evaluate_function_from_dofvals(
    ctx::GridContext{dim}, vals::AbstractVector{S}, x_in::Vec{dim,W};
    outside_value=NaN, project_in=false)::S where {dim,S,W}

    @assert ctx.n == length(vals)

    if ctx.ip isa FEM.Lagrange
        vals_reorder = vals[ctx.node_to_dof]
    else
        vals_reorder = vals[ctx.cell_to_dof]
    end
    return evaluate_function_from_node_or_cellvals(
            ctx, vals_reorder, x_in;
            outside_value=outside_value,
            project_in=project_in
            )
end


"""
    evaluate_function_from_node_or_cellvals_multiple(ctx, vals, xin; is_diag=false, kwargs...)

Like `evaluate_function_from_dofvals_multiple` but uses node- (or cell- if piecewise constant interpolation)
ordering for `vals`, which makes it slightly more efficient.
If vals is a diagonal matrix, set `is_diag` to `true` for much faster evaluation.
"""
function evaluate_function_from_node_or_cellvals_multiple(
            ctx::GridContext{dim}, vals::AbstractMatrix{S},
            x_in::AbstractVector{<:Vec{dim,W}};
            outside_value=NaN, project_in=false, is_diag=false, throw_errors=false
            )::SparseMatrixCSC{S,Int64} where {dim,S,W}


    x::Vector{Vec{dim,W}} = [project_in_xin(ctx, x_cur, project_in) for x_cur in x_in]

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
            local_coordinates::Vec{dim,W}, nodes::Vector{Int}, cellid::Int = locatePoint(ctx, x[current_point])

            if ctx.ip isa FEM.Lagrange
                if !is_diag
                    for i in 1:(size(vals)[2])
                        summed_value = 0.0
                        for (j, nodeid) in enumerate(nodes)
                            val::W = FEM.value(ctx.ip, j, local_coordinates)
                            summed_value += vals[nodeid,i]*val
                        end
                        push!(rows_tmp,i)
                        push!(vals_tmp, summed_value)
                    end
                else
                    for (j, nodeid) in enumerate(nodes)
                        val = FEM.value(ctx.ip, j, local_coordinates)
                        push!(rows_tmp, nodeid)
                        push!(vals_tmp,vals[nodeid,nodeid]*val)
                    end
                end
            elseif ctx.ip isa FEM.PiecewiseConstant
                val = FEM.value(ctx.ip, 1, local_coordinates)
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
            if y isa DomainError
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
    return SparseMatrixCSC(size(vals)[2], npoints, result_colptr, result_rows, result_vals)
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
            ctx::GridContext{dim}, dofvals::AbstractMatrix{S},
            x_in::AbstractVector{<:Vec{dim}};
            outside_value=NaN, project_in=false,throw_errors=false
            )::SparseMatrixCSC{S,Int64} where {dim,S}
    u_vals = zeros(S, size(dofvals))
    if ctx.ip isa FEM.Lagrange
        for i in 1:ctx.n
            u_vals[i,:] = dofvals[ctx.node_to_dof[i],:]
        end
    elseif ctx.ip isa FEM.PiecewiseConstant
        for i in 1:ctx.n
            u_vals[i,:] = dofvals[ctx.cell_to_dof[i],:]
        end
    end
    return evaluate_function_from_node_or_cellvals_multiple(ctx, u_vals, x_in;
                outside_value=outside_value, project_in=project_in, throw_errors=throw_errors)
end



"""
    sample_to(u::Vector{T},ctx_old,ctx_new;[bdata=BoundaryData(),project_in=true,outside_value=NaN])

Perform nodal_interpolation of a function onto a different grid.
"""
function sample_to(u::Vector{T}, ctx_old::GridContext, ctx_new::GridContext;
        bdata=BoundaryData(), project_in=true, outside_value=NaN) where {T}
    if !isa(ctx_new.ip, FEM.Lagrange)
        throw(AssertionError("Nodal interpolation only defined for Lagrange elements"))
    else
        u_node_or_cellvals = undoBCS(ctx_old, u, bdata)[ctx_old.node_to_dof]
    end
    return nodal_interpolation(ctx_new,
                x -> evaluate_function_from_node_or_cellvals(ctx_old, u_node_or_cellvals, x;
                outside_value=outside_value, project_in=project_in))
end

"""
    sample_to(u::AbstractArray{2,T},ctx_old,ctx_new; [bdata=BoundaryData(),project_in=true,outside_value=NaN])

Perform nodal_interpolation of a function onto a different grid for a set of columns of a matrix.
Returns a matrix
"""
function sample_to(u::AbstractMatrix{T}, ctx_old::GridContext, ctx_new::GridContext;
        bdata=BoundaryData(), project_in=true, outside_value=NaN) where {T}

    if !isa(ctx_new.ip, FEM.Lagrange)
        throw(AssertionError("Nodal interpolation only defined for Lagrange elements"))
    end

    ncols = size(u)[2]
    u_node_or_cellvals = zeros(T, ctx_old.n, ncols)
    for j in 1:ncols
        if ctx_old.ip isa FEM.Lagrange
            u_node_or_cellvals[:,j] .= undoBCS(ctx_old,u[:,j],bdata)[ctx_old.node_to_dof]
        else
            u_node_or_cellvals[:,j] .= undoBCS(ctx_old,u[:,j],bdata)[ctx_old.cell_to_dof]
        end
    end
    #TODO: Maybe make this more efficient by calling evaluate_function_from_node_or_cellvals_multiple
    ctx_new_gridpoints = [p.x for p in ctx_new.grid.nodes]
    u_new = zeros(T, ctx_new.n, ncols)*NaN
    for j in 1:ncols
        u_new[:,j] = evaluate_function_from_node_or_cellvals_multiple(ctx_old, u_node_or_cellvals[:,j:j], ctx_new_gridpoints;
            outside_value=outside_value, project_in=project_in)[ctx_new.dof_to_node]
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

function delaunay2(x::Vector{<:Union{Vec{2,Float64},NTuple{2,Float64}}})
    width = VD.max_coord - VD.min_coord
    @inbounds begin
        max_x = maximum(map(v->v[1], x))
        min_x = minimum(map(v->v[1], x))
        max_y = maximum(map(v->v[2], x))
        min_y = minimum(map(v->v[2], x))
    end
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


#See if this cell, or horizontal (and vertical if on_cylinder==false) translations
#of it have already been added to the triangulation.
#This is a helper function
function in_cells_used(cells_used_arg,element::Tuple{Int,Int,Int},num_nodes_in,on_cylinder=false)::Int
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
    ydirections = on_cylinder ? (0:0) : (-1:1)
    for directionx in -1:1
        for directiony in ydirections
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
if `on_torus` is set to `true` similarly with `on_cylinder`.
"""
function FEM.generate_grid(::Type{FEM.Triangle},
     nodes_in::AbstractVector{Vec{2,Float64}};
     on_torus=false, on_cylinder=false, LL=nothing, UR=nothing)
    @assert !(on_torus && on_cylinder)
    if on_torus || on_cylinder
        @assert !(LL === nothing || UR === nothing)
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
        for i in (0, 1, -1), j in (0, 1, -1)
            for (index, node) in enumerate(nodes_in)
                new_point = node .+ (i*dx, j*dy)
                push!(nodes_to_triangulate, Vec{2}((new_point[1], new_point[2])))
                push!(points_mapping, index)
            end
        end
    elseif on_cylinder
        dx = UR[1] - LL[1]
        dy = UR[2] - LL[2]
        for i in (0, 1, -1)
            j = 0
            for (index, node) in enumerate(nodes_in)
                new_point = node .+ (i*dx, j*dy)
                push!(nodes_to_triangulate, Vec{2}((new_point[1], new_point[2])))
                push!(points_mapping, index)
            end
        end
    else
        for (index, node) in enumerate(nodes_in)
            push!(nodes_to_triangulate, node)
            push!(points_mapping, index)
        end
    end

    #Delaunay Triangulation
    tess, m, scale_x, scale_y, min_x, min_y = delaunay2(nodes_to_triangulate)

    nodes = map(FEM.Node, nodes_in)

    #Our grid will require some nodes twice.
    #additional_nodes[i] = j means that we require the node i
    #and that it has been added at index j to the grid's list of nodes
    additional_nodes = Dict{Int,Int}()


    #Cells that are part of the grid later
    cells = FEM.Triangle[]

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
    while tri_iterator !== nothing
        (tri, triindex) = tri_iterator
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

        if !(on_torus || on_cylinder)
            push!(cells, FEM.Triangle(new_tri_nodes_from_tess))
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
                                    num_nodes_in, on_cylinder)
                if thiscell != 0
                    #We have, so nothing to do here except record this.
                   cell_number_table[triindex.ix-1] = thiscell
                else
                    #Now iterate over the nodes of the cell
                    for (index, cur) in enumerate(tri_nodes)
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
                                nodes[original_node] = FEM.Node(nodes_to_triangulate[cur])
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
                                push!(nodes, FEM.Node(nodes_to_triangulate[cur]))
                                additional_nodes[cur] = length(nodes)
                                tri_nodes[index] = length(nodes)
                            end
                        end
                    end
                    #Phew. We can now add the triangle to the triangulation.
                    new_tri = FEM.Triangle(Tuple{Int,Int,Int}(tri_nodes))
                    push!(cells, new_tri)

                    #We now remember that we've used this cell
                    cell_number_table[triindex.ix-1] = length(cells)
                    cells_used[ Tuple{Int,Int,Int}(sort(collect(new_tri_nodes_from_tess))) ] = length(cells)
                end
            else #Deal with this cell later
                cells_to_deal_with[Tuple{Int,Int,Int}(tri_nodes)] = triindex.ix-1
            end
        end
        tri_iterator = Base.iterate(tess, triindex)
    end
    #Write down location of remaining cells
    if on_torus || on_cylinder
        for (c,index) in cells_to_deal_with
            thiscell = in_cells_used(cells_used,c,num_nodes_in,on_cylinder)
            cell_number_table[index] = thiscell
        end
    end

    #Check if everything worked out fine
    used_nodes = zeros(Bool, length(nodes))
    for c in cells
        used_nodes[collect(c.nodes)] .= true
    end
    if any(x -> x == false, used_nodes)
        @warn "Some nodes added that might cause problems with FEM. Proceed at your own risk."
    end

    facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = FEM.Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    locator = DelaunayCellLocator(m, scale_x, scale_y, min_x, min_y, tess,nodes_to_triangulate[switched_nodes_table], points_mapping, cell_number_table)
    return grid, locator
end

function FEM.generate_grid(::Type{FEM.QuadraticTriangle}, nodes_in::Vector{Vec{2,Float64}})

    nodes_to_triangulate = Vector{Vec{2,Float64}}[]
    dx = UR[1] - LL[1]
    dy = UR[2] - LL[2]

    points_mapping = Vector{Int}[]

    tess, m, scale_x, scale_y, minx, miny = delaunay2(nodes_to_triangulate)
    locator = P2DelaunayCellLocator(m, scale_x, scale_y, minx, miny, tess,points_mapping)
    nodes = map(FEM.Node, nodes_in)
    n = length(nodes)
    ctr = n #As we add nodes (for edge vertices), increment the ctr...

    centerNodes = spzeros(n,n)
    cells = FEM.QuadraticTriangle[]
    for tri_id in 1:m
        tri = tess._trigs[locator.internal_triangles[tri_id]]

        #Create non-vertex nodes
        ab = centerNodes[tri._a.id, tri._b.id]
        if ab == 0
            ctr += 1
            ab = centerNodes[tri._a.id,tri._b.id] = centerNodes[tri._b.id,tri._a.id] =  ctr
            center = FEM.Node(0.5*(nodes[tri._b.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end
        ac = centerNodes[tri._a.id, tri._c.id]
        if ac == 0
            ctr += 1
            ac = centerNodes[tri._a.id,tri._c.id] = centerNodes[tri._c.id,tri._a.id] = ctr
            center = FEM.Node(0.5*(nodes[tri._c.id].x + nodes[tri._a.id].x))
            push!(nodes,center)
        end

        bc = centerNodes[tri._c.id, tri._b.id]
        if bc == 0
            ctr += 1
            bc = centerNodes[tri._b.id,tri._c.id] = centerNodes[tri._c.id,tri._b.id] = ctr
            center = FEM.Node(0.5*(nodes[tri._c.id].x + nodes[tri._b.id].x))
            push!(nodes,center)
        end

        J = Tensors.otimes((nodes_in[tri._b.id] - nodes_in[tri._a.id]) , e1)
        J +=  Tensors.otimes((nodes_in[tri._c.id] - nodes_in[tri._a.id]) , e2)
        detJ = det(J)

        @assert det(J) != 0
        if detJ > 0
            new_tri = FEM.QuadraticTriangle((tri._a.id,tri._b.id,tri._c.id,ab,bc,ac))
        else
            new_tri = FEM.QuadraticTriangle((tri._a.id,tri._c.id,tri._b.id,ac,bc,ab))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = FEM.Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    return grid, locator

end

"""
    nodal_interpolation(ctx,f)

Perform nodal interpolation of a function. Returns a vector of coefficients in dof order.
"""
function nodal_interpolation(ctx::GridContext, f::Function)
    return [f(ctx.grid.nodes[ctx.dof_to_node[j]].x) for j in 1:ctx.n]
end

"""
    getCellMidPoint(ctx,cellindex)

Returns the midpoint of the cell `cellindex`
"""

function getCellMidpoint(ctx, cellindex)
    cell = ctx.grid.cells[cellindex]
    nnodes = length(cell.nodes)
    result = Vec((0.0, 0.0))
    for i in 1:length(cell.nodes)
        result += ctx.grid.nodes[cell.nodes[i]].x
    end
    return result/nnodes
end


###P2 Grids in 3D:
#TODO: See if this can be moved upstream

const localnodes = [((1,1,1),(3,1,1),(1,3,1),(1,3,3)),
                    ((1,1,1),(1,1,3),(3,1,1),(1,3,3)),
                    ((3,1,1),(3,3,1),(1,3,1),(1,3,3)),
                    ((3,1,1),(3,3,3),(3,3,1),(1,3,3)),
                    ((3,1,1),(1,1,3),(3,1,3),(1,3,3)),
                    ((3,1,1),(3,1,3),(3,3,3),(1,3,3))
                    ]

_avg(x, y) = (x == 1 && y == 3) || (x == 3 && y == 1) ? 2 : x
_indexavg(x, y) = CartesianIndex(_avg.(Tuple(x), Tuple(y)))

#Based on Ferrite's generate_grid(Tetrahedron, ...) function
function FEM.generate_grid(::Type{FEM.QuadraticTetrahedron}, cells_per_dim::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
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
    nodes = Vector{FEM.Node{3,T}}(undef,total_nodes)
    cells = Vector{FEM.QuadraticTetrahedron}(undef,total_elements)

    # Generate nodes
    node_idx = 1
    @inbounds for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        nodes[node_idx] = FEM.Node((coords_x[i], coords_y[j], coords_z[k]))
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
        for (idx, p1vertices) in enumerate(localnodes)
            v1,v2,v3,v4 = map(CartesianIndex,p1vertices)
            cells[cell_idx + idx] = FEM.QuadraticTetrahedron((cube[v1], cube[v2], cube[v3], cube[v4],
                        cube[_indexavg(v1,v2)], cube[_indexavg(v2,v3)], cube[_indexavg(v1,v3)],
                        cube[_indexavg(v1,v4)], cube[_indexavg(v2,v4)], cube[_indexavg(v3,v4)]))
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

    boundary_matrix = FEM.boundaries_to_sparse([le; ri; bo; to; fr; ba])

    facesets = Dict(
        "left" => Set(le),
        "right" => Set(ri),
        "front" => Set(fr),
        "back" => Set(ba),
        "bottom" => Set(bo),
        "top" => Set(to),
    )
    return FEM.Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end
