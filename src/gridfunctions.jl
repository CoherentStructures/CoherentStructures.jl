#(c) 2017 Nathanael Schilling
#This file implements methods for working with JuAFEM grids
#This includes methods for making grids from Delaunay Triangulations based
#on the code in FEMDL.jl
#There are also functions for evaluating functions on the grid

GP = GeometricalPredicates
VD = VoronoiDelaunay

#JuAFEM has no functions for determining which cell a point is in.

const default_quadrature_order=5
const default_quadrature_order3D=2


"""
    struct gridContext<dim>

Stores everything needed as "context" to be able to work on a FEM grid based on the `JuAFEM` package
Adds a point-location API which makes it possible to plot functions defined on the grid within julia.
Currently only `dim==2` is implemented

# Fields
- `grid::JuAFEM.Grid`,`ip::JuAFEM.Interpolation`,`qr::JuAFEM.QuadratureRule` - See the `JuAFEM` package
- `loc::CellLocator` object used for point-location on the grid.
- `node_to_dof::Vector{Int}`  lookup table for dof index of a node
- `dof_to_node::Vector{Int}`  inverse of node_to_dof
- `n::Int` number of nodes on the grid
- `m::Int` number of elements (e.g. triangles,quadrilaterals, ...) on the grid
- `quadrature_points::Vector{Vec{dim,Float64}}` All quadrature points on the grid, in a fixed order.
- `mass_weights::Vector{Float64}` Weighting for mass matrix
- `spatialBounds` If available, the corners of a bounding box of a domain. For regular grids, the bounds are tight.
- `numberOfPointsInEachDirection` For regular grids, how many (non-interior) nodes make up the regular grid.
- `gridType` A string describing what kind of grid this is (e.g. "regular triangular grid")
"""
mutable struct gridContext{dim} <: abstractGridContext{dim} #TODO: Currently set as mutable, is this sensible?
    grid::JuAFEM.Grid
    ip::JuAFEM.Interpolation
    dh::JuAFEM.DofHandler
    qr::JuAFEM.QuadratureRule
    loc::cellLocator
    node_to_dof::Vector{Int} #node_to_dof[nodeid] contains the index of the corresponding dof
    dof_to_node::Vector{Int} #dof_to_node[dofid] contains the index of the corresponding node
    n::Int #The number of nodes
    m::Int #The number of cells
    quadrature_points::Vector{Tensors.Vec{dim,Float64}} #All quadrature points, ordered by how they are accessed in assemble routines
    mass_weights::Vector{Float64}

    #The following two fields are only well-defined for regular rectangular grids
    spatialBounds::Vector{AbstractVector} #This is {LL,UR} for regular grids
    #This is the number of (non-interior) nodes in each direction (not points)
    numberOfPointsInEachDirection::Vector{Int}

    gridType::String

    function gridContext{dim}(
                grid::JuAFEM.Grid,
                ip::JuAFEM.Interpolation,
                dh::JuAFEM.DofHandler,
                qr::JuAFEM.QuadratureRule,
                loc::cellLocator
            ) where {dim}

        x =new{dim}(grid,ip,dh,qr,loc)
        x.n = JuAFEM.getnnodes(dh.grid)
        x.m = JuAFEM.getncells(dh.grid)

        #TODO: Measure if the sorting below is expensive
        x.node_to_dof = nodeToDHTable(x)
        x.dof_to_node = sortperm(x.node_to_dof)
        x.quadrature_points = getQuadPoints(x)
        x.mass_weights = ones(length(x.quadrature_points))
        return x
    end
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
            LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
            kwargs...
        )

    if gridType == "regular triangular grid"
        return regularTriangularGrid(numnodes, LL,UR;kwargs...)
    elseif gridType == "regular Delaunay grid"
        return regularDelaunayGrid(numnodes, LL,UR; kwargs...)
    elseif gridType == "regular P2 triangular grid"
        return regularP2TriangularGrid(numnodes, LL,UR; kwargs...)
    elseif gridType == "regular P2 Delaunay grid"
        return regularP2DelaunayGrid(numnodes, LL,UR; kwargs...)
    elseif gridType == "regular quadrilateral grid"
        return regularQuadrilateralGrid(numnodes, LL,UR; kwargs...)
    elseif gridType == "regular P2 quadrilateral grid"
        return regularP2QuadrilateralGrid(numnodes, LL,UR; kwargs...)
    else
        fail("Unsupported grid type '$gridType'")
    end
end

#Based on JuAFEM's WriteVTK.vtk_point_data
function nodeToDHTable(ctx::abstractGridContext{dim}) where {dim}
    dh::DofHandler = ctx.dh
    const n = ctx.n
    res = fill(-1,n)
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
"""
    gridContext{2}(JuAFEM.Triangle, node_list, [quadrature_order=default_quadrature_order])

Create a P1-Lagrange grid based on Delaunay Triangulation.
Uses `DelaunayVoronoi.jl` internally.
"""
(::Type{gridContext{2}})(#Defined like this so julia doesn't complain that 2 is not a type
            ::Type{JuAFEM.Triangle},
            node_list::Vector{Tensors.Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order) =
begin
        grid,loc = generate_grid(JuAFEM.Triangle,node_list)
        ip = Lagrange{2, RefTetrahedron, 1}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.gridType="irregular Delaunay grid" #This can be ovewritten by other constructors
        return result
end


"""
    regularDelaunayGrid(numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular grid on a square.
Internally uses Delauny Triangulation.
"""
function regularDelaunayGrid(
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order
        )

    node_list = Tensors.Vec{2,Float64}[]
    for x0 in linspace(LL[1],UR[1],numnodes[1])
        for x1 in linspace(LL[2],UR[2],numnodes[2])
            push!(node_list,Tensors.Vec{2}([x0,x1]))
        end
    end
    result = gridContext{2}(JuAFEM.Triangle,node_list, quadrature_order=quadrature_order)
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
    result.gridType = "regular Delaunay grid"
    return result
end

"""
    gridContext{2}(JuAFEM.QuadraticTriangle, node_list, quadrature_order=default_quadrature_order)

Create a P2 grid given a set of (non-interior) nodes using Delaunay Triangulation.
"""
(::Type{gridContext{2}})(
            ::Type{JuAFEM.QuadraticTriangle},
            node_list::Vector{Tensors.Vec{2,Float64}};
            quadrature_order::Int=default_quadrature_order) =
begin
        grid,loc = generate_grid(JuAFEM.QuadraticTriangle,node_list)
        ip = Lagrange{2, RefTetrahedron, 2}()
        dh = DofHandler(grid)
        qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{2}(grid,ip,dh,qr,loc)
        result.gridType = "irregular P2 Delaunay grid"
        return result
end

"""
    regularP2DelaunayGrid(numnodes=(25,25),LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

    Create a regular P2 triangular grid with numnodes being the number of (non-interior) nodes in each direction.
"""
function regularP2DelaunayGrid(
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order
        )

    node_list = Tensors.Vec{2,Float64}[]
    for x0 in linspace(LL[1],UR[1],numnodes[1])
        for x1 in linspace(LL[2],UR[2],numnodes[2])
            push!(node_list,Tensors.Vec{2}([x0,x1]))
        end
    end
    result = gridContext{2}(JuAFEM.QuadraticTriangle,node_list, quadrature_order=quadrature_order)
    #TODO: Think about what values would be sensible for the two variables below
    result.spatialBounds = [LL,UR]
    result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2]]
    result.gridType = "regular P2 Delaunay grid"
    return result
end


"""
    gridContext{2}(JuAFEM.Triangle, numnodes=(25,25),LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular Triangular Grid. Does not use Delaunay triangulation internally.
"""
( ::Type{gridContext{2}})(::Type{JuAFEM.Triangle},
                         numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
                         quadrature_order::Int=default_quadrature_order) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
        grid = generate_grid(JuAFEM.Triangle,(numnodes[1]-1,numnodes[2]-1),Tensors.Vec{2}(LL),Tensors.Vec{2}(UR) )
        loc = regular2DGridLocator{JuAFEM.Triangle}(numnodes[1],numnodes[2],Tensors.Vec{2}(LL),Tensors.Vec{2}(UR))
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

"""
    regularTriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P1 triangular grid on a Rectangle. Does not use Delaunay triangulation internally.
"""
function regularTriangularGrid(numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);quadrature_order::Int=default_quadrature_order)
    return gridContext{2}(JuAFEM.Triangle,numnodes, LL,UR,quadrature_order=quadrature_order)
end

"""
    gridContext{2}(JUAFEM.QuadraticTriangle, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 triangular grids. Does not use Delaunay triangulation internally.
"""
(::Type{gridContext{2}})(::Type{JuAFEM.QuadraticTriangle},
                         numnodes::Tuple{Int,Int}=(25,25),LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
                         quadrature_order::Int=default_quadrature_order) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = generate_grid(JuAFEM.QuadraticTriangle,(numnodes[1]-1,numnodes[2]-1),Tensors.Vec{2}(LL), Tensors.Vec{2}(UR ))
    loc = regular2DGridLocator{JuAFEM.QuadraticTriangle}(numnodes[1],numnodes[2],Tensors.Vec{2}(LL),Tensors.Vec{2}(UR))
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


"""
    regularP2TriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P2 triangular grid on a Rectangle. Does not use Delaunay triangulation internally.
"""
function regularP2TriangularGrid(
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order
        )
    return gridContext{2}(JuAFEM.QuadraticTriangle,numnodes, LL,UR,quadrature_order=quadrature_order)
end




"""
    gridContext{2}(JUAFEM.Quadrilateral, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P1 quadrilateral grids.
"""
(::Type{gridContext{2}})(
            ::Type{JuAFEM.Quadrilateral},
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order) =
begin
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = generate_grid(JuAFEM.Quadrilateral,(numnodes[1]-1,numnodes[2]-1),Tensors.Vec{2}(LL),Tensors.Vec{2}(UR))
    loc = regular2DGridLocator{JuAFEM.Quadrilateral}(numnodes[1],numnodes[2],Tensors.Vec{2}(LL),Tensors.Vec{2}(UR))
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

"""
    regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P1 quadrilateral grid on a Rectangle.
"""
function regularQuadrilateralGrid(
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order
        )
    return gridContext{2}(JuAFEM.Quadrilateral,numnodes, LL,UR,quadrature_order=quadrature_order)
end


"""
    gridContext{2}(JUAFEM.QuadraticQuadrilateral, numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Constructor for regular P2 quadrilateral grids.
"""
(::Type{gridContext{2}})(
            ::Type{JuAFEM.QuadraticQuadrilateral},
            numnodes::Tuple{Int,Int}=(25,25),
            LL::Tensors.Vec{2}=Tensors.Vec{2}([0.0,0.0]),
            UR::Tensors.Vec{2}=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order) =
begin
    #The -1 below is needed because JuAFEM internally then goes on to increment it
    grid = generate_grid(JuAFEM.QuadraticQuadrilateral,(numnodes[1]-1,numnodes[2]-1),Tensors.Vec{2}(LL), Tensors.Vec{2}(UR) )
    loc = regular2DGridLocator{JuAFEM.QuadraticQuadrilateral}(numnodes[1],numnodes[2],Tensors.Vec{2}(LL),Tensors.Vec{2}(UR))
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


"""
    regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular P2 quadrilateral grid on a Rectangle.
"""
function regularP2QuadrilateralGrid(
            numnodes::Tuple{Int,Int}=(25,25),
            LL::AbstractVector{Float64}=Tensors.Vec{2}([0.0,0.0]),
            UR::AbstractVector{Float64}=Tensors.Vec{2}([1.0,1.0]);
            quadrature_order::Int=default_quadrature_order
        )
    return gridContext{2}(JuAFEM.QuadraticQuadrilateral,numnodes, Tensors.Vec{2}(LL),Tensors.Vec{2}(UR),quadrature_order=quadrature_order)
end


"""
    gridContext{3}(JuAFEM.Tetrahedron, numnodes=(10,10,10),LL=[0.0,0.0,0.0],UR=[1.0,1.0,1.0],quadrature_order=default_quadrature_order)

Create a regular Tetrahedral Grid. Does not use Delaunay triangulation internally.
"""
( ::Type{gridContext{3}})(::Type{JuAFEM.Tetrahedron},
                         numnodes::Tuple{Int,Int,Int}=(10,10,10),LL::AbstractVector=Tensors.Vec{3}([0.0,0.0,0.0]),UR::AbstractVector=Tensors.Vec{3}([1.0,1.0,1.0]);
                         quadrature_order::Int=default_quadrature_order3D) = begin
        #The -1 below is needed because JuAFEM internally then goes on to increment it
        grid = generate_grid(JuAFEM.Tetrahedron,(numnodes[1]-1,numnodes[2]-1,numnodes[3] -1),Tensors.Vec{3}(LL),Tensors.Vec{3}(UR) )
        loc = regular3DGridLocator{JuAFEM.Tetrahedron}(numnodes[1],numnodes[2],numnodes[3],Tensors.Vec{3}(LL),Tensors.Vec{3}(UR))
        ip = Lagrange{3, RefTetrahedron, 1}()
        dh = DofHandler(grid)
        qr = QuadratureRule{3, RefTetrahedron}(quadrature_order)
        push!(dh, :T, 1) #The :T is just a generic name for the scalar field
        close!(dh)
        result =  gridContext{3}(grid,ip,dh,qr,loc)
        result.spatialBounds = [LL,UR]
        result.numberOfPointsInEachDirection = [numnodes[1],numnodes[2],numnodes[3]]
        result.gridType = "3D regular triangular grid"
        return result
end


"""
    regularTriangularGrid3D(numnodes=(10,10,10), LL=[0.0,0.0,0.0],UR=[1.0,1.0,1.0],quadrature_order=default_quadrature_order3D)

Create a regular P1 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.
"""
function regularTriangularGrid3D(numnodes::Tuple{Int,Int,Int}=(10,10,10),LL::AbstractVector=Tensors.Vec{3}([0.0,0.0,0.0]),UR::AbstractVector=Tensors.Vec{3}([1.0,1.0,1.0]);quadrature_order::Int=default_quadrature_order3D)
    return gridContext{3}(JuAFEM.Tetrahedron,numnodes, LL,UR,quadrature_order=quadrature_order)
end



"""
    locatePoint(ctx,x)
Point location on grids.
Returns a tuple (coords, [nodes])
where coords gives the coordinates within the reference shape (e.g. standard simplex)
And [nodes] is the list of corresponding node ids, ordered in the order of the
corresponding shape functions from JuAFEM's interpolation.jl file.
"""#TODO: could this be more efficient, etc.. with multiple dispatch?
function locatePoint(ctx::gridContext{dim}, x::AbstractVector{Float64}) where dim
    return locatePoint(ctx.loc,ctx.grid,x)
end

"""
    evaluate_function_from_nodevals(ctx,nodevals,x_in; [outside_value=0,project_in=false])

Like `evaluate_function_from_dofvals`, but the coefficients from `nodevals` are assumed to be in node order.
"""
function evaluate_function_from_nodevals{dim}(ctx::gridContext{dim},nodevals::Vector{Float64},x_in::AbstractVector{Float64},outside_value=0.0,project_in=false)
    if !project_in
        if dim == 2
            x = Tensors.Vec{dim,Float64}((x_in[1],x_in[2]))
        elseif dim == 3
            x = Tensors.Vec{dim,Float64}((x_in[1],x_in[2],x_in[3]))
        else
            error("dim = $dim not supported")
        end
    else
        if dim == 2
            #TODO: replace this with a macro maybe
            x = Tensors.Vec{3,Float64}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1],x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2],x_in[2]))
                ))
        elseif dim == 3
            #TODO: replace this with a macro maybe
            x = Tensors.Vec{3,Float64}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1],x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2],x_in[2]))
                ,max(ctx.spatialBounds[1][3], min(ctx.spatialBounds[2][3],x_in[3]))
                ))
        else
            error("dim = $dim not supported")
        end
    end
    assert(length(nodevals) == ctx.n)
    local_coordinates,nodes = try
         locatePoint(ctx,x)
    catch y
        if isa(y,DomainError)
            return outside_value
        end
        print("Unexpected error for $x")
        throw(y)
    end
    result = 0.0
    for (j,nodeid) in enumerate(nodes)
        result +=  nodevals[nodeid]*JuAFEM.value(ctx.ip, j, local_coordinates)
    end
    return result
end


"""
    evaluate_function_from_dofvals(ctx,dofvals,x_in; [outside_value=0,project_in=false])

Evaluate a function in the approximation space at the point `x_in`. If `x_in` is out of points, return `outside_value`.
If `project_in` is `true`, points not within `ctx.spatialBounds` are first projected into the domain.

The coefficients in `nodevals` are interpreted to be in dof order.
"""
function evaluate_function_from_dofvals{dim}(ctx::gridContext{dim},dofvals::Vector{Float64},x_in::AbstractVector{Float64},outside_value=0.0,project_in=false)
    if !project_in
        if dim == 2
            x = Tensors.Vec{dim,Float64}((x_in[1],x_in[2]))
        elseif dim == 3
            x = Tensors.Vec{dim,Float64}((x_in[1],x_in[2],x_in[3]))
        else
            error("dim = $dim not supported")
        end
    else
        if dim == 2
            x = Tensors.Vec{dim,Float64}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1],x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2],x_in[2]))
                ))
        elseif dim == 3
            x = Tensors.Vec{dim,Float64}(
                (max(ctx.spatialBounds[1][1], min(ctx.spatialBounds[2][1],x_in[1]))
                ,max(ctx.spatialBounds[1][2], min(ctx.spatialBounds[2][2],x_in[2]))
                ,max(ctx.spatialBounds[1][3], min(ctx.spatialBounds[2][3],x_in[3]))
                ))
        else
            error("dim = $dim not supported")
        end
    end
    assert(length(dofvals) == ctx.n)
    local_coordinates,nodes = try
         locatePoint(ctx,x)
    catch y
        if isa(y,DomainError)
            return outside_value
        end
        print("Unexpected error for $x")
        throw(y)
    end
    result = 0.0
    for (j,nodeid) in enumerate(nodes)
        result +=  dofvals[ctx.node_to_dof[nodeid]]*JuAFEM.value(ctx.ip, j, local_coordinates)
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
    NumberedPoint2D(p::Tensors.Vec{2,Float64}) = new(p[1], p[2], 0)
 end
 GP.Point(x::Real, y::Real, k::Int64) = NumberedPoint2D(x, y, k)
 GP.Point2D(p::NumberedPoint2D) = Point2D(p.x,p.y)
 GP.gety(p::NumberedPoint2D) = p.y
 GP.getx(p::NumberedPoint2D) = p.x

#More or less equivalent to matlab's delaunay function, based on code from FEMDL.jl

function delaunay2(x::Vector{Tensors.Vec{2,Float64}})
    width = VD.max_coord - VD.min_coord
    max_x = maximum(map(v->v[1],x))
    minx = minimum(map(v->v[1],x))
    max_y = maximum(map(v->v[2],x))
    miny = minimum(map(v->v[2],x))
    scale_x = 0.9*width/(max_x - minx)
    scale_y = 0.9*width/(max_y - miny)
    n = length(x)
    a = [NumberedPoint2D(VD.min_coord+(x[i][1] - minx)*scale_x,VD.min_coord+(x[i][2]-miny)*scale_y,i) for i in 1:n]
    for i in 1:n
        assert(!(GP.getx(a[i]) < VD.min_coord || GP.gety(a[i]) > VD.max_coord))
    end
    tess = VD.DelaunayTessellation2D{NumberedPoint2D}(n)
    push!(tess,a)
    m = 0
    for i in tess
        m += 1
    end
    return tess,m,scale_x,scale_y,minx,miny
end

#For delaunay triangulations, we can use the tesselation
#object and the locate() function
struct delaunayCellLocator <: cellLocator
    m::Int64
    scale_x::Float64
    scale_y::Float64
    minx::Float64
    miny::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
end

function locatePoint(loc::delaunayCellLocator, grid::JuAFEM.Grid, x::AbstractVector{Float64})
    point_inbounds = NumberedPoint2D(VD.min_coord+(x[1]-loc.minx)*loc.scale_x,VD.min_coord+(x[2]-loc.miny)*loc.scale_y)
    if min(point_inbounds.x, point_inbounds.y) < VD.min_coord || max(point_inbounds.x,point_inbounds.y) > VD.max_coord
        throw(DomainError())
    end
    t = VD.locate(loc.tess, point_inbounds)
    if VD.isexternal(t)
        throw(DomainError())
    end
    v1::Tensors.Vec{2} = grid.nodes[t._b.id].x - grid.nodes[t._a.id].x
    v2::Tensors.Vec{2} = grid.nodes[t._c.id].x - grid.nodes[t._a.id].x
    J::Tensors.Tensor{2,2,Float64,4} = v1 ⊗ e1  + v2 ⊗ e2
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
    minx::Float64
    miny::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
    internal_triangles::Vector{Int}
    inv_internal_triangles::Vector{Int}
    function p2DelaunayCellLocator(m,scale_x,scale_y,minx,miny,tess)
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
        res = new(m,scale_x,scale_y,minx,miny,tess,internal_triangles,inv_internal_triangles)
        return res
    end
end

function locatePoint(loc::p2DelaunayCellLocator, grid::JuAFEM.Grid, x::AbstractVector{Float64})
    point_inbounds = NumberedPoint2D(VD.min_coord+(x[1]-loc.minx)*loc.scale_x,VD.min_coord+(x[2]-loc.miny)*loc.scale_y)
    if min(point_inbounds.x, point_inbounds.y) < VD.min_coord || max(point_inbounds.x,point_inbounds.y) > VD.max_coord
        throw(DomainError())
    end
    t = VD.findindex(loc.tess, point_inbounds)
    if VD.isexternal(loc.tess._trigs[t])
        throw(DomainError())
    end
    const qTriangle = grid.cells[loc.inv_internal_triangles[t]]
    v1::Tensors.Vec{2} = grid.nodes[qTriangle.nodes[2]].x - grid.nodes[qTriangle.nodes[1]].x
    v2::Tensors.Vec{2} = grid.nodes[qTriangle.nodes[3]].x - grid.nodes[qTriangle.nodes[1]].x
    J::Tensors.Tensor{2,2,Float64,4} = v1 ⊗ e1  + v2 ⊗ e2
    #TODO: Think about whether doing it like this (with the permutation) is sensible
    return (inv(J) ⋅ (x - grid.nodes[qTriangle.nodes[1]].x)), permute!(collect(qTriangle.nodes),[2,3,1,5,6,4])
end

#Here N gives the number of nodes and M gives the number of faces
struct regular2DGridLocator{T} <: cellLocator where {M,N,T <: JuAFEM.Cell{2,M,N}}
    nx::Int
    ny::Int
    LL::Tensors.Vec{2}
    UR::Tensors.Vec{2}
end
function locatePoint(loc::regular2DGridLocator{JuAFEM.Triangle},grid::JuAFEM.Grid, x::AbstractVector{Float64})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.nx-1) #If we hit the right hand edge
        n1 = loc.nx-2
        loc1 = 1.0
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        n2 = loc.ny-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in:
    ll = n1 + n2*loc.nx
    lr = ll + 1
    ul = n1 + (n2+1)*loc.nx
    ur = ul + 1
    assert(ur < (loc.nx * loc.ny))
    if loc1 + loc2 <= 1.0 # ◺
        return Tensors.Vec{2}([loc1,loc2]), [lr+1, ul+1,ll+1]
    else # ◹
        #The transformation that maps ◹ (with bottom node at origin) to ◺ (with ll node at origin)
        #Does [0,1] ↦ [1,0] and [-1,1] ↦ [0,1]
        #So it has representation matrix (columnwise) [ [1,-1] | [1,0] ]
        const tM = Tensors.Tensor{2,2,Float64,4}((1.,-1.,1.,0.))
        return tM⋅Tensors.Vec{2}([loc1-1,loc2]), [ ur+1, ul+1,lr+1]
    end
end

function locatePoint(loc::regular2DGridLocator{JuAFEM.Quadrilateral},grid::JuAFEM.Grid, x::AbstractVector{Float64})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.nx-1) #If we hit the right hand edge
        n1 = loc.nx-2
        loc1 = 1.0
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        n2 = loc.ny-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in:
    ll = n1 + n2*loc.nx
    lr = ll + 1
    ul = n1 + (n2+1)*loc.nx
    ur = ul + 1
    assert(ur < (loc.nx * loc.ny))
    return Tensors.Vec{2}([2*loc1-1,2*loc2-1]), [ll+1,lr+1,ur+1,ul+1]
end

#Same principle as for Triangle type above
function locatePoint(loc::regular2DGridLocator{JuAFEM.QuadraticTriangle},grid::JuAFEM.Grid, x::AbstractVector{Float64})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.nx-1) #If we hit the right hand edge
        n1 = loc.nx-2
        loc1 = 1.0
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        n2 = loc.ny-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in
    #Zero-indexing of the array here, so we need to +1 for everything being returned
    num_x_with_edge_nodes::Int = loc.nx  + loc.nx - 1
    num_y_with_edge_nodes::Int = loc.ny + loc.ny -1
    ll = 2*n1 + 2*n2*num_x_with_edge_nodes
    lr = ll + 2
    ul = 2*n1 + 2(n2+1)*num_x_with_edge_nodes
    ur = ul + 2
    middle_left =  2*n1 + (2*n2+1)*num_x_with_edge_nodes
    assert(ur < (num_x_with_edge_nodes*num_y_with_edge_nodes)) #Sanity check
    if loc1 + loc2 <= 1.0 # ◺
        return Tensors.Vec{2}([loc1,loc2]), [lr+1,ul+1,ll+1, middle_left+2, middle_left+1, ll+2]
    else # ◹

        #The transformation that maps ◹ (with bottom node at origin) to ◺ (with ll node at origin)
        #Does [0,1] ↦ [1,0] and [-1,1] ↦ [0,1]
        #So it has representation matrix (columnwise) [ [1,-1] | [1,0] ]
        const tM = Tensors.Tensor{2,2,Float64,4}((1.,-1.,1.,0.))
        return tM⋅Tensors.Vec{2}([loc1-1,loc2]), [ ur+1, ul+1,lr+1,ul+2,middle_left+2, middle_left+3]
    end
    return
end


function locatePoint(loc::regular2DGridLocator{JuAFEM.QuadraticQuadrilateral},grid::JuAFEM.Grid, x::AbstractVector{Float64})
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1),1)
    n1 = Int(n1f)
    n2 = Int(n2f)
    if n1 == (loc.nx-1) #If we hit the right hand edge
        n1 = loc.nx-2
        loc1 = 1.0
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        n2 = loc.ny-2
        loc2 = 1.0
    end
    #Get the four node numbers of quadrilateral the point is in
    #Zero-indexing of the array here, so we need to +1 for everything being returned
    num_x_with_edge_nodes::Int = loc.nx  + loc.nx - 1
    num_y_with_edge_nodes::Int = loc.ny + loc.ny -1
    ll = 2*n1 + 2*n2*num_x_with_edge_nodes
    lr = ll + 2
    ul = 2*n1 + 2(n2+1)*num_x_with_edge_nodes
    ur = ul + 2
    middle_left =  2*n1 + (2*n2+1)*num_x_with_edge_nodes
    assert(ur < (num_x_with_edge_nodes*num_y_with_edge_nodes)) #Sanity check
    #permute!(collect(qTriangle.nodes),[2,3,1,5,6,4])
    return Tensors.Vec{2}([2*loc1-1,2*loc2-1]), [ll+1,lr+1,ur+1,ul+1,ll+2,middle_left+3, ul+2, middle_left+1,middle_left+2]
end


struct regular3DGridLocator{T} <: cellLocator where {M,N,T <: JuAFEM.Cell{3,M,N}}
    nx::Int
    ny::Int
    nz::Int
    LL::Tensors.Vec{3}
    UR::Tensors.Vec{3}
end


#TODO: Make this more robust
function in_tetrahedron(a,b,c,d,p)
    function mydet(p1,p2,p3)
        M = zeros(3,3)
        M[:,1] = p1
        M[:,2] = p2
        M[:,3] = p3
        return det(M)
    end
    my0 = eps()
    return (mydet(b-a,c-a,p-a) >= -my0) && (mydet(b-a,d-a,p-a) <= my0) && (mydet(d-b,c-b,p-b) >= -my0) && (mydet(d-a,c-a,p-a) <= my0)
end

function locatePoint(loc::regular3DGridLocator{JuAFEM.Tetrahedron},grid::JuAFEM.Grid,x::AbstractVector{Float64})
  if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[3] > loc.UR[3] || x[1] < loc.LL[1] || x[2] < loc.LL[2] || x[3] < loc.LL[3]
        throw(DomainError())
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    #warning: all the coputation is done with zero-indexing
    n1f,loc1= divrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1),1)
    n2f,loc2 = divrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1),1)
    n3f,loc3 = divrem((x[3] - loc.LL[3])/(loc.UR[3] - loc.LL[3]) * (loc.nz-1),1)


    n1 = Int(n1f)
    n2 = Int(n2f)
    n3 = Int(n3f)
    if n1 == (loc.nx-1) #If we hit the right hand edge
        n1 = loc.nx-2
        loc1 = 1.0
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        n2 = loc.ny-2
        loc2 = 1.0
    end

    if n3 == (loc.nz-1) #If we hit the top edge
        n3 = loc.nz-2
        loc3 = 1.0
    end
    #Get the 8 node numbers of the rectangular hexahedron the point is in:
    #Ordering is like tmp of JuAFEM's generate_grid(::Type{Tetrahedron})


    if false
        nodes = [
            n1 + n2*loc.nx + n3*loc.nx*loc.ny,
            n1 + 1 + n2*loc.nx + n3*loc.nx*loc.ny,
            n1 + 1 + (n2 + 1)*loc.nx + n3*loc.nx*loc.ny,
            n1 + (n2 + 1)*loc.nx + n3*loc.nx*loc.ny,

            n1 + n2*loc.nx + (n3 + 1)*loc.nx*loc.ny,
            n1 + 1 + n2*loc.nx + (n3+1)*loc.nx*loc.ny,
            n1 + 1 + (n2 + 1)*loc.nx + (n3+1)*loc.nx*loc.ny,
            n1 + (n2 + 1)*loc.nx + (n3+1)*loc.nx*loc.ny,

            ]
    else
        i = n1+1
        j = n2+1
        k = n3+1
        node_array = reshape(collect(0:(loc.nx*loc.ny*loc.nz - 1)), (loc.nx, loc.ny, loc.nz))
        nodes = (node_array[i,j,k], node_array[i+1,j,k], node_array[i+1,j+1,k], node_array[i,j+1,k],
                   node_array[i,j,k+1], node_array[i+1,j,k+1], node_array[i+1,j+1,k+1], node_array[i,j+1,k+1])
   end

    standard_cube = [Tensors.Vec{3}((0.,0.,0.)),Tensors.Vec{3}((1.,0.,0.)),Tensors.Vec{3}((1.,1.,0.)),Tensors. Vec{3}((0.,1.,0.)),
        Tensors.Vec{3}((0.,0.,1.)),Tensors.Vec{3}((1.,0.,1.)),Tensors.Vec{3}((1.,1.,1.)),Tensors.Vec{3}((0.,1.,1.))]

    tetrahedra = [[1,2,4,8], [1,5,2,8], [2,3,4,8], [2,7,3,8], [2,5,6,8], [2,6,7,8]]
    for tet in tetrahedra
        p1,p2,p3,p4 = standard_cube[tet]
        if in_tetrahedron(p1,p2,p3,p4,[loc1,loc2,loc3])
            M = zeros(3,3)
            M[:,1] = p2-p1
            M[:,2] = p3-p1
            M[:,3] = p4-p1
            tMI::Tensors.Tensor{2,3,Float64,9} =  Tensors.Tensor{2,3,Float64}(M)
            return inv(tMI) ⋅ Vec{3}([loc1,loc2,loc3] - p1), nodes[tet] .+ 1
        end
    end
    throw(DomainError()) #In case we didn't land in any tetrahedron
end





function JuAFEM.generate_grid(::Type{JuAFEM.Triangle}, nodes_in::Vector{Tensors.Vec{2,Float64}})
    tess,m,scale_x,scale_y,minx,miny = delaunay2(nodes_in)
    nodes = Node{2,Float64}[]
    for node_coords in  nodes_in
        push!(nodes,Node(node_coords))
    end
    cells = JuAFEM.Triangle[]
    for tri in tess
        J = (nodes_in[tri._b.id] - nodes_in[tri._a.id]) ⊗ e1
        J +=  (nodes_in[tri._c.id] - nodes_in[tri._a.id]) ⊗ e2
        detJ = det(J)
        @assert det(J) != 0
        if detJ > 0
            new_tri  = JuAFEM.Triangle((tri._a.id,tri._b.id,tri._c.id))
        else
            new_tri  = JuAFEM.Triangle((tri._a.id,tri._c.id,tri._b.id))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    locator = delaunayCellLocator(m,scale_x,scale_y,minx,miny,tess)
    return grid, locator
end

function JuAFEM.generate_grid(::Type{JuAFEM.QuadraticTriangle}, nodes_in::Vector{Tensors.Vec{2,Float64}})
    tess,m,scale_x,scale_y,minx,miny = delaunay2(nodes_in)
    locator = p2DelaunayCellLocator(m,scale_x,scale_y,minx,miny,tess)
    nodes = Node{2,Float64}[]
    #TODO: replace below with map
    for node_coords in  nodes_in
        push!(nodes,Node(node_coords))
    end
    n = length(nodes)
    ctr = n #As we add nodes (for edge vertices), increment the ctr...

    centerNodes = spzeros(n,n)
    cells = JuAFEM.QuadraticTriangle[]
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
            new_tri  = JuAFEM.QuadraticTriangle((tri._a.id,tri._b.id,tri._c.id,ab,bc,ac))
        else
            new_tri  = JuAFEM.QuadraticTriangle((tri._a.id,tri._c.id,tri._b.id,ac,bc,ab))
        end
        push!(cells, new_tri)
    end
    #facesets = Dict{String,Set{Tuple{Int,Int}}}()#TODO:Does it make sense to add to this?
    #boundary_matrix = spzeros(Bool, 3, m)#TODO:Maybe treat the boundary correctly?
    #TODO: Fix below if this doesn't work
    grid = Grid(cells, nodes)#, facesets=facesets, boundary_matrix=boundary_matrix)
    return grid, locator

end

"""
    nodal_interpolation(ctx,f)

Perform nodal interpolation of a function. Returns a vector of coefficients in dof order
"""
function nodal_interpolation(ctx::gridContext, f::Function)
    nodal_values = [f(ctx.grid.nodes[ctx.dof_to_node[j]].x) for j in 1:ctx.n]
end

"""
    mutable struct boundaryData

Represent (a combination of) homogeneous Dirichlet and periodic boundary conditions.
Fields:
 - `dbc_dofs` list of dofs that should have homogeneous Dirichlet boundary conditions. Must be sorted.
 - `periodic_dofs_from` and `periodic_dofs_to` are both `Vector{Int}`. The former *must* be strictly increasing, both must be the same length. `periodic_dofs_from[i]` is identified with `periodic_dofs_to[i]`. `periodic_dofs_from[i]` must be strictly larger than `periodic_dofs_to[i]`. Multiple dofs can be identified with the same dof. If some dof is identified with another dof and one of them is in `dbc_dofs`, both points *must* be in `dbc_dofs`
"""
mutable struct boundaryData
    dbc_dofs::Vector{Int}
    periodic_dofs_from::Vector{Int}
    periodic_dofs_to::Vector{Int}
    function boundaryData(dbc_dofs::Vector{Int}=Vector{Int}(),periodic_dofs_from::Vector{Int}=Vector{Int}(), periodic_dofs_to::Vector{Int}=Vector{Int}())
        assert( length(periodic_dofs_from) == length(periodic_dofs_to))
        assert(issorted(dbc_dofs))
        assert(issorted(periodic_dofs_from))
        return new(dbc_dofs, periodic_dofs_from,periodic_dofs_to)
    end
    function boundaryData{dim}(ctx::gridContext{dim},predicate::Function,which_dbc=[])
        dbcs = getHomDBCS(ctx,which_dbc).dbc_dofs
        from, to = identifyPoints(ctx,predicate)
        return boundaryData(dbcs,from,to)
    end
end

"""
    getHomDBCS(ctx,which="all")

Return `boundaryData` object corresponding to homogeneous Dirichlet Boundary Conditions for a set of facesets.
`which="all"` is shorthand for `["left","right","top","bottom"]`.
"""
function getHomDBCS{dim}(ctx::gridContext{dim},which="all")
    dbcs = ConstraintHandler(ctx.dh)
    #TODO: See if newer version of JuAFEM export a "boundary" nodeset
    if which == "all"
        if dim == 2
            dbc = Dirichlet(:T,
                    union(getfaceset(ctx.grid, "left"),
                     getfaceset(ctx.grid, "right"),
                     getfaceset(ctx.grid, "top"),
                     getfaceset(ctx.grid, "bottom"),
                       ), (x,t)->0)
       else
            dbc = Dirichlet(:T,
                    union(getfaceset(ctx.grid, "left"),
                     getfaceset(ctx.grid, "right"),
                     getfaceset(ctx.grid, "top"),
                     getfaceset(ctx.grid, "bottom"),
                     getfaceset(ctx.grid, "front"),
                     getfaceset(ctx.grid, "back"),
                       ), (x,t)->0)
       end
   elseif isempty(which)
       return boundaryData(Vector{Int}())
   else
       dbc = JuAFEM.Dirichlet(:T,
               union([getfaceset(ctx.grid, str) for str in which]...)
               ,(x,t) -> 0
           )
   end
    add!(dbcs, dbc)
    close!(dbcs)
    update!(dbcs, 0.0)
    return boundaryData(dbcs.prescribed_dofs)
end

"""
    undoBCS(ctx,u,bdata)

Given a vector `u` in dof order with boundary conditions applied, return the corresponding
`u` in dof order without the boundary conditions.
"""
function undoBCS(ctx, u,bdata)
        n = ctx.n
        if length(bdata.dbc_dofs) == 0 && length(bdata.periodic_dofs_from) == 0
            return copy(u)
        end
        if n == length(u)
            error("u is already of length n, no need for undoBCS")
        end
        correspondsTo = BCTable(ctx,bdata)
        result = zeros(n)
        for i in 1:n
            if correspondsTo[i] == 0
                result[i] = 0.0
            else
                result[i] = u[correspondsTo[i]]
            end
        end
        return result
end

"""
    getDofCoordinates(ctx,dofindex)

Return the coordinates of the node corresponding to the dof with index `dofindex`
"""
function getDofCoordinates{dim}(ctx::gridContext{dim},dofindex::Int)
    return ctx.grid.nodes[ctx.dof_to_node[dofindex]].x
end

function BCTable{dim}(ctx::gridContext{dim},bdata::boundaryData)
    dbcs_prescribed_dofs=bdata.dbc_dofs
    periodic_dofs_from = bdata.periodic_dofs_from
    periodic_dofs_to = bdata.periodic_dofs_to
    n = ctx.n
    k = length(dbcs_prescribed_dofs)
    l = length(periodic_dofs_from)

    if dbcs_prescribed_dofs==nothing
        dbcs_prescribed_dofs = getHomDBCS(ctx).prescribed_dofs
    end
    if !issorted(dbcs_prescribed_dofs)
        error("DBCS are not sorted")
    end
    for i in 1:l
        if i != 1
            if periodic_dofs_from[i-1] >= periodic_dofs_from[i]
                error("periodic_dofs_from is not strictly increasing")
            end
        end
        if periodic_dofs_from[i] <= periodic_dofs_to[i]
            error("periodic_dofs_from[$i] ≦ periodic_dofs_to[$i]")
        end
    end
    correspondsTo = zeros(Int, n)
    dbc_ptr = 0
    boundary_ptr = 0
    skipcounter = 0
    for j in 1:n
        skipcounterincreased = false
        correspondsTo[j] = j - skipcounter
        jnew = j
        if boundary_ptr <l && periodic_dofs_from[boundary_ptr+1] == j
            jnew = periodic_dofs_to[boundary_ptr+1]
            boundary_ptr += 1
            if jnew != j
                skipcounter += 1
                skipcounterincreased = true
            end
        end
        if (dbc_ptr < k)  && (dbcs_prescribed_dofs[dbc_ptr + 1] == j)
            dbc_ptr += 1
            correspondsTo[j] = 0
            if !skipcounterincreased
                skipcounter += 1
            end
            continue
        end
        correspondsTo[j] =  correspondsTo[jnew]
    end
    return correspondsTo
end

#TODO: Make this more efficient
"""
    nDofs(ctx,bdata)

Get the number of dofs that are left after the boundary conditions in `bdata` have been applied.
"""
function nDofs{dim}(ctx::gridContext{dim},bdata::boundaryData)
    return length(unique(BCTable(ctx,bdata)))
end

"""
    applyBCS(ctx,K,bdata)

Apply the boundary conditions from `bdata` to the `ctx.n` by `ctx.n` sparse matrix `K`.
"""
function applyBCS{dim}(ctx::gridContext{dim},K,bdata::boundaryData)
    k = length(bdata.dbc_dofs)
    n = ctx.n

    correspondsTo = BCTable(ctx,bdata)
    new_n = length(unique(correspondsTo))
    if 0 ∈ correspondsTo
        new_n -= 1
    end
    if issparse(K)
        vals = nonzeros(K)
        rows = rowvals(K)

        #Make an empty sparse matrix
        I = Int[]
        sizehint!(I,length(rows))
        J = Int[]
        sizehint!(J,length(rows))
        vals = nonzeros(K)
        for j in 1:n
            if correspondsTo[j] == 0
                continue
            end
            for i in nzrange(K,j)
                row = rows[i]
                if correspondsTo[row] == 0
                    continue
                end
                push!(I,correspondsTo[j])
                push!(J,correspondsTo[row])
            end
        end
        V = zeros(length(I))
        #TODO: Find out if pairs (I,J) need to be unique
        Kres = sparse(I,J,V)

        for j = 1:n
            if correspondsTo[j] == 0
                continue
            end
            for i in nzrange(K,j)
                    row = rows[i]
                    if correspondsTo[row] == 0
                        continue
                    end
                    @inbounds Kres[correspondsTo[row],correspondsTo[j]] += vals[i]
            end
        end
        return Kres
    else
        Kres = zeros(new_n,new_n)
        for j = 1:n
            if correspondsTo[j] == 0
                continue
            end
            for i in 1:n
                if correspondsTo[i] == 0
                    continue
                end
                Kres[correspondsTo[i],correspondsTo[j]] = K[i,j]
            end
        end
        return Kres
    end
end

function identifyPoints{dim}(ctx::gridContext{dim},predicate)
    boundary_dofs = getHomDBCS(ctx).dbc_dofs
    identify_from = Int[]
    identify_to = Int[]
    for (index, i) in enumerate(boundary_dofs)
        for j in 1:(i-1)
            if predicate(getDofCoordinates(ctx,i),getDofCoordinates(ctx,j))
                push!(identify_from,i)
                push!(identify_to,j)
                break
            end
        end
    end
    return identify_from,identify_to
end
