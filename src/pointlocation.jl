
"""
    locatePoint(ctx, x)
Point location on grids.
Returns a tuple (coords, [nodes])
where coords gives the coordinates within the reference shape (e.g. standard simplex)
And [nodes] is the list of corresponding node ids, ordered in the order of the
corresponding shape functions from FEM's interpolation.jl file.
"""#
function locatePoint(ctx::GridContext{dim}, x::AbstractVector{T}) where {dim,T}
    dim == lenght(x) || throw(DimensionMismatch("point has dimension $(length(x)), grid has dimension $dim"))
    if dim == 2
        return locatePoint(ctx,Vec{2,T}((x[1], x[2])))
    elseif dim == 3
        return locatePoint(ctx,Vec{3,T}((x[1], x[2], x[3])))
    else
        throw(DomainError("Wrong dimension"))
    end
end
function locatePoint(ctx::GridContext{dim}, x::Vec{dim,T}) where {dim,T}
    return locatePoint(ctx.loc, ctx.grid, x)
end

struct Regular1dGridLocator{C}<:PointLocator where {C <: FEM.Cell}
    nx::Int
    LL::Vec{1,Float64}
    UR::Vec{1,Float64}
end

function locatePoint(loc::Regular1dGridLocator{S}, grid::FEM.Grid, x::Vec{1,T}) where {S,T}
    if x[1] > loc.UR[1] || x[1] < loc.LL[1]
        throw(DomainError("Not in domain"))
    end
    if isnan(x[1])
        throw(DomainError("NaN coordinates"))
    end
    n1, loc1 = gooddivrem((x[1] - loc.LL[1]) / (loc.UR[1] - loc.LL[1]) * (loc.nx - 1), 1.0)
    if n1 == (loc.nx - 1) #If we hit the right hand edge
        if loc1 ≈ 0.0
            n1 = loc.nx - 2
            loc1 += 1.0
        else
            throw(DomainError("Not in domain"))
        end
    end

    if S === FEM.Line
        ll = n1
        lr = ll + 1
        @assert lr < (2loc.nx + 1)
        return Vec{1,T}((2 * loc1 - 1,)), [ll + 1, lr + 1], (n1 + 1)
    elseif S === FEM.QuadraticLine
        ll = 2n1
        lr = 2n1 + 2
        lm = 2n1 +1
        return Vec{1,T}((2loc1-1,)), [ll + 1, lr + 1, lm + 1], (n1 + 1)
    else
        throw(AssertionError("Invalid 1D Cell type"))
    end
end


#For delaunay triangulations, we can use the tesselation
#object and the locate() function
struct DelaunayCellLocator <: PointLocator
    m::Int64
    scale_x::Float64
    scale_y::Float64
    minx::Float64
    miny::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
    extended_points::Vector{Vec{2,Float64}}
    point_number_table::Vector{Int}
    cell_number_table::Vector{Int}
end

function locatePoint(loc::DelaunayCellLocator, grid::FEM.Grid, x::Vec{2,Float64})
    #TODO: can this work for x that are not Float64?
    point_inbounds = NumberedPoint2D(VD.min_coord+(x[1]-loc.minx)*loc.scale_x, VD.min_coord+(x[2]-loc.miny)*loc.scale_y)
    if min(point_inbounds.x, point_inbounds.y) < VD.min_coord || max(point_inbounds.x,point_inbounds.y) > VD.max_coord
        throw(DomainError("point(s) outside of domain"))
    end
    t_index = VD.findindex(loc.tess, point_inbounds)
    t = loc.tess._trigs[t_index]
    if VD.isexternal(t)
        throw(DomainError("triangle outside of domain"))
    end
    v1::Vec{2} = loc.extended_points[t._b.id] - loc.extended_points[t._a.id]
    v2::Vec{2} = loc.extended_points[t._c.id] - loc.extended_points[t._a.id]
    J::Tensor{2,2,Float64,4} = Tensors.otimes(v1 , e1)  + Tensors.otimes(v2 , e2)
    #J = [v1[1] v2[1]; v1[2] v2[2]]
    #TODO: rewrite this so that we actually find the cell in question and get the ids
    #From there (instead of from the tesselation). Then get rid of the permutation that
    #is implicit below (See also comment below in P2DelaunayCellLocator locatePoint())
    return (inv(J) ⋅ (x - loc.extended_points[t._a.id])), loc.point_number_table[[t._b.id, t._c.id, t._a.id]], loc.cell_number_table[t_index]
end

#For delaunay triangulations with P2-Lagrange Elements
struct P2DelaunayCellLocator <: PointLocator
    m::Int64
    scale_x::Float64
    scale_y::Float64
    minx::Float64
    miny::Float64
    tess::VD.DelaunayTessellation2D{NumberedPoint2D}
    internal_triangles::Vector{Int}
    inv_internal_triangles::Vector{Int}
    point_number_table::Vector{Int}
    function P2DelaunayCellLocator(m,scale_x,scale_y,minx,miny,tess,point_number_table)
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
        res = new(m,scale_x,scale_y,minx,miny,tess,internal_triangles,inv_internal_triangles,point_number_table)
        return res
    end
end

function locatePoint(loc::P2DelaunayCellLocator, grid::FEM.Grid, x::Vec{2,Float64})
    #TODO: can this work for x that are not Float64?
    point_inbounds = NumberedPoint2D(VD.min_coord+(x[1]-loc.minx)*loc.scale_x,VD.min_coord+(x[2]-loc.miny)*loc.scale_y)
    if min(point_inbounds.x, point_inbounds.y) < VD.min_coord || max(point_inbounds.x,point_inbounds.y) > VD.max_coord
        throw(DomainError("point(s) outside of domain"))
    end
    t = VD.findindex(loc.tess, point_inbounds)
    if VD.isexternal(loc.tess._trigs[t])
        throw(DomainError("triangle outside of domain"))
    end
    qTriangle = grid.cells[loc.inv_internal_triangles[t]]
    v1::Vec{2} = grid.nodes[qTriangle.nodes[2]].x - grid.nodes[qTriangle.nodes[1]].x
    v2::Vec{2} = grid.nodes[qTriangle.nodes[3]].x - grid.nodes[qTriangle.nodes[1]].x
    J::Tensor{2,2,Float64,4} = Tensors.otimes(v1 , e1)  + Tensors.otimes(v2 , e2)
    #TODO: Think about whether doing it like this (with the permutation) is sensible
    return (inv(J) ⋅ (x - grid.nodes[qTriangle.nodes[1]].x)), loc.point_number_table[[permute!(collect(qTriangle.nodes),[2,3,1,5,6,4])]],1
end

#Here N gives the number of nodes and M gives the number of faces
struct Regular2DGridLocator{C} <: PointLocator where {C <: FEM.Cell}
    nx::Int
    ny::Int
    LL::Vec{2,Float64}
    UR::Vec{2,Float64}
end

function locatePoint(loc::Regular2DGridLocator{S}, grid::FEM.Grid, x::Vec{2,T})::Tuple{Vec{2,T}, Vector{Int}, Int} where {S,T}
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[1] < loc.LL[1] || x[2] < loc.LL[2]
        throw(DomainError("Not in domain"))
    end
    if isnan(x[1]) || isnan(x[2])
        throw(DomainError("NaN coordinates"))
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    n1::Int, loc1 = gooddivrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1), 1.0)
    n2::Int, loc2 = gooddivrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1), 1.0)

    if n1 == (loc.nx-1) #If we hit the right hand edge
        if loc1 ≈ 0.0
            n1 = loc.nx-2
            loc1 += 1.0
        else
            throw(DomainError("Not in domain"))
        end
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        if loc2 ≈ 0.0
            n2 = loc.ny-2
            loc2 += 1.0
        else
            throw(DomainError("Not in domain"))
        end
    end
    #Get the four node numbers of quadrilateral the point is in:

    if S === FEM.Triangle || S === FEM.Quadrilateral
        ll = n1 + n2*loc.nx
        lr = ll + 1
        ul = n1 + (n2+1)*loc.nx
        ur = ul + 1
        @assert ur < (loc.nx * loc.ny)

        if S === FEM.Triangle
            if loc1 + loc2 < 1.0 # ◺
                return Vec{2,T}((loc1, loc2)), [lr+1, ul+1, ll+1], (2*n1 + 2*n2*(loc.nx-1)) +1
            else # ◹
                #The transformation that maps ◹ (with bottom node at origin) to ◺ (with ll node at origin)
                #Does [0,1] ↦ [1,0] and [-1,1] ↦ [0,1]
                #So it has representation matrix (columnwise) [ [1,-1] | [1,0] ]
                tM = Tensor{2,2,Float64,4}((1.,-1.,1.,0.))
                return tM⋅Vec{2,T}((loc1-1,loc2)), [ ur+1, ul+1, lr+1], (2*n1 + 2*n2*(loc.nx-1)) +2
            end
        else # S == FEM.Quadrilateral
            return Vec{2,T}((2 * loc1 - 1, 2 * loc2 - 1)), [ll+1, lr+1, ur+1, ul+1], (ll+1)
        end
    elseif S === FEM.QuadraticTriangle || S === FEM.QuadraticQuadrilateral
        #Get the four node numbers of quadrilateral the point is in
        #Zero-indexing of the array here, so we need to +1 for everything being returned
        num_x_with_edge_nodes::Int = loc.nx  + loc.nx - 1
        num_y_with_edge_nodes::Int = loc.ny + loc.ny -1
        ll = 2*n1 + 2*n2*num_x_with_edge_nodes
        lr = ll + 2
        ul = 2*n1 + 2(n2+1)*num_x_with_edge_nodes
        ur = ul + 2
        middle_left =  2*n1 + (2*n2+1)*num_x_with_edge_nodes
        @assert ur < (num_x_with_edge_nodes*num_y_with_edge_nodes) #Sanity check
        if S === FEM.QuadraticTriangle
            if loc1 + loc2 <= 1.0 # ◺
                return Vec{2,T}((loc1,loc2)), [lr+1,ul+1,ll+1, middle_left+2, middle_left+1, ll+2], (2*n1 + 2*n2*(loc.nx-1) + 1)
            else # ◹
                #The transformation that maps ◹ (with bottom node at origin) to ◺ (with ll node at origin)
                #Does [0,1] ↦ [1,0] and [-1,1] ↦ [0,1]
                #So it has representation matrix (columnwise) [ [1,-1] | [1,0] ]
                tM = Tensor{2,2,Float64,4}((1.,-1.,1.,0.))
                return tM⋅Vec{2,T}((loc1-1,loc2)), [ ur+1, ul+1,lr+1,ul+2,middle_left+2, middle_left+3], (2*n1 + 2*n2*(loc.nx-1) + 2)
            end
        else # S === FEM.QuadraticQuadrilateral
            return Vec{2,T}((2*loc1-1,2*loc2-1)), [ll+1,lr+1,ur+1,ul+1,ll+2,middle_left+3, ul+2, middle_left+1,middle_left+2], (n1 + loc.nx*n2 + 1)
        end
    else
        throw(AssertionError("Case should not be reached"))
    end
end

struct Regular3DGridLocator{T} <: PointLocator where {T<:FEM.Cell{3}}
    nx::Int
    ny::Int
    nz::Int
    LL::Vec{3,Float64}
    UR::Vec{3,Float64}
end

function mydet(p1,p2,p3)
    M = zeros(3,3)
    M[:,1] = p1
    M[:,2] = p2
    M[:,3] = p3
    return det(M)
end

#TODO: Make this more robust
function in_tetrahedron(a,b,c,d,p)
    my0 = eps()
    return (mydet(b-a,c-a,p-a) >= -my0) && (mydet(b-a,d-a,p-a) <= my0) && (mydet(d-b,c-b,p-b) >= -my0) && (mydet(d-a,c-a,p-a) <= my0)
end

const standard_cube = [Vec{3}((0.,0.,0.)), Vec{3}((1.,0.,0.)), Vec{3}((1.,1.,0.)), Vec{3}((0.,1.,0.)),
    Vec{3}((0.,0.,1.)), Vec{3}((1.,0.,1.)), Vec{3}((1.,1.,1.)), Vec{3}((0.,1.,1.))]
const tetrahedra = [[1,2,4,8], [1,5,2,8], [2,3,4,8], [2,7,3,8], [2,5,6,8], [2,6,7,8]]

avg(x,y) = ((x == 1 && y == 3) || (x == 3 && y == 1)) ? 2 : x

indexavg(x,y) = CartesianIndex(avg.(Tuple(x),Tuple(y)))
const tetrahedra_3d =[ ((1,1,1),(3,1,1),(1,3,1),(1,3,3)),
                        ((1,1,1),(1,1,3),(3,1,1),(1,3,3)),
                        ((3,1,1),(3,3,1),(1,3,1),(1,3,3)),
                        ((3,1,1),(3,3,3),(3,3,1),(1,3,3)),
                        ((3,1,1),(1,1,3),(3,1,3),(1,3,3)),
                        ((3,1,1),(3,1,3),(3,3,3),(1,3,3)) ]

function locatePoint(
    loc::Regular3DGridLocator{S}, grid::FEM.Grid, x::Vec{3,T}
    ) where {T,S <: Union{FEM.Tetrahedron,FEM.QuadraticTetrahedron}}
    if x[1] > loc.UR[1]  || x[2] >  loc.UR[2] || x[3] > loc.UR[3] || x[1] < loc.LL[1] || x[2] < loc.LL[2] || x[3] < loc.LL[3]
        throw(DomainError("Not in domain"))
    end
    #Get integer and fractional part of coordinates
    #This is the lower left corner
    #warning: all the coputation is done with zero-indexing
    n1::Int,loc1 = gooddivrem((x[1] - loc.LL[1])/(loc.UR[1] - loc.LL[1]) * (loc.nx-1),1.0)
    n2::Int,loc2 = gooddivrem((x[2] - loc.LL[2])/(loc.UR[2] - loc.LL[2]) * (loc.ny-1),1.0)
    n3::Int,loc3 = gooddivrem((x[3] - loc.LL[3])/(loc.UR[3] - loc.LL[3]) * (loc.nz-1),1.0)

    if n1 == (loc.nx-1) #If we hit the right hand edge
        if loc1 ≈ 0.0
            n1 = loc.nx-2
            loc1 += 1.0
        else
            throw(DomainError("Not in domain"))
        end
    end
    if n2 == (loc.ny-1) #If we hit the top edge
        if loc2 ≈ 0.0
            n2 = loc.ny-2
            loc2 += 1.0
        else
            throw(DomainError("Not in domain"))
        end
    end

    if n3 == (loc.nz-1) #If we hit the top edge
        if loc3 ≈ 0.0
            n3 = loc.nz-2
            loc3 += 1.0
        else
            throw(DomainError("Not in domain"))
        end
    end
    #Get the 8 node numbers of the rectangular hexahedron the point is in:
    #Ordering is like tmp of FEM's generate_grid(::Type{Tetrahedron})

    i = n1+1
    j = n2+1
    k = n3+1

    if S === FEM.Tetrahedron
        node_array = reshape(collect(0:(loc.nx*loc.ny*loc.nz - 1)), (loc.nx, loc.ny, loc.nz))
        nodes = (node_array[i,j,k], node_array[i+1,j,k], node_array[i+1,j+1,k], node_array[i,j+1,k],
                   node_array[i,j,k+1], node_array[i+1,j,k+1], node_array[i+1,j+1,k+1], node_array[i,j+1,k+1])
    else
        node_array = reshape(
           collect(0:( (2*loc.nx-1)*(2*loc.ny-1)*(2*loc.nz - 1) - 1)), (2*loc.nx - 1, 2*loc.ny - 1,2*loc.nz-1)
           )
        #TODO: does this cause a type instability?
        #TODO: Finish this.
        nodes = node_array[(2*(i-1) + 1):(2*i +1), (2*(j-1) + 1):(2*j+1), (2*(k-1)+1):(2*k+1)]
    end

    for (index, tet) in enumerate(tetrahedra)
        p1, p2, p3, p4 = standard_cube[tet]
        if in_tetrahedron(p1, p2, p3, p4, [loc1, loc2, loc3])
            M = zeros(3,3)
            M[:,1] = p2-p1
            M[:,2] = p3-p1
            M[:,3] = p4-p1
            tMI::Tensor{2,3,Float64,9} = Tensor{2,3,Float64}(M)
            if S === FEM.Tetrahedron
                return inv(tMI) ⋅ Vec{3,T}((loc1,loc2,loc3) .- (p1[1],p1[2],p1[3])), collect(nodes[tet]) .+ 1, 1
            else
                v1,v2,v3,v4 =  map(CartesianIndex, tetrahedra_3d[index])
                resulting_nodes = [nodes[v1]
                                   nodes[v2]
                                   nodes[v3]
                                   nodes[v4]
                                   nodes[indexavg(v1,v2)]
                                   nodes[indexavg(v2,v3)]
                                   nodes[indexavg(v1,v3)]
                                   nodes[indexavg(v1,v4)]
                                   nodes[indexavg(v2,v4)]
                                   nodes[indexavg(v3,v4)]
                                   ]
                return inv(tMI) ⋅ Vec{3,T}((loc1, loc2, loc3) .- (p1[1], p1[2], p1[3])), (resulting_nodes.+1), 1
            end
        end
    end
    throw(DomainError("Not in domain (could be a bug/rounding error)")) #In case we didn't land in any tetrahedron
end
