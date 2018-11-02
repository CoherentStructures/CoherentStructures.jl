#(c) 2018 Nathanael Schilling
# Code for dealing with combinations of homogeneous Dirichlet and periodic
# boundary conditions for stiffness/mass matrices.


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
        @assert length(periodic_dofs_from) == length(periodic_dofs_to)
        @assert issorted(dbc_dofs)
        @assert issorted(periodic_dofs_from)
        return new(dbc_dofs, periodic_dofs_from, periodic_dofs_to)
    end
    function boundaryData(ctx::gridContext{dim}, predicate::T, which_dbc=[]) where {dim, T <: Union{Function,Distances.Metric}}
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
function getHomDBCS(ctx::gridContext{dim}, which="all") where dim
    dbcs = JuAFEM.ConstraintHandler(ctx.dh)
    #TODO: See if newer version of JuAFEM export a "boundary" nodeset
    if which == "all"
        if dim == 1
            dbc = JuAFEM.Dirichlet(:T,
                    union(JuAFEM.getfaceset(ctx.grid, "left"),
                     JuAFEM.getfaceset(ctx.grid, "right")
                       ), (x,t)->0)
        elseif dim == 2
            dbc = JuAFEM.Dirichlet(:T,
                    union(JuAFEM.getfaceset(ctx.grid, "left"),
                     JuAFEM.getfaceset(ctx.grid, "right"),
                     JuAFEM.getfaceset(ctx.grid, "top"),
                     JuAFEM.getfaceset(ctx.grid, "bottom"),
                       ), (x,t)->0)
       elseif dim == 3
            dbc = JuAFEM.Dirichlet(:T,
                    union(JuAFEM.getfaceset(ctx.grid, "left"),
                     JuAFEM.getfaceset(ctx.grid, "right"),
                     JuAFEM.getfaceset(ctx.grid, "top"),
                     JuAFEM.getfaceset(ctx.grid, "bottom"),
                     JuAFEM.getfaceset(ctx.grid, "front"),
                     JuAFEM.getfaceset(ctx.grid, "back"),
                       ), (x,t)->0)
       else
           throw(AssertionError("dim ∉ [1,2,3]"))
       end
   elseif isempty(which)
       return boundaryData(Vector{Int}())
   else
       dbc = JuAFEM.Dirichlet(:T,
               union([JuAFEM.getfaceset(ctx.grid, str) for str in which]...)
               ,(x,t) -> 0
           )
   end
    JuAFEM.add!(dbcs, dbc)
    JuAFEM.close!(dbcs)
    JuAFEM.update!(dbcs, 0.0)
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
            if correspondsTo[i] != 0
                result[i] = u[correspondsTo[i]]
            end
        end
        return result
end

"""
    getDofCoordinates(ctx,dofindex)

Return the coordinates of the node corresponding to the dof with index `dofindex`
"""
function getDofCoordinates(ctx::gridContext{dim},dofindex::Int) where dim
    return ctx.grid.nodes[ctx.dof_to_node[dofindex]].x
end

function BCTable(ctx::gridContext{dim},bdata::boundaryData) where dim
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
        if boundary_ptr < l && periodic_dofs_from[boundary_ptr+1] == j
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
function nDofs(ctx::gridContext{dim},bdata::boundaryData) where dim
    return length(unique(BCTable(ctx,bdata)))
end

"""
    doBCS(ctx,u,bdata)

Take a vector `u` in dof order and throw away uneccessary dofs.
This is a left-inverse to undoBCS
"""
function doBCS(ctx, u::AbstractVector{T}, bdata) where T
    @assert length(u) == ctx.n
    #Is = findall(i -> ∉(i,bdata.dbc_dofs) && ∉(i,bdata.periodic_dofs_from), 1:ctx.n)
    #return u[Is]
     result = T[]
     for i in 1:ctx.n
         if i in bdata.dbc_dofs
             continue
         end
         if i in bdata.periodic_dofs_from
             continue
         end
         push!(result,u[i])
     end
     return result
end

"""
    applyBCS(ctx_row,K,bdata_row; [ctx_col, bdata_col,bdata_row,add_vals=true])

Apply the boundary conditions from `bdata_row` and `bdata_col` to the sparse matrix `K`.
Only applies boundary conditions accross columns (rows) if `bdata_row==nothing` (`bdata_col==nothing`)
If `add_vals==true`, then
"""
function applyBCS(ctx_row::gridContext{dim},K,bdata_row;
        ctx_col::gridContext{dim}=ctx_row, bdata_col=bdata_row,
        add_vals = true
        ) where dim

    n,m = size(K)

    is_symmetric = issymmetric(K) && (bdata_row == bdata_col)

    if bdata_col != nothing
        correspondsTo_col = BCTable(ctx_col,bdata_col)
    else
        correspondsTo_col = collect(1:(size(K)[2]))
    end

    if bdata_row != nothing
        correspondsTo_row = BCTable(ctx_row,bdata_row)
    else
        correspondsTo_row = collect(1:(size(K)[1]))
    end

    new_n = length(unique(correspondsTo_row))
    new_m = length(unique(correspondsTo_col))

    if 0 ∈ correspondsTo_col
        new_n -= 1
    end
    if 0 ∈ correspondsTo_row
        new_m -= 1
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
        V = Float64[]
        sizehint!(V,length(rows))
        for j = 1:m
            if correspondsTo_col[j] == 0
                continue
            end
            for i in nzrange(K,j)
                row = rows[i]
                if correspondsTo_row[row] == 0
                    continue
                end
                if is_symmetric && (correspondsTo_row[j] < correspondsTo_row[row])
                    continue
                end
                push!(I, correspondsTo_row[row])
                push!(J, correspondsTo_col[j])
                push!(V,vals[i])
                if is_symmetric && correspondsTo_row[row] != correspondsTo_row[j]
                    push!(J, correspondsTo_row[row])
                    push!(I, correspondsTo_col[j])
                    push!(V,vals[i])
                end
            end
        end
        if add_vals
            Kres = sparse(I,J,V,new_n,new_m)
        else
            Kres = sparse(I,J,V,new_n,new_m,(x,y) -> x)
        end
        return Kres
    else
        Kres = zeros(new_n,new_m)
        for j = 1:m
            if correspondsTo_col[j] == 0
                continue
            end
            for i in 1:n
                if correspondsTo_row[i] == 0
                    continue
                end
                if add_vals
                    Kres[correspondsTo_row[i],correspondsTo_col[j]] += K[i,j]
                else
                    Kres[correspondsTo_row[i],correspondsTo_col[j]] = K[i,j]
                end
            end
        end
        return Kres
    end
end

function identifyPoints(ctx::gridContext{dim},predicate::Function) where dim
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


function identifyPoints(ctx::gridContext{dim},predicate::Distances.Metric) where dim

    identify_from = Int[]
    identify_to = Int[]

    l = zeros(dim, ctx.n)
    for i in 1:ctx.n
        coords = getDofCoordinates(ctx,i)
        l[:, i] .= coords
    end
    TOL = 1e-12 #TODO: set this somewhere globally

    S = NN.BallTree(l,predicate)

    for index in 1:ctx.n
        res = NN.inrange(S,getDofCoordinates(ctx,index),TOL,true)
        if res[1] != index
            push!(identify_from,index)
            push!(identify_to,res[1])
        end
    end
    return identify_from,identify_to
end
