# (c) 2018-19 Daniel Karrasch & Alvaro de Diego

##############
# spatiotemporal, time averaged metrics
##############
"""
    STmetric(metric, p)

Creates a spatiotemporal, averaged-in-time metric. At each time instance, the
distance between two states `a` and `b` is computed via `metric(a, b)`.
The resulting distances are subsequently ``ℓ^p``-averaged, with ``p=`` `p`.

## Fields
   * `metric=Euclidean()`: a `SemiMetric` as defined in the `Distances.jl` package,
   e.g., [`Euclidean`](@ref Distances.Euclidean),
   [`PeriodicEuclidean`](@ref Distances.PeriodicEuclidean), or
   [`Haversine`](@ref Distances.Haversine);
      * `p = Inf`: maximum
      * `p = 2`: mean squared average
      * `p = 1`: arithmetic mean
      * `p = -1`: harmonic mean (does not yield a metric!)
      * `p = -Inf`: minimum (does not yield a metric!)

## Example
```jldoctest; setup=(using Distances, StaticArrays)
julia> x = [@SVector rand(2) for _ in 1:10];

julia> y = [@SVector rand(2) for _ in 1:10];

julia> d = STmetric(Euclidean(), 1) # Euclidean distances arithmetically averaged
STmetric{Euclidean,Int64}(Euclidean(0.0), 1)

julia> d(x, y) ≈ sum(Euclidean().(x, y))/10
true

julia> d = STmetric(Euclidean(), 2);

julia> d(x, y) ≈ sqrt(sum(abs2, Euclidean().(x, y))/10)
true
```
"""
struct STmetric{M<:Dists.SemiMetric,T<:Real} <: Dists.SemiMetric
    metric::M
    p::T
end

# defaults: metric = Euclidean(), dim = 2, p = 1
STmetric(d = Dists.Euclidean()) = STmetric(d, 1)

function (dist::STmetric)(a::AbstractArray, b::AbstractArray)
    return Dists._evaluate(dist, a, b, nothing)
end

function Dists.result_type(
    d::STmetric,
    a::AbstractVector{S},
    b::AbstractVector{S},
) where {S}
    T = Dists.result_type(d.metric, zero(S), zero(S))
    return typeof(Dists.eval_end(d, Dists.eval_reduce(d, zero(T), zero(T)), 2))
end

Base.@propagate_inbounds function Dists._evaluate(d::STmetric, a::AbstractArray, b::AbstractArray, ::Nothing)
    n = length(a)
    @boundscheck if n != length(b)
        throw(DimensionMismatch("first array has length $n which does not match the length of the second, $(length(b))."))
    end
    if n == 0
        return zero(Dists.result_type(d, a, b))
    end
    @inbounds begin
        s = Dists.eval_start(d, a, b)
        if (IndexStyle(a, b) === IndexLinear() && eachindex(a) == eachindex(b)) || axes(a) == axes(b)
            @simd for I in eachindex(a, b)
                ai = a[I]
                bi = b[I]
                s = Dists.eval_reduce(d, s, d.metric(ai, bi))
            end
        else
            for (Ia, Ib) in zip(eachindex(a), eachindex(b))
                ai = a[Ia]
                bi = b[Ib]
                s = Dists.eval_reduce(d, s, d.metric(ai, bi))
            end
        end
        return Dists.eval_end(d, s, n)
    end
end

@inline Dists.eval_start(d::STmetric, a, b) = d.p == -Inf ? typemax(Dists.result_type(d, a, b)) : zero(Dists.result_type(d, a, b))

@inline function Dists.eval_reduce(d::STmetric, s1, s2)
    p = d.p
    if p == 1
        return s1 + s2
    elseif p == 2
        return s1 + s2 * s2
    elseif p == -2
        return s1 + 1 / (s2 * s2)
    elseif p == -1
        return s1 + 1 / s2
    elseif p == Inf
        return max(s1, s2)
    elseif p == -Inf
        return min(s1, s2)
    else
        return s1 + s2^p
    end
end

@inline function Dists.eval_end(d::STmetric, s, n)
    p = d.p
    isinf(p) && return s
    t = s/n
    if p == 1
        return t
    elseif p == 2
        return √t
    elseif p == -2
        return 1 / √t
    elseif p == -1
        return 1 / t
    else
        return t^(1 / p)
    end
end

function stmetric(
    a::AbstractVector{T},
    b::AbstractVector{T},
    d::Dists.SemiMetric = Dists.Euclidean(),
    p::Real = 1,
) where {T<:SVector}
    return STmetric(d, p)(a, b)
end

function Dists.pairwise!(
    r::AbstractMatrix,
    metric::STmetric,
    a::AbstractVector{<:AbstractVector{T}},
    b::AbstractVector{<:AbstractVector{T}};
    dims::Union{Nothing,Integer} = nothing,
) where {T<:SVector}
    la, lb = length.((a, b))
    size(r) == (la, lb) || throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((na, nb)))."))
    @inbounds for j in 1:lb
        bj = b[j]
        for i in 1:la
            r[i, j] = metric(a[i], bj)
        end
    end
    r
end
function Dists.pairwise!(
    r::AbstractMatrix,
    metric::STmetric,
    a::AbstractVector{<:AbstractVector{<:SVector}};
    dims::Union{Nothing,Integer} = nothing,
)
    la = length(a)
    size(r) == (la, la) || throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((n, n)))."))
    @inbounds for j in 1:la
        aj = a[j]
        for i in (j+1):la
            r[i, j] = metric(a[i], aj)
        end
        r[j, j] = 0
        for i in 1:(j-1)
            r[i, j] = r[j, i]
        end
    end
    r
end

function Dists.pairwise(
    metric::STmetric,
    a::AbstractVector{<:AbstractVector{T}},
    b::AbstractVector{<:AbstractVector{T}};
    dims::Union{Nothing,Integer} = nothing,
) where {T<:SVector}
    la = length(a)
    lb = length(b)
    (la == 0 || lb == 0) && return reshape(Float64[], 0, 0)
    r = Matrix{Dists.result_type(metric, first(a), first(b))}(undef, la, lb)
    Dists.pairwise!(r, metric, a, b)
end
function Dists.pairwise(
    metric::STmetric,
    a::AbstractVector{<:AbstractVector{<:SVector}};
    dims::Union{Nothing,Integer} = nothing,
)
    la = length(a)
    la == 0 && return reshape(Float64[], 0, 0)
    r = Matrix{Dists.result_type(metric, first(a), first(a))}(undef, la, la)
    Dists.pairwise!(r, metric, a)
end

function Dists.colwise!(
    r::AbstractVector,
    metric::STmetric,
    a::AbstractVector{T},
    b::AbstractVector{<:AbstractVector{T}},
) where {T<:SVector}
    lb = length(b)
    length(r) == lb || throw(DimensionMismatch("incorrect length of r (got $(length(r)), expected $lb)"))
    @inbounds for i in eachindex(b, r)
        r[i] = metric(a, b[i])
    end
    r
end
function Dists.colwise!(
    r::AbstractVector,
    metric::STmetric,
    a::AbstractVector{<:AbstractVector{T}},
    b::AbstractVector{T},
) where {T<:SVector}
    Dists.colwise!(r, metric, b, a)
end
function Dists.colwise!(
    r::AbstractVector,
    metric::STmetric,
    a::T,
    b::T,
) where {T<:AbstractVector{<:AbstractVector{<:SVector}}}
    la = length(a)
    lb = length(b)
    la == lb || throw(DimensionMismatch("lengths of a, $la, and b, $lb, do not match"))
    la == length(r) || throw(DimensionMismatch("incorrect size of r, got $(length(r)), but expected $la"))
    @inbounds for i in eachindex(a, b, r)
        r[i] = metric(a[i], b[i])
    end
    r
end

function Dists.colwise(
    metric::STmetric,
    a::AbstractVector{T},
    b::AbstractVector{<:AbstractVector{T}},
) where {T<:SVector}
    Dists.colwise!(
        Vector{Dists.result_type(metric, a, first(b))}(undef, length(b)),
        metric,
        a,
        b,
    )
end
function Dists.colwise(
    metric::STmetric,
    a::AbstractVector{<:AbstractVector{T}},
    b::AbstractVector{T},
) where {T<:SVector}
    return Dists.colwise(metric, b, a)
end
function Dists.colwise(
    metric::STmetric,
    a::T,
    b::T,
) where {T<:AbstractVector{<:AbstractVector{<:SVector}}}
    la = length(a)
    lb = length(b)
    la == lb || throw(DimensionMismatch("lengths of a, $la, and b, $lb, do not match"))
    la == 0 && return reshape(Float64[], 0, 0)
    Dists.colwise!(
        Vector{Dists.result_type(metric, first(a), first(b))}(undef, la),
        metric,
        a,
        b,
    )
end

################## sparse pairwise distance computation ###################
"""
    spdist(data, sp_method, metric=Euclidean()) -> SparseMatrixCSC

Return a sparse distance matrix as determined by the sparsification method `sp_method`
and `metric`.
"""
function spdist(
    data::AbstractArray{<:SVector},
    sp_method::Neighborhood,
    metric::Dists.PreMetric = Distances.Euclidean(),
)
    N = length(data) # number of states
    data = vec(data)
    # TODO: check for better leafsize values
    tree =
        metric isa NN.MinkowskiMetric ? NN.KDTree(data, metric; leafsize = 10) :
        NN.BallTree(data, metric; leafsize = 10)
    idxs = NN.inrange(tree, data, sp_method.ε, false)
    Js = vcat(idxs...)
    Is = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
    Vs = fill(1.0, length(Is))
    return sparse(Is, Js, Vs, N, N, *)
end
function spdist(
    data::AbstractVector{<:SVector},
    sp_method::Union{KNN,MutualKNN},
    metric::Dists.PreMetric = Distances.Euclidean(),
)
    N = length(data) # number of states
    data = vec(data)
    # TODO: check for better leafsize values
    tree =
        metric isa NN.MinkowskiMetric ? NN.KDTree(data, metric; leafsize = 10) :
        NN.BallTree(data, metric; leafsize = 10)
    idxs, dists = NN.knn(tree, data, sp_method.k, false)
    Js = vcat(idxs...)
    Is = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
    Ds = vcat(dists...)
    D = sparse(Is, Js, Ds, N, N)
    SparseArrays.dropzeros!(D)
    if sp_method isa KNN
        return max.(D, permutedims(D))
    else # sp_method isa MutualKNN
        return min.(D, permutedims(D))
    end
end
function spdist(
    data::AbstractArray{<:AbstractArray{<:SVector}},
    sp_method::Neighborhood,
    metric::STmetric,
)
    N = length(data) # number of trajectories
    T = Dists.result_type(metric, first(data), first(data))
    I1 = collect(1:N)
    J1 = collect(1:N)
    V1 = zeros(T, N)
    tempfun(j) = begin
        Is, Js, Vs = Int[], Int[], T[]
        aj = data[j]
        for i in (j+1):N
            temp = metric(data[i], aj)
            if temp < sp_method.ε
                push!(Is, i, j)
                push!(Js, j, i)
                push!(Vs, temp, temp)
            end
        end
        return Is, Js, Vs
    end
    if Distributed.nprocs() > 1
        IJV = Distributed.pmap(
            tempfun,
            1:(N-1);
            batch_size = (N ÷ Distributed.nprocs()^2),
        )
    else
        IJV = map(tempfun, 1:(N-1))
    end
    Is = vcat(I1, getindex.(IJV, 1)...)
    Js = vcat(J1, getindex.(IJV, 2)...)
    Vs = vcat(V1, getindex.(IJV, 3)...)
    return sparse(Is, Js, Vs, N, N)
end
function spdist(
    data::AbstractArray{<:AbstractArray{<:SVector}},
    sp_method::Union{KNN,MutualKNN},
    metric::STmetric,
)
    data = vec(data)
    N, k = length(data), sp_method.k
    Is = repeat(1:N, inner=k+1)
    Js = SharedArray{Int}(N * (k + 1))
    T = typeof(metric(data[1], data[1]))
    Ds = SharedArray{T}(N * (k + 1))
    perm = collect(1:N)
    ds = Vector{T}(undef, N)
    # Distributed.@everywhere index = Vector{Int}(undef, k+1)
    @inbounds @sync Distributed.@distributed for i in 1:N
        Dists.colwise!(ds, metric, data[i], data)
        index = partialsortperm!(perm, ds, 1:(k+1), initialized = true)
        Js[(i-1)*(k+1)+1:i*(k+1)] = index
        Ds[(i-1)*(k+1)+1:i*(k+1)] = ds[index]
    end
    D = sparse(Is, Js, Ds, N, N)
    # SparseArrays.dropzeros!(D)
    if sp_method isa KNN
        return max.(D, permutedims(D))
    else # sp_method isa MutualKNN
        return min.(D, permutedims(D))
    end
end

########### parallel pairwise computation #################

# function Dists.pairwise!(r::SharedMatrix{T}, d::STmetric, a::AbstractMatrix, b::AbstractMatrix) where T <: Real
#     ma, na = size(a)
#     mb, nb = size(b)
#     size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
#     ma == mb || throw(DimensionMismatch("First and second array have different numbers of time instances."))
#     q, s = divrem(ma, d.dim)
#     s == 0 || throw(DimensionMismatch("Number of rows is not a multiple of spatial dimension $(d.dim)."))
#     dists = Vector{T}(q)
#     Distributed.@everywhere dists = $dists
#     @inbounds @sync Distributed.@distributed for j = 1:nb
#         for i = 1:na
#             eval_space!(dists, d, view(a, :, i), view(b, :, j), q)
#             r[i, j] = reduce_time(d, dists, q)
#         end
#     end
#     r
# end
#
# function Dists.pairwise!(r::SharedMatrix{T}, d::STmetric, a::AbstractMatrix) where T <: Real
#     m, n = size(a)
#     size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
#     q, s = divrem(m, d.dim)
#     s == 0 || throw(DimensionMismatch("Number of rows is not a multiple of spatial dimension $(d.dim)."))
#     entries = div(n*(n+1),2)
#     dists = Vector{T}(undef, q)
#     i, j = 1, 1
#     Distributed.@everywhere dists, i, j = $dists, $i, $j
#     @inbounds @sync Distributed.@distributed for k = 1:entries
#         tri_indices!(i, j, k)
#         if i == j
#             r[i, i] = zero(T)
#         else
#             eval_space!(dists, d, view(a, :, i), view(a, :, j), q)
#             r[i, j] = reduce_time(d, dists, q)
#         end
#     end
#     for j = 1:n
#         for i=1:(j-1)
#             r[i, j] = r[j, i]
#         end
#     end
#     r
# end
#
# function Dists.pairwise(metric::STmetric, a::AbstractMatrix, b::AbstractMatrix)
#     m = size(a, 2)
#     n = size(b, 2)
#     r = SharedArray{Dists.result_type(metric, a, b)}(m, n)
#     Dists.pairwise!(r, metric, a, b)
# end
#
# function Dists.pairwise(metric::STmetric, a::AbstractMatrix)
#     n = size(a, 2)
#     r = SharedArray{Dists.result_type(metric, a, a)}(n, n)
#     Dists.pairwise!(r, metric, a)
# end
#
# @inline function tri_indices!(i::Int, j::Int, n::Int)
#     i = floor(Int, 0.5 * (1 + sqrt(8n - 7)))
#     j = n - div(i * (i - 1), 2)
#     return i, j
# end
