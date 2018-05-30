# (c) 2018 Alvaro de Diego & Daniel Karrasch

import Distances: result_type, evaluate, eval_start, eval_op, eval_reduce, eval_end, pairwise, pairwise!

const Dists = Distances
struct PEuclidean{W <: Dists.RealAbstractArray} <: Dists.Metric
    periods::W
end

"""
    PEuclidean(L)
Create a Euclidean metric on a rectangular periodic domain.
Periods per dimension are contained in the vector `L`.
For dimensions without periodicity put `Inf` in the respective component.

# Usage
```@jldoctest
julia> x, y, L = rand(2), rand(2), [0.5, Inf]
([0.551552, 0.38173], [0.57283, 0.874754], [0.5, Inf])

julia> Distances.evaluate(PEuclidean(L),x,y)
0.49348274281688614
"""

PEuclidean() = Dists.Euclidean()

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function evaluate(d::PEuclidean, a::Union{Array, Dists.ArraySlice}, b::Union{Array, Dists.ArraySlice})
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(d.periods)
        throw(DimensionMismatch("arrays have length $(length(a)) but periods have length $(length(d.periods))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        @simd for I in eachindex(a, b, d.periods)
            ai = a[I]
            bi = b[I]
            li = d.periods[I]
            s = eval_reduce(d, s, eval_op(d, ai, bi, li))
        end
        return eval_end(d, s)
    end
end

@inline function evaluate(d::PEuclidean, a::AbstractArray, b::AbstractArray)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(d.periods)
        throw(DimensionMismatch("arrays have length $(length(a)) but periods have length $(length(d.periods))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if size(a) == size(b)
            @simd for I in eachindex(a, b, d.periods)
                ai = a[I]
                bi = b[I]
                li = d.periods[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi, li))
            end
        else
            for (Ia, Ib, Ip) in zip(eachindex(a), eachindex(b), eachindex(d.weights))
                ai = a[Ia]
                bi = b[Ib]
                li = d.periods[Ip]
                s = eval_reduce(d, s, eval_op(d, ai, bi, li))
            end
        end
    end
    return eval_end(d, s)
end

function evaluate(dist::PEuclidean, a::T, b::T) where {T <: Number}
    eval_end(dist, eval_op(dist, a, b, one(eltype(dist.periods))))
end
function result_type(dist::PEuclidean, ::AbstractArray{T1}, ::AbstractArray{T2}) where {T1, T2}
    typeof(evaluate(dist, one(T1), one(T2)))
end
@inline function eval_start(d::PEuclidean, a::AbstractArray, b::AbstractArray)
    zero(result_type(d, a, b))
end
@inline eval_op(::PEuclidean, ai, bi, li) = begin d = mod(abs(ai - bi), li); d = min(d, li-d); abs2(d) end
@inline eval_reduce(::PEuclidean, s1, s2) = s1 + s2
@inline eval_end(::PEuclidean, s) = sqrt(s)

peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray) = evaluate(PEuclidean(p), a, b)
peuclidean(a::AbstractArray, b::AbstractArray) = evaluate(PEuclidean(), a, b)
peuclidean(a::Number, b::Number, p::Number) = begin d = mod(a - b, p); d = min(d, p - d) end

########## spatiotemporal, time averaged metrics ##############

"""
    STmetric(Smetric, dim, p)

Creates a spatiotemporal, averaged in time metric.

# Properties

   * `Smetric` is a metric as defined in the `Dists` package, e.g.,
     `Euclidean`, `PEuclidean`, or `Haversine`
   * `dim` corresponds to the spatial dimension
   * `p` corresponds to the kind of average applied to the vector of spatial Dists:
     - `p = Inf`: maximum
     - `p = 2`: mean squared average
     - `p = 1`: arithmetic mean
     - `p = -1`: harmonic mean (does not yield a metric!)
     - `p = -Inf`: minimum (does not yield a metric!)

# Usage
```@jldoctest
julia> x, y = rand(10), rand(10)
([0.0645218, 0.824624, 0.723568, 0.786856, 0.529069, 0.666899, 0.956035, 0.960833, 0.753796, 0.319134], [0.372017, 0.838669, 0.873848, 0.253589, 0.724321, 0.862853, 0.958319, 0.0306237, 0.352692, 0.169052])

julia> Distances.evaluate(STmetric(Distances.Euclidean(),2,1),x,y)
2.4969539623437083
```
"""

struct STmetric{T <: Real, M <: Dists.Metric} <: Dists.Metric
    Smetric::M
    dim::Int
    p::T
end

STmetric()          = STmetric(Dists.Euclidean(), 2, 1)
STmetric(p::Real)   = STmetric(Dists.Euclidean(), 2, p)

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function evaluate(d::STmetric, a::Union{Array, Dists.ArraySlice}, b::Union{Array, Dists.ArraySlice})
    la = length(a)
    lb = length(b)
    (q, r) = divrem(la,d.dim)
    p = copy(d.p)
    @boundscheck if la != lb
        throw(DimensionMismatch("First array has size $(size(a)) which does not match the size of the second, $(size(b))."))
    elseif r!= 0
        throw(DimensionMismatch("Number of rows is not a multiple of spatial dimension $(d.dim)."))
    end
    if la == 0
        return zero(result_type(d, a, b))
    end
    return reduce_time(d, eval_space(d, a, b, d.Smetric, d.dim, q), d.p, q)
end

# this version is needed for NearestNeighbors `NNTree`
@inline function evaluate(d::STmetric, a::AbstractArray, b::AbstractArray)
    la = length(a)
    lb = length(b)
    (q, r) = divrem(la,d.dim)
    @boundscheck if la != lb
        throw(DimensionMismatch("First array has size $(size(a)) which does not match the size of the second, $(size(b))."))
    elseif r!= 0
        throw(DimensionMismatch("Number of rows is not a multiple of spatial dimension $(d.dim)."))
    end
    if la == 0
        return zero(result_type(d, a, b))
    end
    return reduce_time(d, eval_space(d, a, b, d.Smetric, d.dim, q), d.p, q)
end

function result_type(d::STmetric, a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1, T2}
    result_type(d.Smetric, a, b)
end

@inline eval_space(::STmetric,
                   a::AbstractArray,
                   b::AbstractArray,
                   sm::Distances.Metric,
                   dim::Int, q::Int) =
   Distances.colwise(sm, reshape(a, dim, q), reshape(b, dim, q))

@inline reduce_time(::STmetric, s, p, q) = q^(-inv(p)) * vecnorm(s, p)

stmetric(a::AbstractArray, b::AbstractArray, d::Dists.PreMetric, dim::Int, p::Real) = evaluate(STmetric(d, dim, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, d::Dists.PreMetric, p::Real) = evaluate(STmetric(d, 2, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, p::Real) = evaluate(STmetric(p), a, b)
stmetric(a::AbstractArray, b::AbstractArray) = evaluate(STmetric(), a, b)

########### parallel pairwise computation #################

function pairwise!(r::SharedMatrix{T}, d::STmetric, a::AbstractMatrix, b::AbstractMatrix) where T <: Real
    ma, na = size(a)
    mb, nb = size(b)
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    ma == mb || throw(DimensionMismatch("First and second array have different numbers of time instances."))
    q, s = divrem(ma, d.dim)
    s == 0 || throw(DimensionMismatch("Number of rows is not a multiple of spatial dimension $(d.dim)."))
    dists = Vector{T}(q)
    @everywhere @eval dists = $(dists)
    @inbounds @sync @parallel for j = 1:nb
        for i = 1:na
            dists .= eval_space(d, view(a, :, i), view(b, :, j), d.Smetric, d.dim, q)
            r[i, j] = reduce_time(d, dists, d.p)
        end
    end
    r
end

function pairwise!(r::SharedMatrix{T}, d::STmetric, a::AbstractMatrix) where T <: Real
    m, n = size(a)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    q, s = divrem(m, d.dim)
    s == 0 || throw(DimensionMismatch("Number of rows is not a multiple of spatial dimension $(d.dim)."))
    entries = div(n*(n+1),2)
    dists = Vector{T}(q)
    @everywhere @eval dists = $(dists)
    @inbounds @sync @parallel for k = 1:entries
        i, j = tri_indices(k)
        if i == j
            r[i, i] = zero(T)
        else
            dists .= eval_space(d, view(a, :, i), view(a, :, j), d.Smetric, d.dim, q)
            r[i, j] = reduce_time(d, dists, d.p)
        end
    end
    for j = 1:n
        for i=1:(j-1)
            r[i, j] = r[j, i]
        end
    end
    r
end

function pairwise(metric::STmetric, a::AbstractMatrix, b::AbstractMatrix)
    m = size(a, 2)
    n = size(b, 2)
    r = SharedMatrix{result_type(metric, a, b)}(m, n) #(uninitialized, m, n)
    pairwise!(r, metric, a, b)
end

function pairwise(metric::STmetric, a::AbstractMatrix)
    n = size(a, 2)
    r = SharedMatrix{result_type(metric, a, a)}(n, n) #(uninitialized, n, n)
    pairwise!(r, metric, a)
end

@inline function tri_indices(n::Int)
    i = floor(Int, 0.5*(1 + sqrt(8n-7)))
    j = n - div(i*(i-1),2)
    return i, j
end
