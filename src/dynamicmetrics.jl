# (c) 2018 Alvaro de Diego & Daniel Karrasch

import Distances: result_type, evaluate, eval_start, eval_op, eval_reduce, eval_end, pairwise, pairwise!

#const Dists = Distances

struct PEuclidean{W <: Distances.RealAbstractArray} <: Distances.Metric
    periods::W
end

"""
    PEuclidean([L])
Create a Euclidean metric on a rectangular periodic domain.
Periods per dimension are contained in the vector `L`.
For dimensions without periodicity put `Inf` in the respective component.
"""

PEuclidean() = Distances.Euclidean()

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function evaluate(d::PEuclidean, a::Union{Array, Distances.ArraySlice}, b::Union{Array, Distances.ArraySlice})
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
            pi = d.periods[I]
            s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
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
                pi = d.periods[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
            end
        else
            for (Ia, Ib, Ip) in zip(eachindex(a), eachindex(b), eachindex(d.weights))
                ai = a[Ia]
                bi = b[Ib]
                pi = d.periods[Ip]
                s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
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
@inline eval_end(::PEuclidean, s) = sqrt(s)
@inline eval_op(::PEuclidean, ai, bi, pi) = begin d = abs(ai - bi); d = mod(d, pi); d = min(d, abs(pi-d)); abs2(d) end
@inline eval_reduce(::PEuclidean, s1, s2) = s1 + s2

peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray) = evaluate(PEuclidean(p), a, b)
peuclidean(a::AbstractArray, b::AbstractArray) = evaluate(PEuclidean(), a, b)

########## spatiotemporal, time averaged metrics ##############

"""
    STmetric(Smetric, dim, p)

Creates a spatiotemporal, averaged in time metric.

# Properties

   * `Smetric` is a metric as defined in the `Distances` package, e.g.,
     `Euclidean`, `PEuclidean`, or `Haversine`
   * `dim` corresponds to the spatial dimension
   * `p` corresponds to the kind of average applied to the vector of spatial distances:
     - `p = Inf`: maximum
     - `p = 2`: mean squared average
     - `p = 1`: arithmetic mean
     - `p = -1`: harmonic mean (does not yield a metric!)
     - `p = -Inf`: minimum (does not yield a metric!)
"""

struct STmetric{T <: Real, M <: Distances.Metric} <: Distances.Metric
    Smetric::M
    dim::Int
    p::T
end

STmetric()          = STmetric(Distances.Euclidean(), 2, 1)
STmetric(p::Real)   = STmetric(Distances.Euclidean(), 2, p)

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function evaluate(d::STmetric, a::Union{Array, Distances.ArraySlice}, b::Union{Array, Distances.ArraySlice})
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
    return reduce_time(d, eval_space(d, a, b, d.Smetric, d.dim, q), d.p)
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
    return reduce_time(d, eval_space(d, a, b, d.Smetric, d.dim, q), d.p)
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

@inline reduce_time(::STmetric, s, p) = vecnorm(s, p)
# @inline reduce_time(::STmetric, s, p, q) = q^(-inv(p)) * vecnorm(s, p)

stmetric(a::AbstractArray, b::AbstractArray, d::Distances.PreMetric, dim::Int, p::Real) = evaluate(STmetric(d, 2, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, d::Distances.PreMetric, p::Real) = evaluate(STmetric(d, 2, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, p::Real) = evaluate(STmetric(p), a, b)
stmetric(a::AbstractArray, b::AbstractArray) = evaluate(STmetric(), a, b)

########### parallel pairwise computation #################

function pairwise!(r::SharedMatrix{T}, metric::STmetric, a::AbstractMatrix, b::AbstractMatrix) where T <: Real
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
