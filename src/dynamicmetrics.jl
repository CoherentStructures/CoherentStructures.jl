# (c) 2018 Alvaro de Diego & Daniel Karrasch

import Distances: result_type, evaluate, eval_start, eval_op
import Distances: eval_reduce, eval_end, pairwise, pairwise!

struct PEuclidean{W <: Union{Dists.RealAbstractArray,Real}} <: Dists.Metric
    periods::W
end

"""
    PEuclidean(L)
Create a Euclidean metric on a rectangular periodic domain.
Periods per dimension are contained in the vector `L`.
For dimensions without periodicity put `Inf` in the respective component.

# Example
```jldoctest
julia> using Distances

julia> x, y, L = [0.0, 0.0], [0.7, 0.0], [0.5, Inf]
([0.0, 0.0], [0.7, 0.0], [0.5, Inf])

julia> evaluate(PEuclidean(L),x,y)
0.19999999999999996
```
"""
PEuclidean() = Dists.Euclidean()

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function Dists.evaluate(d::PEuclidean, a::Union{Array, Dists.ArraySlice}, b::Union{Array, Dists.ArraySlice})
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
        @simd for I in 1:length(a)
            ai = a[I]
            bi = b[I]
            li = d.periods[I]
            s = eval_reduce(d, s, eval_op(d, ai, bi, li))
        end
        return eval_end(d, s)
    end
end

@inline function Dists.evaluate(d::PEuclidean, a::AbstractArray, b::AbstractArray)
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
            for (Ia, Ib, Ip) in zip(eachindex(a), eachindex(b), eachindex(d.periods))
                ai = a[Ia]
                bi = b[Ib]
                li = d.periods[Ip]
                s = eval_reduce(d, s, eval_op(d, ai, bi, li))
            end
        end
    end
    return eval_end(d, s)
end

function Dists.evaluate(dist::PEuclidean, a::T, b::T) where {T <: Number}
    # eval_end(dist, eval_op(dist, a, b, dist.periods[1]))
    peuclidean(a, b, dist.periods[1])
end
# function result_type(dist::PEuclidean, ::AbstractArray{T1}, ::AbstractArray{T2}) where {T1, T2}
#     typeof(evaluate(dist, one(T1), one(T2)))
# end
@inline function Dists.eval_start(d::PEuclidean, a::AbstractArray, b::AbstractArray)
    zero(result_type(d, a, b))
end
@inline Dists.eval_op(::PEuclidean, ai, bi, li) = begin d = mod(abs(ai - bi), li); d = min(d, li-d); abs2(d) end
@inline Dists.eval_reduce(::PEuclidean, s1, s2) = s1 + s2
@inline Dists.eval_end(::PEuclidean, s) = sqrt(s)

peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray) = evaluate(PEuclidean(p), a, b)
peuclidean(a::AbstractArray, b::AbstractArray) = euclidean(a, b)
peuclidean(a::Number, b::Number, p::Number) = begin d = mod(abs(a - b), p); d = min(d, p - d) end

########## spatiotemporal, time averaged metrics ##############

"""
    STmetric(Smetric, dim, p)

Creates a spatiotemporal, averaged in time metric.

# Properties

   * `Smetric` is a metric as defined in the `Distances` package, e.g.,
     `Euclidean`, `PEuclidean`, or `Haversine`;
   * `dim` corresponds to the spatial dimension;
   * `p` corresponds to the kind of average applied to the vector of spatial distances:
     - `p = Inf`: maximum
     - `p = 2`: mean squared average
     - `p = 1`: arithmetic mean
     - `p = -1`: harmonic mean (does not yield a metric!)
     - `p = -Inf`: minimum (does not yield a metric!)

# Example
```julia
julia> x, y = rand(10), rand(10)

julia> Distances.evaluate(STmetric(Distances.Euclidean(),2,1),x,y)
```
"""
struct STmetric{M <: Dists.Metric, T <: Real} <: Dists.Metric
    Smetric::M
    dim::Int
    p::T
end

# defaults: metric = Euclidean(), dim = 2, p = 1
STmetric()                          = STmetric(Dists.Euclidean(), 2, 1)
STmetric(p::Real)                   = STmetric(Dists.Euclidean(), 2, p)
STmetric(d::Dists.Metric)           = STmetric(d, 2, 1)
STmetric(d::Dists.Metric, p::Real)  = STmetric(d, 2, p)

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function Dists.evaluate(d::STmetric, a::Union{Array, Dists.ArraySlice}, b::Union{Array, Dists.ArraySlice})
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
    return reduce_time(d, eval_space(d, a, b, q), q)
end

# this version is needed for NearestNeighbors `NNTree`
@inline function Dists.evaluate(d::STmetric, a::AbstractArray, b::AbstractArray)
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
    return reduce_time(d, eval_space(d, a, b, q), q)
end

function Dists.result_type(d::STmetric, a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1, T2}
    result_type(d.Smetric, a, b)
end

@inline eval_space(d::STmetric, a::AbstractArray, b::AbstractArray, q::Int) =
        Distances.colwise(d.Smetric, reshape(a, d.dim, q), reshape(b, d.dim, q))
@inline eval_space!(dist::AbstractVector, d::STmetric, a::AbstractArray, b::AbstractArray, q::Int) =
        Distances.colwise!(dist, d.Smetric, reshape(a, d.dim, q), reshape(b, d.dim, q))

@inline reduce_time(d::STmetric, s, q) = q^(-1 / d.p) * norm(s, d.p)

stmetric(a::AbstractArray, b::AbstractArray, d::Dists.Metric, dim::Int, p::Real) =
        evaluate(STmetric(d, dim, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, d::Dists.Metric, p::Real) =
        evaluate(STmetric(d, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, d::Dists.Metric) =
        evaluate(STmetric(d), a, b)
stmetric(a::AbstractArray, b::AbstractArray, p::Real) =
        evaluate(STmetric(p), a, b)
stmetric(a::AbstractArray, b::AbstractArray) =
        evaluate(STmetric(), a, b)

########### parallel pairwise computation #################

# function Dists.pairwise!(r::SharedArrays.SharedMatrix{T}, d::STmetric, a::AbstractMatrix, b::AbstractMatrix) where T <: Real
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
# function Dists.pairwise!(r::SharedArrays.SharedMatrix{T}, d::STmetric, a::AbstractMatrix) where T <: Real
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
#     r = SharedArrays.SharedArray{result_type(metric, a, b)}(m, n)
#     pairwise!(r, metric, a, b)
# end
#
# function Dists.pairwise(metric::STmetric, a::AbstractMatrix)
#     n = size(a, 2)
#     r = SharedArrays.SharedArray{result_type(metric, a, a)}(n, n)
#     pairwise!(r, metric, a)
# end

@inline function tri_indices!(i::Int, j::Int, n::Int)
    i = floor(Int, 0.5*(1 + sqrt(8n-7)))
    j = n - div(i*(i-1),2)
    return i, j
end
