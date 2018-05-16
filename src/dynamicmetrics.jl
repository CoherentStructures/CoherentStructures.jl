# (c) 2018 Alvaro de Diego & Daniel Karrasch

import Distances: result_type, evaluate, eval_start, eval_op, eval_reduce, eval_end

Dists = Distances

struct PEuclidean{W <: Dists.RealAbstractArray} <: Dists.Metric
    periods::W
end

"""
    PEuclidean([L])
Create a Euclidean metric on a rectangular periodic domain.
Periods per dimension are contained in the vector `L`.
For dimensions without periodicity put `Inf` in the respective component.
"""

PEuclidean() = PEuclidean(fill(Inf,2))

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
@inline eval_op(::PEuclidean, ai, bi, pi) = begin d = abs(ai - bi); d = mod(d, pi); d = min(d, pi-d); abs2(d) end
@inline eval_reduce(::PEuclidean, s1, s2) = s1 + s2

peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray) = evaluate(PEuclidean(p), a, b)
peuclidean(a::AbstractArray, b::AbstractArray) = evaluate(PEuclidean(), a, b)

########## spatiotemporal, time averaged metrics ##############

```
    STmetric(Smetric, dim, p)

Creates a spatiotemporal, averaged in time metric.

## Properties
   * `Smetric` is a metric as defined in the `Distances` package, e.g.,
     `Euclidean`, `PEuclidean`, or `Haversine`
   * `dim` corresponds to the spatial dimension
   * `p` corresponds to the kind of average applied to the vector of spatial distances:
     - `p = Inf`: maximum
     - `p = 2`: mean squared average
     - `p = 1`: arithmetic mean
     - `p = -1`: harmonic mean (does not yield a metric!)
     - `p = -Inf`: minimum (does not yield a metric!)
```

struct STmetric <: Dists.Metric
    Smetric::Dists.Metric
    dim::Int
    p::Real
end

STmetric()          = STmetric(Dists.Euclidean(), 2, 1)
STmetric(p::Real)   = STmetric(Dists.Euclidean(), 2, p)

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function evaluate(d::STmetric, a::Union{Array, Dists.ArraySlice}, b::Union{Array, Dists.ArraySlice})
    la = length(a)
    lb = length(b)
    @boundscheck if la != lb
        throw(DimensionMismatch("First array has size $(size(a)) which does not match the size of the second, $(size(b))."))
    end
    if la == 0
        return zero(result_type(d, a, b))
    end
    q = div(la,d.dim)
    # s = eval_start(d, a, b, sm)
    s = eval_reduce(d, eval_op(d, a, b, d.Smetric, d.dim, q), d.p, q)
    return eval_end(d, s)
end

# this version is needed for NearestNeighbors `NNTree`
@inline function evaluate(d::STmetric, a::AbstractArray, b::AbstractArray)
    la = length(a)
    lb = length(b)
    @boundscheck if la != lb
        throw(DimensionMismatch("First array has size $(size(a)) which does not match the size of the second, $(size(b))."))
    end
    if la == 0
        return zero(result_type(d, a, b))
    end
    # number of time steps
    q = div(la,d.dim)
    return eval_reduce(d, eval_op(d, a, b, d.Smetric, d.dim, q), d.p, q)
end

function result_type(d::STmetric, a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1, T2}
    result_type(d.Smetric, a, b)
end

@inline eval_op(::STmetric, a::AbstractArray, b::AbstractArray, sm::Dists.Metric, dim::Int, q::Int) = Dists.colwise(sm, reshape(a, dim, q), reshape(b, dim, q))

@inline eval_reduce(::STmetric, s, p, q) = q^(-inv(p))*vecnorm(s, p)

@inline eval_end(::STmetric, s) = s

stmetric(a::AbstractArray, b::AbstractArray, d::Dists.PreMetric, dim::Int, p::Real) = evaluate(STmetric(d, 2, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, d::Dists.PreMetric, p::Real) = evaluate(STmetric(d, 2, p), a, b)
stmetric(a::AbstractArray, b::AbstractArray, p::Real) = evaluate(STmetric(p), a, b)
stmetric(a::AbstractArray, b::AbstractArray) = evaluate(STmetric(), a, b)

"""
        meanmetric(F, av, metric)

For a set of trajectories ``x_i^t`` calculate an "averaged" distance matrix
``K ∈ R^{N×N}`` where ``k_{ij} = av(metric(x_i^1, x_j^1), … , metric(x_i^T, x_j^T))``.
# Arguments
   * `F`: trajectory data with `size(F) = dim, T, N`
   * `av`: time-averaging function, see below for examples
   * `metric`: spatial distance function metric

## Examples for metrics
   * `Euclidean()`
   * `Haversine(r)`: geodesic distance on a sphere of radius `r` in the
        same units as `r`
   * `PEuclidean(L)`: Euclidean distance on a periodic domain, periods
        are contained in the vector `L`

## Examples for time averages
   * `av = mean`: arithmetic time-average, L¹ in time [Hadjighasem et al.]
   * `av = x->1/mean(inv,x)`: harmonic time-average [de Diego et al.]
   * `av = max`: sup/L\^infty in time [mentioned by Hadjighasem et al.]
   * `av = x->min(x)<=ε`: encounter adjacency [Rypina et al., Padberg-Gehle & Schneide]
   * `av = x->mean(abs2,x)`: Euclidean delay coordinate metric [cf. Froyland & Padberg-Gehle]
"""
function meanmetric(F::AbstractArray{T,3},
                    av::Function,
                    metric::Dists.PreMetric = Dists.Euclidean()
                    ) where T <: Real
    dim, t, N = size(F)

    entries = div(N*(N+1),2)

    # allocate distance matrix
    R = SharedArray{T,2}(N,N)

    # allocate `dists` and make it known on all workers
    dists = Vector{Distances.result_type(metric,F[:,1,1],F[:,1,1])}(t)
    @everywhere @eval dists = $(dists)

    # enumerate the entries linearly to distribute them evenly across the workers
    @views @sync @parallel for n = 1:entries
    # Threads.@threads for n = 1:entries
            i, j = tri_indices(n)
            # fill_distances!(dists, F[:,:,i], F[:,:,j], metric, t)
            Dists.colwise!(dists, metric, F[:,:,i], F[:,:,j])
            R[i,j] = av(dists)
    end
    Symmetric(R,:L)
end

# fill_distances! could be used instead of Distances.colwise! for "metrics"
# that are not defined as a subtype of Distances.PreMetric

# function fill_distances!(dists, xis, xjs, k, t)
#     @views for l in 1:t
#         dists[l] = k(xis[:,l], xjs[:,l])
#     end
#     return dists
# end

@inline function tri_indices(n::Int)
    i = floor(Int, 0.5*(1 + sqrt(8n-7)))
    j = n - div(i*(i-1),2)
    return i, j
end
