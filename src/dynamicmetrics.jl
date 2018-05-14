# (c) 2018 Alvaro de Diego & Daniel Karrasch

import Distances: result_type, evaluate, eval_start, eval_op, eval_reduce, eval_end

Dists = Distances

struct PEuclidean{W <: Dists.RealAbstractArray} <: Dists.Metric
    periods::W
end

"""
    PEuclidean([L])
Create a Euclidean metric on a periodic domain.
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

    # linear storage of triangular matrix
    R = SharedArray{T,2}(N,N)

    # enumerate the entries linearly to distribute them evenly across the workers
    @everywhere dists = $(Vector{Distances.result_type(metric,F[:,1,1],F[:,1,1])}(t))

    @views @sync @parallel for n = 1:entries
       i, j = tri_indices(n)
       # fill_distances!(dists, F[:,:,i], F[:,:,j], metric, t)
       Dists.colwise!(dists, metric, F[:,:,i], F[:,:,j])
       R[i,j] = av(dists)
       # R[i,j] = av(Dists.colwise(metric, view(F,:,:,i), view(F,:,:,j)))
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
