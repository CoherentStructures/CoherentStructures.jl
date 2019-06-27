# (c) 2018-19 Daniel Karrasch & Alvaro de Diego

import Distances: evaluate, eval_reduce, eval_end
import Distances: pairwise, pairwise!, colwise, colwise!

###### should be deleted once Distances.jl with PeriodicEuclidean is tagged ####
import Distances: result_type, evaluate, eval_start, eval_op
import Distances: eval_reduce, eval_end, pairwise, pairwise!

"""
    PEuclidean(L)
Create a Euclidean metric on a rectangular periodic domain.
Periods per dimension are contained in the vector `L`.
For dimensions without periodicity put `Inf` in the respective component.

## Example
```
julia> using Distances

julia> x, y, L = [0.0, 0.0], [0.7, 0.0], [0.5, Inf]
([0.0, 0.0], [0.7, 0.0], [0.5, Inf])

julia> evaluate(PEuclidean(L), x, y)
0.19999999999999996
```
"""
struct PEuclidean{W <: Union{Dists.RealAbstractArray,Real}} <: Dists.Metric
    periods::W
end

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
@inline function Dists.eval_op(d::PEuclidean, ai, bi, p)
    s1 = abs(ai - bi)
    s2 = mod(s1, p)
    s3 = min(s2, p - s2)
    abs2(s3)
end
@inline Dists.eval_reduce(::PEuclidean, s1, s2) = s1 + s2
@inline Dists.eval_end(::PEuclidean, s) = sqrt(s)

peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray) = evaluate(PEuclidean(p), a, b)
peuclidean(a::AbstractArray, b::AbstractArray) = euclidean(a, b)
peuclidean(a::Number, b::Number, p::Number) = begin d = mod(abs(a - b), p); d = min(d, p - d) end
####### delete until here ####################

########## spatiotemporal, time averaged metrics ##############

"""
    STmetric(metric, p)

Creates a spatiotemporal, averaged-in-time metric. At each time instance, the
distance between two states `a` and `b` is computed via `evaluate(metric, a, b)`.
The resulting distances are subsequently ``ℓ^p``-averaged, with ``p=`` `p`.

## Fields
   * `metric=Euclidean()`: a `SemiMetric` as defined in the `Distances.jl` package, e.g.,
     [`Euclidean`](@ref), [`PeriodicEuclidean`](@ref), or [`Haversine`](ref);
      * `p = Inf`: maximum
      * `p = 2`: mean squared average
      * `p = 1`: arithmetic mean
      * `p = -1`: harmonic mean (does not yield a metric!)
      * `p = -Inf`: minimum (does not yield a metric!)

## Example
```
julia> using Distances, StaticArrays; x = [@SVector rand(2) for _ in 1:10];

julia> STmetric(Euclidean(), 1) # Euclidean distances arithmetically averaged
STmetric{Euclidean,Int64}(Euclidean(0.0), 1)

julia> evaluate(STmetric(Euclidean(), 1), x, x)
0.0
```
"""
struct STmetric{M<:Dists.SemiMetric, T<:Real} <: Dists.SemiMetric
    metric::M
    p::T
end

# defaults: metric = Euclidean(), dim = 2, p = 1
STmetric(d::Dists.PreMetric=Dists.Euclidean()) = STmetric(d, 1)

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function Dists.evaluate(d::STmetric,
            a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector})
    la = length(a)
    lb = length(b)
    @boundscheck if la != lb
        throw(DimensionMismatch("first array has length $la which does not match the length of the second, $lb."))
    elseif la == 0
        return zero(Float64)
    end
	@inbounds begin
		s = zero(result_type(d, a, b))
    	if IndexStyle(a, b) === IndexLinear()
			@simd for I in 1:length(a)
                ai = a[I]
                bi = b[I]
        		s = Dists.eval_reduce(d, s, Dists.evaluate(d.metric, ai, bi))
			end
		else
			if size(a) == size(b)
                @simd for I in eachindex(a, b)
                    ai = a[I]
                    bi = b[I]
					s = Dists.eval_reduce(d, s, Dists.evaluate(d.metric, ai, bi))
				end
			else
				for (Ia, Ib) in zip(eachindex(a), eachindex(b))
					ai = a[Ia]
					bi = b[Ib]
					s = Dists.eval_reduce(d, s, Dists.evaluate(d.metric, ai, bi))
				end
			end
		end
		return Dists.eval_end(d, s / la)
    end
end

function Dists.result_type(d::STmetric, a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector})
	T = result_type(d.metric, zero(eltype(a)), zero(eltype(b)))
	return typeof(Dists.eval_end(d, Dists.eval_reduce(d, zero(T), zero(T)) / 2))
end

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
        return s1 + s2 ^ d.p
    end
end
@inline function Dists.eval_end(d::STmetric, s)
	p = d.p
    if p == 1
        return s
    elseif p == 2
        return √s
    elseif p == -2
        return 1 / √s
    elseif p == -1
        return 1 / s
    elseif p == Inf
        return s
    elseif p == -Inf
        return s
    else
        return s ^ (1 / d.p)
    end
end

stmetric(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector},
			d::Dists.SemiMetric=Dists.Euclidean(), p::Real=1) =
    evaluate(STmetric(d, p), a, b)

function pairwise!(r::AbstractMatrix, metric::STmetric,
                   a::AbstractVector{<:AbstractVector{<:SVector}}, b::AbstractVector{<:AbstractVector{<:SVector}};
                   dims::Union{Nothing,Integer}=nothing)
    la, lb = length.((a, b))
    size(r) == (la, lb) || throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((na, nb)))."))
	@inbounds for j = 1:lb
        bj = b[j]
        for i = 1:la
            r[i, j] = evaluate(metric, a[i], bj)
        end
    end
    r
end
function pairwise!(r::AbstractMatrix, metric::STmetric, a::AbstractVector{<:AbstractVector{<:SVector}};
                   dims::Union{Nothing,Integer}=nothing)
	la = length(a)
    size(r) == (la, la) || throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((n, n)))."))
	@inbounds for j = 1:la
        aj = a[j]
        for i = (j + 1):la
            r[i, j] = evaluate(metric, a[i], aj)
        end
        r[j, j] = 0
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
    end
    r
end

function pairwise(metric::STmetric, a::AbstractVector{<:AbstractVector{<:SVector}}, b::AbstractVector{<:AbstractVector{<:SVector}};
                  dims::Union{Nothing,Integer}=nothing)
    la = length(a)
    lb = length(b)
	(la == 0 || lb == 0) && return reshape(Float64[], 0, 0)
    r = Matrix{result_type(metric, a[1], b[1])}(undef, la, lb)
    pairwise!(r, metric, a, b)
end
function pairwise(metric::STmetric, a::AbstractVector{<:AbstractVector{<:SVector}};
                  dims::Union{Nothing,Integer}=nothing)
    la = length(a)
	la == 0 && return reshape(Float64[], 0, 0)
    r = Matrix{result_type(metric, a[1], a[1])}(undef, la, la)
    pairwise!(r, metric, a)
end

function colwise!(r::AbstractVector, metric::STmetric, a::AbstractVector{<:SVector}, b::AbstractVector{<:AbstractVector{<:SVector}})
	lb = length(b)
    length(r) == lb || throw(DimensionMismatch("incorrect length of r (got $(length(r)), expected $lb)"))
	@inbounds for i = 1:lb
        r[i] = evaluate(metric, a, b[i])
    end
    r
end
function colwise!(r::AbstractVector, metric::STmetric, a::AbstractVector{<:AbstractVector{<:SVector}}, b::AbstractVector{<:SVector})
	colwise!(r, metric, b, a)
end
function colwise!(r::AbstractVector, metric::STmetric, a::T, b::T) where {T<:AbstractVector{<:AbstractVector{<:SVector}}}
	la = length(a)
	lb = length(b)
	la == lb || throw(DimensionMismatch("lengths of a, $la, and b, $lb, do not match"))
	la == length(r) || throw(DimensionMismatch("incorrect size of r, got $(length(r)), but expected $la"))
	@inbounds for i = 1:la
        r[i] = evaluate(metric, a[i], b[i])
    end
    r
end

function colwise(metric::STmetric, a::AbstractVector{<:SVector}, b::AbstractVector{<:AbstractVector{<:SVector}})
	lb = length(b)
	colwise!(zeros(Dists.result_type(metric, a[1], b[1]), lb), metric, a, b)
end
function colwise(metric::STmetric, a::AbstractVector{<:AbstractVector{<:SVector}}, b::AbstractVector{<:SVector})
	la = length(a)
	colwise!(zeros(Dists.result_type(metric, a[1], b[1]), la), metric, b, a)
end
function colwise(metric::STmetric, a::T, b::T) where {T<:AbstractVector{<:AbstractVector{<:SVector}}}
	la = length(a)
	lb = length(b)
    la == lb || throw(DimensionMismatch("lengths of a, $la, and b, $lb, do not match"))
	la == 0 && return reshape(Float64[], 0, 0)
	colwise!(zeros(Dists.result_type(metric, a[1], b[1]), la), metric, a, b)
end

################## sparse pairwise distance computation ###################
"""
    spdist(data, sp_method, metric=Euclidean()) -> SparseMatrixCSC

Return a sparse distance matrix as determined by the sparsification method `sp_method`
and `metric`.
"""
function spdist(data::AbstractVector{<:SVector}, sp_method::Neighborhood, metric::Dists.PreMetric=Distances.Euclidean())
    N = length(data) # number of states
	# TODO: check for better leafsize values
    tree = metric isa NN.MinkowskiMetric ? NN.KDTree(data, metric;  leafsize = 10) : NN.BallTree(data, metric; leafsize = 10)
	idxs = NN.inrange(tree, data, sp_method.ε, false)
	Js = vcat(idxs...)
    Is = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
	Vs = fill(1.0, length(Is))
    return sparse(Is, Js, Vs, N, N, *)
end
function spdist(data::AbstractVector{<:SVector}, sp_method::Union{KNN,MutualKNN}, metric::Dists.PreMetric=Distances.Euclidean())
	N = length(data) # number of states
	# TODO: check for better leafsize values
    tree = metric isa NN.MinkowskiMetric ? NN.KDTree(data, metric;  leafsize = 10) : NN.BallTree(data, metric; leafsize = 10)
	idxs, dists = NN.knn(tree, data, sp_method.k, false)
	Js = vcat(idxs...)
	Is = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
	Ds = vcat(dists...)
	D = sparse(Is, Js, Ds, N, N)
	Dvals = nonzeros(D)
	Dtvals = nonzeros(permutedims(D))
    if sp_method isa KNN
        Dvals .= max.(Dvals, Dtvals)
    else # sp_method isa MutualKNN
        Dvals .= min.(Dvals, Dtvals)
    end
	return D
end
function spdist(data::AbstractVector{<:AbstractVector{<:SVector}}, sp_method::Neighborhood, metric::STmetric)
	N = length(data) # number of trajectories
	T = Dists.result_type(metric, data[1], data[1])
	I1 = collect(1:N)
	J1 = collect(1:N)
	V1 = zeros(T, N)
	tempfun(j) = begin
		Is, Js, Vs = Int[], Int[], T[]
		aj = data[j]
        for i = (j + 1):N
            temp = Dists.evaluate(metric, data[i], aj)
			if temp < sp_method.ε
				push!(Is, i, j)
				push!(Js, j, i)
				push!(Vs, temp, temp)
			end
        end
		return Is, Js, Vs
	end
	if Distributed.nprocs() > 1
		IJV = Distributed.pmap(tempfun, 1:(N-1); batch_size=(N÷Distributed.nprocs()))
	else
		IJV = map(tempfun, 1:(N-1))
	end
	Is = vcat(I1, getindex.(IJV, 1)...)
	Js = vcat(J1, getindex.(IJV, 2)...)
	Vs = vcat(V1, getindex.(IJV, 3)...)
	return sparse(Is, Js, Vs, N, N)
end
function spdist(data::AbstractVector{<:AbstractVector{<:SVector}}, sp_method::Union{KNN,MutualKNN}, metric::STmetric)
	N, k = length(data), sp_method.k
    Is = SharedArray{Int}(N*(k+1))
    Js = SharedArray{Int}(N*(k+1))
	T = typeof(evaluate(metric, data[1], data[1]))
    Ds = SharedArray{T}(N*(k+1))
    # index = Vector{Int}(undef, k+1)
	ds = Vector{T}(undef, N)
    # Distributed.@everywhere index = Vector{Int}(undef, k+1)
    @inbounds @sync Distributed.@distributed for i=1:N
		colwise!(ds, metric, data[i], data)
        index = partialsortperm(ds, 1:(k+1))
        Is[(i-1)*(k+1)+1:i*(k+1)] .= i
        Js[(i-1)*(k+1)+1:i*(k+1)] = index
        Ds[(i-1)*(k+1)+1:i*(k+1)] = ds[index]
    end
	D = sparse(Is, Js, Ds, N, N)
	Dvals = D.nzval
	Dtvals = permutedims(D).nzval
    if sp_method isa KNN
        Dvals .= max.(Dvals, Dtvals)
    else # sp_method isa MutualKNN
        Dvals .= min.(Dvals, Dtvals)
    end
	return SparseArrays.dropzeros!(D)
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
#     r = SharedArray{result_type(metric, a, b)}(m, n)
#     pairwise!(r, metric, a, b)
# end
#
# function Dists.pairwise(metric::STmetric, a::AbstractMatrix)
#     n = size(a, 2)
#     r = SharedArray{result_type(metric, a, a)}(n, n)
#     pairwise!(r, metric, a)
# end

@inline function tri_indices!(i::Int, j::Int, n::Int)
    i = floor(Int, 0.5*(1 + sqrt(8n-7)))
    j = n - div(i*(i-1),2)
    return i, j
end
