# (c) 2018 Alvaro de Diego & Daniel Karrasch

Dists = Distances

doc"""
    diff_op(data, kernel, ε; α, metric)

Return a diffusion/Markov matrix `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `kernel`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `ε`: distance threshold;
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function diff_op(data::AbstractArray{T, 2},
                    kernel::Function,
                    ε::T;
                    α=1.0,
                    metric::Dists.PreMetric = Dists.Euclidean()) where T <: Number

    N = size(data, 2)
    D = Dists.pairwise(metric,data)
    Is, Js = findn(D .< ε)
    Vs = kernel.(D[D .< ε])
    P = sparse(Is, Js, Vs, N, N)
    α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

doc"""
    sparse_diff_op_family(data, k, ε, dim, op_reduce; α, metric)

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `ε`: distance threshold;
   * if `dim` is given, the columns are interpreted as concatenations of `dim`-
     dimensional points, to which `metric` is applied individually;
   * `op_reduce`: time-reduction of diffusion operators, e.g. `mean` or `P -> prod(LinearMaps.LinearMap,reverse(P))`
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function sparse_diff_op_family( data::AbstractArray{T, 2},
                                kernel::F,
                                ε::S,
                                dim::Int,
                                op_reduce::Function = P -> prod(LinearMaps.LinearMap,reverse(P));
                                α=1.0,
                                metric::Dists.PreMetric = Dists.Euclidean()
                                ) where {T <: Number, S <: Real, F <: Function}
    dimt, N = size(data)
    (q, r) = divrem(dimt,dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    P = pmap(1:q) do t
        # @time Pₜ = sparse_diff_op( view(sols,:,t,:), kernel, ε; metric = metric )
        @time Pₜ = sparse_diff_op( data[(t-1)*dim+1:t*dim,:], kernel, ε; α=α, metric = metric )
        println("Timestep $t/$q done")
        Pₜ
    end
    return op_reduce(P)
end

doc"""
    sparse_diff_op(data, k, ε; α=1.0, metric=Euclidean())

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `ε`: distance threshold;
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

@inline function sparse_diff_op(data::AbstractArray{T, 2},
                        kernel::F,
                        ε::S;
                        α=1.0,
                        metric::Dists.PreMetric = Dists.Euclidean()
                        ) where {T <: Real, S <: Real, F <: Function}

    typeof(metric) == STmetric && metric.p < 1 && throw("Cannot use balltrees for sparsification with $(metric.p)<1.")
    P = sparseaffinitykernel(data, kernel, ε, metric)
    α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

"""
    sparseaffinitykernel(A, k, ε, metric=Euclidean())

Return a sparse matrix `K` where ``k_{ij} = k(x_i, x_j)``.
The ``x_i`` are taken from the columns of `A`. Entries are
only calculated for pairs where ``metric(x_i, x_j)≦ε``.
Default metric is `Euclidean()`.
"""

@inline function sparseaffinitykernel(data::Array{T, 2},
                               kernel::F,
                               ε::S,
                               metric::Dists.PreMetric = Dists.Euclidean()
                               ) where {T <: Real, S <: Number, F <:Function}
    dim, N = size(data)

    balltree = NearestNeighbors.BallTree(data, metric)
    idxs = NearestNeighbors.inrange(balltree, data, ε, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i,length(idxs[i])) for i in eachindex(idxs)]...)
    Vs::Vector{T} = kernel.(Dists.colwise(metric, view(data,:,Is), view(data,:,Js)))
    return sparse(Is,Js,Vs,N,N)
end

doc"""
    sparse_adjacency_family(data, k, ε, dim; α, metric)

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `ε`: distance threshold;
   * if `dim` is given, the columns are interpreted as concatenations of `dim`-
     dimensional points, to which `metric` is applied individually;
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function sparse_adjacency_family(data::AbstractArray{T, 2},
                                    kernel::F,
                                    ε::S,
                                    dim::Int;
                                    α=1.0,
                                    metric::Dists.PreMetric = Dists.Euclidean()
                                ) where {T <: Real, S <: Real, F <: Function}
    dimt, N = size(data)
    (q, r) = divrem(dimt,dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    As = pmap(1:q) do t
        @time A = sparse_adjacency_list( data[(t-1)*dim+1:t*dim,:], kernel, ε; α=α, metric = metric )
        println("Timestep $t/$q done")
        A
    end
    IJ = unique(vcat(As...))
    Is, Js = [ij[1] for ij in IJ], [ij[2] for ij in IJ]
    Vs = fill(1,length(Is))
    return sparse(Is,Js,Vs,N,N)
end

doc"""
    sparse_adjacency_list(data, ε; metric=Euclidean()) -> idxs::Vector{Vector}

Return two lists of indices of data points that are adjacent.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `ε`: distance threshold;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

@inline function sparse_adjacency_list(data::AbstractArray{T, 2},
                                ε::S;
                                metric::Dists.PreMetric = Dists.Euclidean()
                               )::Vector{Vector{Int}} where {T <: Real, S <: Real}

    typeof(metric) == STmetric && metric.p < 1 && throw("Cannot use balltrees for sparsification with $(metric.p)<1.")
    balltree = NearestNeighbors.BallTree(data, metric)
    idxs = NearestNeighbors.inrange(balltree, data, ε, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i,length(idxs[i])) for i in eachindex(idxs)]...)
    return vcat.(Is, Js)
end

doc"""
    α_normalize!(A, α = 0.5)
Normalize rows and columns of `A` in-place with the respective row-sum to the α-th power;
i.e., return $ a_{ij}:=a_{ij}/q_i^{\\alpha}/q_j^{\\alpha}$, where
$ q_k = \\sum_{\\ell} a_{k\\ell}$. Default for `α` is 0.5.
"""

@inline function α_normalize!(A::T, α::S = 0.5) where T <: AbstractMatrix where S <: Real
    LinAlg.checksquare(A)
    qₑ = 1./squeeze(sum(A, 2),2).^α
    scale!(A,qₑ)
    scale!(qₑ,A)
    return A
end

doc"""
    wLap_normalize!(A)
Normalize rows of `A` in-place with the respective row-sum; i.e., return
$ a_{ij}:=a_{ij}/q_i$.
"""

@inline function wLap_normalize!(A::T) where T <: AbstractMatrix
    LinAlg.checksquare(A)
    dᵅ = 1./squeeze(sum(A, 2),2)
    scale!(dᵅ,A)
    return A
 end

 """
     stationary_distribution(P,N)

 Compute the stationary distribution for a Markov transition operator.
 `P` may be dense or sparse, or a `LinearMap` matrix-vector multiplication
 is given by a function.
 """

 function stationary_distribution(P::Union{SparseMatrixCSC{T,Int64},LinearMaps.LinearMap{T},DenseMatrix{T}}) where T <: Real

     E   = eigs(P; nev=1, maxiter=1000, which=:LM)
     π   = squeeze(real(E[2]),2) # stationary distribution
     ext = extrema(π)
     prod(ext) < zero(eltype(ext)) && throw(error("Both signs in stationary distribution"))
     if any(ext .< zero(eltype(ext)))
         π .= -π
     end
     return π
 end

 @inline function L_mul_Lt(L::LinearMaps.LinearMap{T},π::Vector{T})::LinearMaps.LinearMap{T} where T <: Real

     πsqrt  = Diagonal(sqrt.(π)) # alternavtive: Diagonal(sqrt.(π))
     πinv   = Diagonal(1./π) # alternative: Diagonal(inv.(π))
     return πsqrt * L * πinv * transpose(L) * πsqrt
 end

 @inline function L_mul_Lt(L::AbstractArray{T},π::Vector{T})::LinearMaps.LinearMap{T} where T <: Real

     πsqrt = sqrt.(π)
     πinvsqrt = 1./πsqrt
     scale!(πsqrt,L)
     scale!(L,πinvsqrt)
     LMap = LinearMaps.LinearMap(L)
     return LMap * transpose(LMap)
 end

 """
     diffusion_coordinates(P,n_coords)

 Compute the (time-coupled) diffusion coordinates for `P`, where `P` is either
 a graph Laplacian or a list of sparse Markovian transition matrices (as output
 by [sparse_diff_op](@ref)). `n_coords` determines the number of diffusion coordinates
 to be computed.
 """

 function diffusion_coordinates(P::S,n_coords::Int) where S <: Union{SparseMatrixCSC,LinearMaps.LinearMap,DenseMatrix}

     N = LinAlg.checksquare(P)

     π = stationary_distribution(transpose(P))

     # Compute relevant SVD info for P by computing eigendecomposition of P*P'
     L = L_mul_Lt(P, π)
     E = eigs(L; nev=n_coords, maxiter=1000, which=:LM)
     Σ = sqrt.(real.(E[1]))
     Ψ = real(E[2])

     # Compute diffusion map Ψ and extract the diffusion coordinates
     scale!(Ψ,Σ)
     @. π = 1/sqrt(π)
     scale!(π,Ψ)
     return Σ, Ψ
 end

 """
     diffusion_distance(diff_coord)

 Returns the distance matrix of pairs of points whose diffusion distances
 correspond to the diffusion coordinates given by `diff_coord`.
 """
 function diffusion_distance(Ψ::AbstractArray{T,2})::Symmetric{T,Array{T,2}} where T
     D = Dists.pairwise(Dists.Euclidean(),Ψ)
     return Symmetric(D)
 end
