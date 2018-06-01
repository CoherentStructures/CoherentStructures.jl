# (c) 2018 Alvaro de Diego & Daniel Karrasch

const gaussian_kernel = x -> exp(-abs2(x))

const LinMaps{T} = Union{SparseMatrixCSC{T,Int},LinearMaps.LinearMap{T},DenseMatrix{T}}

```
    struct mutualKNN(k)

Defines the mutual KNN (k-nearest neighbors) sparsification method. In this
approach, first `k` nearest neighbors are sought. In the final graph Laplacian,
only those particle pairs are included which are mutually contained in each
others k-neighborhood.
```
struct mutualKNN <: SparsificationMethod
    k::Int
end

```
    struct neighborhood(ϵ)

Defines the ϵ-neighborhood sparsification method. In the final graph Laplacian,
only those particle pairs are included which have distance less than `ϵ`.
```

struct neighborhood{T <: Real} <: SparsificationMethod
    ϵ::T
end

# meta function
function DM_heatflow(flow_fun::Function,
                        p0,
                        sp_method::S,
                        k::F,
                        dim::Int = 2;
                        metric::Distances.PreMetric = Distances.Euclidean()
                        ) where {F <: Function, S <: SparsificationMethod}

    data = parallel_flow(flow_fun,p0)
    sparse_diff_op_family(data, sp_method, k, dim; metric=metric)
end

# diffusion operator/graph Laplacian related functions
doc"""
    diff_op(data, sp_method, kernel = gaussian_kernel; α=1.0, metric=Euclidean"()")

Return a diffusion/Markov matrix `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `sp_method`: employed sparsification method ([`neighborhood`](@ref) or [`mutualKNN`](@ref));
   * `kernel`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function diff_op(data::AbstractArray{T, 2},
                    sp_method::neighborhood,
                    kernel::F = gaussian_kernel;
                    α=1.0,
                    metric::Distances.PreMetric = Distances.Euclidean()
                )::SparseMatrixCSC{T,Int} where {T <: Real, F <: Function}

    N = size(data, 2)
    D = Distances.pairwise(metric,data)
    Is, Js = findn(D .< sp_method.ϵ)
    Vs = kernel.(D[D .< sp_method.ϵ])
    P = sparse(Is, Js, Vs, N, N)
    α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

function diff_op(data::AbstractArray{T, 2},
                    sp_method::mutualKNN,
                    kernel::F = gaussian_kernel;
                    α=1.0,
                    metric::Distances.PreMetric = Distances.Euclidean()
                )::SparseMatrixCSC{T,Int} where {T <: Real, F <: Function}

    N, k = size(data, 2), sp_method.k
    D = Distances.pairwise(metric,data)
    Is = SharedArray{Int}(N*(k+1))
    Js = SharedArray{Int}(N*(k+1))
    Vs = SharedArray{T}(N*(k+1))
    index = Vector{Int}(k+1)
    @everywhere @eval index = $index
    @inbounds @sync @parallel for i=1:N
        di = view(D,i,:)
        selectperm!(index, di, 1:(k+1))
        Is[(i-1)*(k+1)+1:i*(k+1),1] = i
        Js[(i-1)*(k+1)+1:i*(k+1),2] = index
        Vs[(i-1)*(k+1)+1:i*(k+1),2] = kernel.(di[index])
    end
    P = sparse(Is, Js, Vs, N, N)
    @. P = min(P, transpose(P))
    α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

doc"""
    sparse_diff_op_family(data, sp_method, kernel=gaussian_kernel, dim=2; op_reduce, α, metric)

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `ε`: distance threshold;
   * if `dim` is given, the columns are interpreted as concatenations of `dim`-
     dimensional points, to which `metric` is applied individually;
   * `op_reduce`: time-reduction of diffusion operators, e.g. `mean` or
     `P -> prod(LinearMaps.LinearMap,reverse(P))` (default)
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function sparse_diff_op_family( data::AbstractArray{T, 2},
                                sp_method::S,
                                kernel::F = gaussian_kernel,
                                dim::Int = 2;
                                op_reduce::Function = P -> prod(LinearMaps.LinearMap,reverse(P)),
                                α=1.0,
                                metric::Distances.PreMetric = Distances.Euclidean()
                                ) where {T <: Number, S <: SparsificationMethod, F <: Function}
    dimt, N = size(data)
    (q, r) = divrem(dimt,dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    P = pmap(1:q) do t
        Pₜ = sparse_diff_op( data[(t-1)*dim+1:t*dim,:], sp_method, kernel;
                                    α=α, metric = metric )
        println("Timestep $t/$q done")
        Pₜ
    end
    return op_reduce(P)
end

doc"""
    sparse_diff_op(data, sp_method, kernel; α=1.0, metric=Euclidean()) -> SparseMatrixCSC

Return a sparse diffusion/Markov matrix `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `sp_method`: distance threshold;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x)` (default);
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

@inline function sparse_diff_op(data::AbstractArray{T, 2},
                        sp_method::S,
                        kernel::F = gaussian_kernel;
                        α=1.0,
                        metric::Distances.PreMetric = Distances.Euclidean()
                        ) where {T <: Real, S <: SparsificationMethod, F <: Function}

    typeof(metric) == STmetric && metric.p < 1 && throw("Cannot use balltrees for sparsification with $(metric.p)<1.")
    P = sparseaffinitykernel(data, sp_method, kernel, metric)
    α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

"""
    sparseaffinitykernel(data, sp_method, kernel, metric=Euclidean()) -> SparseMatrixCSC

Return a sparse matrix `W` where ``w_{ij} = k(x_i, x_j)``.
The ``x_i`` are taken from the columns of `data`. Entries are
only calculated for pairs determined by the sparsification method `sp_method`.
Default metric is `Euclidean()`.
"""

@inline function sparseaffinitykernel(data::Array{T, 2},
                               sp_method::mutualKNN,
                               kernel::F,
                               metric::Distances.PreMetric = Distances.Euclidean()
                               ) where {T <: Real, F <:Function}
    dim, N = size(data)

    if typeof(metric) <: NearestNeighbors.MinkowskiMetric
        tree = NearestNeighbors.KDTree(data, metric;  leafsize = metric == STmetric ? 20 : 10)
    else
        tree = NearestNeighbors.BallTree(data, metric; leafsize = metric == STmetric ? 20 : 10)
    end
    idxs, dists = NearestNeighbors.knn(tree, data, sp_method.k, false)
    J = vcat(idxs...)
    I = vcat([fill(i,length(idxs[i])) for i in eachindex(idxs)]...)
    V = kernel.(vcat(dists...))
    W = sparse(I,J,V,N,N)
    Base.SparseArrays.droptol!(W,eps(T))
    return min.(W, transpose(W))
end

@inline function sparseaffinitykernel(data::Array{T, 2},
                               sp_method::neighborhood,
                               kernel::F,
                               metric::Distances.PreMetric = Distances.Euclidean()
                               ) where {T <: Real, F <:Function}
    dim, N = size(data)

    if typeof(metric) <: NearestNeighbors.MinkowskiMetric
        tree = NearestNeighbors.KDTree(data, metric;  leafsize = metric == STmetric ? 20 : 10)
    else
        tree = NearestNeighbors.BallTree(data, metric; leafsize = metric == STmetric ? 20 : 10)
    end
    idxs = NearestNeighbors.inrange(tree, data, sp_method.ϵ, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i,length(idxs[i])) for i in eachindex(idxs)]...)
    Vs::Vector{T} = kernel.(Distances.colwise(metric, view(data,:,Is), view(data,:,Js)))
    return sparse(Is,Js,Vs,N,N)
end

doc"""
    α_normalize!(A, α = 0.5)
Normalize rows and columns of `A` in-place with the respective row-sum to the α-th power;
i.e., return $ a_{ij}:=a_{ij}/q_i^{\\alpha}/q_j^{\\alpha}$, where
$ q_k = \\sum_{\\ell} a_{k\\ell}$. Default for `α` is 0.5.
"""

@inline function α_normalize!(A::T, α::S = 0.5) where {T <: AbstractMatrix, S <: Real}
    LinAlg.checksquare(A)
    qₑ = 1./squeeze(sum(A,2), 2).^α
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
    dᵅ = 1./squeeze(sum(A,2), 2)
    scale!(dᵅ,A)
    return A
 end

# adjacency-related functions

doc"""
    sparse_adjacency(data, ε[, dim]; metric) -> SparseMatrixCSC

Return a sparse adjacency matrix `A` with integer entries `0` or `1`. If the
third argument `dim` is passed, then `data` is interpreted as concatenated
points of length `dim`, to which `metric` is applied individually. Otherwise,
metric is applied to the whole columns of `data`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `ε`: distance threshold;
   * `dim`: the columns of `data` are interpreted as concatenations of `dim`-
     dimensional points, to which `metric` is applied individually;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function sparse_adjacency(data::AbstractArray{T, 2},
                            ε::S,
                            dim::Int;
                            metric::Distances.PreMetric = Distances.Euclidean()
                        )::SparseMatrixCSC{Bool,Int} where {T <: Real, S <: Real}
    dimt, N = size(data)
    (q, r) = divrem(dimt,dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    IJs = pmap(1:q) do t
        I, J = sparse_adjacency_list( data[(t-1)*dim+1:t*dim,:], ε; metric = metric )
        println("Timestep $t/$q done")
        I, J
    end
    I = vcat([ijs[1] for ijs in IJs]...)
    J = vcat([ijs[2] for ijs in IJs]...)
    V = fill(true,length(I))
    # no need for unique indices: for Boolean-values, multiply occuring indices
    # are reduced by |
    return sparse(I,J,V,N,N)
end

function sparse_adjacency(data::AbstractArray{T, 2},
                            ε::S;
                            metric::Distances.PreMetric = Distances.Euclidean()
                            )::SparseMatrixCSC{Bool,Int} where {T <: Real, S <: Real}
    dimt, N = size(data)
    (q, r) = divrem(dimt,dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"
    I, J = sparse_adjacency_list( data, ε; metric = metric )
    V = fill(true,length(I))
    return sparse(I,J,V,N,N)
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
                                metric::Distances.PreMetric = Distances.Euclidean()
                               )::Tuple{Vector{Int},Vector{Int}} where {T <: Real, S <: Real}

    typeof(metric) == STmetric && metric.p < 1 && throw("Cannot use balltrees for sparsification with $(metric.p)<1.")
    balltree = NearestNeighbors.BallTree(data, metric)
    idxs = NearestNeighbors.inrange(balltree, data, ε, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i,length(idxs[i])) for i in eachindex(idxs)]...)
    return Is, Js
end

# spectral clustering/diffusion map related functions

 """
     stationary_distribution(P) -> Vector

 Compute the stationary distribution for a Markov transition operator.
 `P` may be dense or sparse, or a `LinearMap` matrix-vector multiplication
 is given by a function.
 """

 function stationary_distribution(P::LinMaps{T})::Vector{T} where T <: Real

     E   = eigs(P; nev=1, maxiter=1000, which=:LM)
     π   = squeeze(real(E[2]),2) # stationary distribution
     ext = extrema(π)
     prod(ext) < 0 && throw(error("Both signs in stationary distribution"))
     if any(ext .< 0)
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
     diffusion_coordinates(P,n_coords) -> (Σ::Vector, Ψ::Matrix)

 Compute the (time-coupled) diffusion coordinates `Ψ` and the coordinate weights
 `Σ` for a linear map `P`. `n_coords` determines the number of diffusion
 coordinates to be computed.
 """

 function diffusion_coordinates(P::LinMaps,n_coords::Int)::Tuple{Vector{Float64},Array{Float64}}

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
     diffusion_distance(diff_coord) -> SymmetricMatrix

 Returns the distance matrix of pairs of points whose diffusion distances
 correspond to the diffusion coordinates given by `diff_coord`.
 """
 function diffusion_distance(Ψ::AbstractArray{T,2})::Symmetric{T,Array{T,2}} where {T}
     D = Distances.pairwise(Distances.Euclidean(),Ψ)
     return Symmetric(D)
 end
