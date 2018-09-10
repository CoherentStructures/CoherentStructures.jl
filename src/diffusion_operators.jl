# (c) 2018 Alvaro de Diego & Daniel Karrasch

const gaussian_kernel = x::Number -> exp(-abs2(x))

const LinMaps{T} = Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}
const NN = NearestNeighbors

"""
    KNN(k)

Defines the KNN (k-nearest neighbors) sparsification method. In this
approach, first `k` nearest neighbors are sought. In the final graph Laplacian,
only those particle pairs are included which are contained in some
k-neighborhood.
"""
struct KNN <: SparsificationMethod
    k::Int
end

"""
    mutualKNN(k)

Defines the mutual KNN (k-nearest neighbors) sparsification method. In this
approach, first `k` nearest neighbors are sought. In the final graph Laplacian,
only those particle pairs are included which are mutually contained in each
others k-neighborhood.
"""
struct mutualKNN <: SparsificationMethod
    k::Int
end

"""
    neighborhood(ε)

Defines the ε-neighborhood sparsification method. In the final graph Laplacian,
only those particle pairs are included which have distance less than `ε`.
"""
struct neighborhood{T <: Real} <: SparsificationMethod
    ε::T
end

# meta function
function DM_heatflow(flow_fun,
                        p0,
                        sp_method::S,
                        k,
                        dim::Int = 2;
                        metric::Distances.Metric = Distances.Euclidean()
                        ) where S <: SparsificationMethod

    data = parallel_flow(flow_fun, p0)
    sparse_diff_op_family(data, sp_method, k, dim; metric=metric)
end

# diffusion operator/graph Laplacian related functions
"""
    diff_op(data, sp_method, kernel = gaussian_kernel; α=1.0, metric=Euclidean"()")

Return a diffusion/Markov matrix `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `sp_method`: employed sparsification method ([`neighborhood`](@ref) or [`mutualKNN`](@ref));
   * `kernel`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``.
"""
function diff_op(data::AbstractMatrix{T},
                    sp_method::neighborhood,
                    kernel = gaussian_kernel;
                    α=1.0,
                    metric::Distances.Metric = Distances.Euclidean()
                )::SparseMatrixCSC{T,Int} where T <: Real

    N = size(data, 2)
    D = Distances.pairwise(metric,data)
    CIs::Vector{CartesianIndex{2}} = findall(D .<= sp_method.ε)
    Is::Vector{Int} = [i[1] for i in CIs]
    Js::Vector{Int} = [j[2] for j in CIs]
    Vs::Vector{T} = kernel.(D[CIs])
    P = SparseArrays.sparse(Is, Js, Vs, N, N)
    !iszero(α) && α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

function diff_op(data::AbstractMatrix{T},
                    sp_method::Union{KNN,mutualKNN},
                    kernel = gaussian_kernel;
                    α=1.0,
                    metric::Distances.Metric = Distances.Euclidean()
                )::SparseMatrixCSC{T,Int} where T <: Real

    N, k = size(data, 2), sp_method.k
    D = Distances.pairwise(metric,data)
    Is = SharedArray{Int}(N*(k+1))
    Js = SharedArray{Int}(N*(k+1))
    Vs = SharedArray{T}(N*(k+1))
    index = Vector{Int}(undef, k+1)
    Distributed.@everywhere index = $index
    @inbounds @sync Distributed.@distributed for i=1:N
        di = view(D,i,:)
        partialsortperm!(index, di, 1:(k+1))
        Is[(i-1)*(k+1)+1:i*(k+1)] .= i
        Js[(i-1)*(k+1)+1:i*(k+1)] = index
        Vs[(i-1)*(k+1)+1:i*(k+1)] = kernel.(di[index])
    end
    P = SparseArrays.sparse(Is, Js, Vs, N, N)
    if typeof(sp_method) <: KNN
        @. P = max(P, PermutedDimsArray(P, (2,1)))
    else
        @. P = min(P, PermutedDimsArray(P, (2,1)))
    end
    α>0 && α_normalize!(P, α)
    wLap_normalize!(P)
    return P
end

"""
    sparse_diff_op_family(data, sp_method, kernel=gaussian_kernel, dim=2; op_reduce, α, metric)

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `sp_method`: sparsification method;
   * `kernel`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `dim`: the columns are interpreted as concatenations of `dim`-
     dimensional points, to which `metric` is applied individually;
   * `op_reduce`: time-reduction of diffusion operators, e.g. `mean` or
     `P -> prod(LinearMaps.LinearMap,Iterators.reverse(P))` (default)
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``.
"""
function sparse_diff_op_family( data::AbstractMatrix,
                                sp_method::S,
                                kernel = gaussian_kernel,
                                dim::Int = 2;
                                op_reduce::Function = (P -> prod(reverse(LinearMaps.LinearMap.(P)))),
                                α=1.0,
                                metric::Distances.Metric = Distances.Euclidean()
                                ) where {S <: SparsificationMethod}
    dimt, N = size(data)
    q, r = divrem(dimt, dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    P = Distributed.pmap(1:q) do t
        @time Pₜ = sparse_diff_op(data[(t-1)*dim+1:t*dim,:], sp_method, kernel;
                                    α=α, metric=metric)
        # println("Timestep $t/$q done")
        # Pₜ
    end
    @time P = op_reduce(P)
    return P
end

"""
    sparse_diff_op(data, sp_method, kernel; α=1.0, metric=Euclidean()) -> SparseMatrixCSC

Return a sparse diffusion/Markov matrix `P`.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `sp_method`: sparsification method;
   * `kernel`: diffusion kernel, e.g., `x -> exp(-x*x)` (default);
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``.
"""
@inline function sparse_diff_op(data::AbstractMatrix,
                        sp_method::S,
                        kernel = gaussian_kernel;
                        α=1.0,
                        metric::Distances.Metric = Distances.Euclidean()
                        ) where {S <: SparsificationMethod}

    typeof(metric) == STmetric && metric.p < 1 && throw("Cannot use balltrees for sparsification with $(metric.p)<1.")
    P = sparseaffinitykernel(data, sp_method, kernel, metric)
    α>0 && α_normalize!(P, α)
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
@inline function sparseaffinitykernel(data::AbstractMatrix{T},
                               sp_method::Union{KNN, mutualKNN},
                               kernel=gaussian_kernel,
                               metric::Distances.PreMetric=Distances.Euclidean()
                               ) where T <: Real
    dim, N = size(data)

    if typeof(metric) <: NN.MinkowskiMetric
        tree = NN.KDTree(data, metric;  leafsize = 10)
    else
        tree = NN.BallTree(data, metric; leafsize = 10)
    end
    idxs::Vector{Vector{Int}}, dists::Vector{Vector{T}} = NN.knn(tree, data, sp_method.k, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
    Vs = kernel.(vcat(dists...))
    W = SparseArrays.sparse(Is, Js, Vs, N, N)
    SparseArrays.droptol!(W, eps(eltype(W)))
    if typeof(sp_method) <: KNN
        return max.(W, permutedims(W))
    else
        return min.(W, permutedims(W))
    end
end

@inline function sparseaffinitykernel(data::AbstractMatrix{T},
                               sp_method::neighborhood,
                               kernel,
                               metric::Distances.PreMetric = Distances.Euclidean()
                               ) where T <: Real
    dim, N = size(data)
    # TODO: check for better leafsize values
    if typeof(metric) <: NN.MinkowskiMetric
        tree = NN.KDTree(data, metric;  leafsize = 10)
    else
        tree = NN.BallTree(data, metric; leafsize = 10)
    end
    idxs = NN.inrange(tree, data, sp_method.ε, false)
    Js = vcat(idxs...)
    Is = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
    Vs = kernel.(Distances.colwise(metric, view(data, :, Is), view(data, :, Js)))
    return SparseArrays.sparse(Is, Js, Vs, N, N)
end

"""
    α_normalize!(A, α = 1.0)
Normalize rows and columns of `A` in-place with the respective row-sum to the α-th power;
i.e., return ``a_{ij}:=a_{ij}/q_i^{\\alpha}/q_j^{\\alpha}``, where
``q_k = \\sum_{\\ell} a_{k\\ell}``. Default for `α` is 1.0.
"""
@inline function α_normalize!(A::TA, α=1.0) where {TA <: AbstractMatrix{T} where {T <: Real}}
    LinearAlgebra.checksquare(A)
    qₑ = Diagonal(dropdims(sum(A, dims=2), dims=2) .^-α)
    LinearAlgebra.rmul!(A, qₑ)
    LinearAlgebra.lmul!(qₑ, A)
    return A
end

"""
    wLap_normalize!(A)
Normalize rows of `A` in-place with the respective row-sum; i.e., return
``a_{ij}:=a_{ij}/q_i``.
"""
@inline function wLap_normalize!(A::TA) where {TA <: AbstractMatrix{T} where {T <: Real}}
    LinearAlgebra.checksquare(A)
    dᵅ = Diagonal(inv.(dropdims(sum(A, dims=2), dims=2)))
    LinearAlgebra.lmul!(dᵅ, A)
    return A
 end

# adjacency-related functions

"""
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
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``.
"""
function sparse_adjacency(data::AbstractMatrix{T},
                            ε::S,
                            dim::Int;
                            metric::Distances.Metric = Distances.Euclidean()
                        )::SparseMatrixCSC{Bool,Int} where {T <: Real, S <: Real}
    dimt, N = size(data)
    q, r = divrem(dimt, dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    IJs = Distributed.pmap(1:q) do t
        I, J = sparse_adjacency_list( data[(t-1)*dim+1:t*dim,:], ε; metric = metric )
        # println("Timestep $t/$q done")
        # I, J
    end
    Is::Vector{Int} = vcat([ijs[1] for ijs in IJs]...)
    Js::Vector{Int} = vcat([ijs[2] for ijs in IJs]...)
    Vs::Vector{Bool} = fill(1.0, length(Is))
    return SparseArrays.sparse(Is, Js, Vs, N, N, *)
end

function sparse_adjacency(data::AbstractMatrix,
                            ε::S;
                            metric::Distances.Metric = Distances.Euclidean()
                            )::SparseMatrixCSC{Bool,Int} where {S <: Real}
    dimt, N = size(data)
    q, r = divrem(dimt, dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"
    Is, Js = sparse_adjacency_list(data, ε; metric=metric)
    Vs = fill(1.0, length(I))
    return SparseArrays.sparse(Is, Js, Vs, N, N, *)
end

"""
    sparse_adjacency_list(data, ε; metric=Euclidean()) -> idxs::Vector{Vector}

Return two lists of indices of data points that are adjacent.

## Arguments
   * `data`: 2D array with columns correspdonding to data points;
   * `ε`: distance threshold;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``.
"""
@inline function sparse_adjacency_list(data::AbstractMatrix{T},
                                ε::S;
                                metric::Distances.Metric = Distances.Euclidean()
                               )::Tuple{Vector{Int},Vector{Int}} where {T <: Real, S <: Real}

    typeof(metric) <: STmetric && metric.p < 1 && throw("Cannot use balltrees for sparsification with $(metric.p)<1.")
    if typeof(metric) <: NN.MinkowskiMetric
        tree = NN.KDTree(data, metric;  leafsize = metric == STmetric ? 20 : 10)
    else
        tree = NN.BallTree(data, metric; leafsize = metric == STmetric ? 20 : 10)
    end
    idxs = NN.inrange(tree, data, ε, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i, length(idxs[i])) for i in eachindex(idxs)]...)
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

     E = Arpack.eigs(P; nev=1, ncv=50)
     Π = dropdims(real(E[2]), dims=2) # stationary distribution
     ext = extrema(Π)
     prod(ext) < 0 && throw(error("Both signs in stationary distribution (extrema are $ext)"))
     if (ext[1] < 0)
         Π .= -Π
     end
     return Π
 end

 @inline function L_mul_Lt(L::LinearMaps.LinearMap{T},
                            Π::Vector{T})::LinearMaps.LinearMap{T} where T <: Real

     Πsqrt = Diagonal(sqrt.(Π))
     Πinv  = Diagonal(inv.(Π))
     return LinearMaps.LinearMap(Πsqrt * L * Πinv * LinearAlgebra.transpose(L) * Πsqrt;
                    issymmetric=true, ishermitian=true, isposdef=true)
 end

 @inline function L_mul_Lt(L::AbstractMatrix{T},
                            Π::Vector{T})::LinearMaps.LinearMap{T} where T <: Real

     Πsqrt = Diagonal(sqrt.(Π))
     Πinvsqrt = Diagonal(inv.(Πsqrt))
     LinearAlgebra.lmul!(Πsqrt, L)
     LinearAlgebra.rmul!(L, Πinvsqrt)
     LMap = LinearMaps.LinearMap(L)
     return LinearMaps.LinearMap(LMap * LinearAlgebra.transpose(LMap); issymmetric=true,
                ishermitian=true, isposdef=true)
 end

 """
     diffusion_coordinates(P,n_coords) -> (Σ::Vector, Ψ::Matrix)

 Compute the (time-coupled) diffusion coordinates `Ψ` and the coordinate weights
 `Σ` for a linear map `P`. `n_coords` determines the number of diffusion
 coordinates to be computed.
 """
function diffusion_coordinates(P::LinMaps,n_coords::Int)
    N = LinearAlgebra.checksquare(P)

    Π = stationary_distribution(LinearAlgebra.transpose(P))

    # Compute relevant SVD info for P by computing eigendecomposition of P*P'
    L = L_mul_Lt(P, Π)
    E = Arpack.eigs(L; nev=n_coords, ncv=max(50, 2*n_coords+1))

    # eigenvalues close to zero can be negative even though they
    # should be positive.
    drop_num_zeros(x) = abs(x) < eps(E[1][1]) ? zero(x) : x
    Σ = drop_num_zeros.(E[1])

    if any(Σ .< 0)
        @warn "Negative eigenvalue bigger than eps($(Σ[1]))in $(Σ)! "*
        "Using absolute value instead."
    end

    Σ .= (sqrt∘abs).(Σ)
    Ψ = E[2]

    # Compute diffusion map Ψ and extract the diffusion coordinates
    LinearAlgebra.rmul!(Ψ, Diagonal(Σ))
    @. Π = 1 / sqrt(Π)
    LinearAlgebra.lmul!(Diagonal(Π), Ψ)
    return Σ, Ψ
end

 """
     diffusion_distance(diff_coord) -> SymmetricMatrix

 Returns the distance matrix of pairs of points whose diffusion distances
 correspond to the diffusion coordinates given by `diff_coord`.
 """
 function diffusion_distance(Ψ::AbstractMatrix{T})::Symmetric{T,Array{T,2}} where T
     D = Distances.pairwise(Distances.Euclidean(), Ψ)
     return Symmetric(D)
 end
