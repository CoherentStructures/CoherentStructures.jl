# (c) 2018-2019 Daniel Karrasch & Alvaro de Diego

"""
    gaussian(σ)

Returns the Euclidean heat kernel as a callable function
```math
x \\mapsto \\exp\\left(-\\frac{x^2}{4\\sigma}\\right)
```

## Example
```jldoctest
julia> kernel = gaussian(2.0);

julia> kernel(0.)
1.0
"""
function gaussian(σ::Real=1.0)
    let s = -1/4σ
        x -> exp(s*abs2(x))
    end
end

"""
    gaussiancutoff(σ, θ)

Computes the positive value at which [`gaussian(σ)`](@CoherentStructures.gaussian)
equals `θ`, i.e.,
```math
\\sqrt{-4\\sigma\\log(\\theta)}
```
"""
gaussiancutoff(σ::Real, cutoff::Real=eps()) = sqrt(-4σ*log(cutoff))

const LinMaps{T} = Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}

"""
    KNN(k) <: SparsificationMethod

Defines the KNN (k-nearest neighbors) sparsification method. In this
approach, first `k` nearest neighbors are sought. In the final graph Laplacian,
only those particle pairs are included which are contained in some
k-Neighborhood.
"""
struct KNN <: SparsificationMethod
    k::Int
end

"""
    MutualKNN(k) <: SparsificationMethod

Defines the mutual KNN (k-nearest neighbors) sparsification method. In this
approach, first `k` nearest neighbors are sought. In the final graph Laplacian,
only those particle pairs are included which are mutually contained in each
others k-Neighborhood.
"""
struct MutualKNN <: SparsificationMethod
    k::Int
end

"""
    Neighborhood(ε) <: SparsificationMethod

Defines the ε-Neighborhood sparsification method. In the final graph Laplacian,
only those particle pairs are included which have distance less than `ε`.
"""
struct Neighborhood{T <: Real} <: SparsificationMethod
    ε::T
end

# meta function
function DM_heatflow(flow_fun, p0, sp_method::SparsificationMethod, kernel;
                        metric::Dists.SemiMetric = Dists.Euclidean())

    data = pmap(flow_fun, p0; batch_size=ceil(sqrt(length(p0))))
    sparse_diff_op_family(data, sp_method, kernel; metric=metric)
end

# diffusion operator/graph Laplacian related functions

"""
    sparse_diff_op_family(data, sp_method, kernel=gaussian(); op_reduce, α, metric)

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `data`: a list of trajectories, each a list of states of type `SVector`;
   * `sp_method`: a [`sparsification method`](@SparsificationMethod);
   * `kernel`: diffusion kernel, e.g., [`gaussian(σ)`](@gaussian);
   * `op_reduce=P -> prod(LMs.LinearMap,Iterators.reverse(P))`: time-reduction of
     diffusion operators, e.g. `mean` (space-time diffusion maps), `P -> max.(P...)`
     (network-based coherence) or the default (time coupled diffusion maps)

## Keyword arguments
   * `α=1`: exponent in diffusion-map normalization;
   * `metric=Euclidean()`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``;
   * `verbose=false`: whether to print intermediate progress reports.
"""
function sparse_diff_op_family(data::AbstractVector{<:AbstractVector{<:SVector}},
                                sp_method::SparsificationMethod,
                                kernel = gaussian(),
                                op_reduce::Function = (P -> prod(reverse(LMs.LinearMap.(P))));
                                α::Real=1,
                                metric::Dists.SemiMetric = Dists.Euclidean(),
                                verbose::Bool=false
                                )
    N = length(data) # number of trajectories
    N == 0 && throw("no data available")
    q = length(data[1]) # number of time steps
    q == 0 && throw("trajectories have length 0")
    timeslices = map(i -> getindex.(data, i), 1:q)
    P = Distributed.pmap(enumerate(timeslices)) do (t, timeslice)
        Pt = sparse_diff_op(timeslice, sp_method, kernel; α=α, metric=metric)
        verbose && println("time step $t/$q done")
        Pt
    end
    return op_reduce(P)
end

"""
    sparse_diff_op(data, sp_method, kernel; α=1.0, metric=Euclidean()) -> SparseMatrixCSC

Return a sparse diffusion/Markov matrix `P`.

## Arguments
   * `data`: a list of trajectories, each a list of states of type `SVector`, or
     a list of states of type `SVector`;
   * `sp_method`: a [`sparsification method`](@SparsificationMethod);
   * `kernel`: diffusion kernel, e.g., [`gaussian(σ)`](@gaussian);

## Keyword arguments
   * `α`: exponent in diffusion-map normalization;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where ``metric(x_i, x_j)\\leq \\varepsilon``.
"""
function sparse_diff_op(data::Union{T, AbstractVector{T}},
                        sp_method::SparsificationMethod,
                        kernel = gaussian();
                        α::Real=1.0,
                        metric::Dists.SemiMetric = Dists.Euclidean()
                        ) where {T<:AbstractVector{<:SVector}}
    P = spdist(data, sp_method, metric)
    N = LinearAlgebra.checksquare(P)
    if sp_method isa Neighborhood # P corresponds to the adjacency matrix
        if kernel != Base.one # otherwise no need to change entries
            rows = SparseArrays.rowvals(P)
            vals = SparseArrays.nonzeros(P)
            for i = 1:N
               for j in SparseArrays.nzrange(P, i)
                  vals[j] = kernel(Dists.evaluate(metric, data[rows[j]], data[i]))
               end
            end
        end
    else # sp_method isa *KNN (P already contains the distances), need to apply kernel
        vals = SparseArrays.nonzeros(P)
        vals .= kernel.(vals)
    end
    droptol!(P, eps(eltype(P)))
    @inbounds kde_normalize!(P, α)
    row_normalize!(P)
    return P
end

"""
    kde_normalize!(A, α = 1)
Normalize rows and columns of `A` in-place with the respective row-sum to the α-th power;
i.e., return ``a_{ij}:=a_{ij}/q_i^{\\alpha}/q_j^{\\alpha}``, where
``q_k = \\sum_{\\ell} a_{k\\ell}``. Default for `α` is `1`.
"""
@inline function kde_normalize!(A::SparseMatrixCSC{<:Real}, α::Real=1)
    if α == 0
        return A
    end
    n = LinearAlgebra.checksquare(A)
    qₑ = A * ones(eltype(A), size(A, 2))
    if α == 1
        qₑ .= inv.(qₑ)
    elseif α == 1/2
        qₑ .= inv.(sqrt.(qₑ))
    else
        qₑ .= inv.(qₑ.^α)
    end
    Anzval = A.nzval
    Arowval = A.rowval
    @inbounds for col = 1:n, p = SparseArrays.nzrange(A, col)
        Anzval[p] = qₑ[Arowval[p]] * Anzval[p] * qₑ[col]
    end
    return A
end
@inline function kde_normalize!(A::AbstractMatrix{<:Real}, α::Real=1)
    if α == 0
        return A
    end
    @boundscheck LinearAlgebra.checksquare(A)
    qₑ = A * ones(eltype(A), size(A, 2))
    if α == 1
        qₑ .= inv.(qₑ)
    elseif α == 1/2
        qₑ .= inv.(sqrt.(qₑ))
    else
        qₑ .= inv.(qₑ.^α)
    end
    A .= qₑ .* A .* permutedims(qₑ)
    return A
end

"""
    row_normalize!(A)
Normalize rows of `A` in-place with the respective row-sum; i.e., return
``a_{ij}:=a_{ij}/q_i``.
"""
@inline function row_normalize!(A::AbstractMatrix)
    # this should be once Julia PR #30208 is backported:
    # ldiv!(Diagonal(A * ones(eltype(A), size(A, 2))), A)
    dᵅ = Diagonal(inv.(A * ones(eltype(A), size(A, 2))))
    lmul!(dᵅ, A)
    return A
end

# spectral clustering/diffusion map related functions

 """
     stationary_distribution(P) -> Vector

 Compute the stationary distribution for a Markov transition operator.
 `P` may be dense or sparse, or a `LinearMap` matrix-vector multiplication
 is given by a function.
 """
function stationary_distribution(P::LinMaps{T}) where T <: Real
     E = Arpack.eigs(P; nev=1, ncv=50)
     Π = dropdims(real(E[2]), dims=2) # stationary distribution
     ext = extrema(Π)
     if (prod(ext) < 0) && (all(abs.(ext) .> eps(eltype(ext))))
         throw(error("Both signs in stationary distribution (extrema are $ext)"))
     end
     Π .= abs.(Π)
     return Π
 end

 @inline function L_mul_Lt(L::LMs.LinearMap{T},
                            Π::Vector{T})::LMs.LinearMap{T} where T <: Real
     Πsqrt = Diagonal(sqrt.(Π))
     Πinv  = Diagonal(inv.(Π))
     return LMs.LinearMap(Πsqrt * L * Πinv * transpose(L) * Πsqrt;
                    issymmetric=true, ishermitian=true, isposdef=true)
 end
 @inline function L_mul_Lt(L::AbstractMatrix{T},
                            Π::Vector{T})::LMs.LinearMap{T} where T <: Real
     L .= sqrt.(Π) .* L .* permutedims(inv.(sqrt.(Π)))
     LMap = LMs.LinearMap(L)
     return LMs.LinearMap(LMap * transpose(LMap); issymmetric=true,
                ishermitian=true, isposdef=true)
 end

 """
     diffusion_coordinates(P, n_coords) -> (Σ::Vector, Ψ::Matrix)

 Compute the (time-coupled) diffusion coordinates `Ψ` and the coordinate weights
 `Σ` for a linear map `P`. `n_coords` determines the number of diffusion
 coordinates to be computed.
 """
function diffusion_coordinates(P::LinMaps, n_coords::Int)
    N = LinearAlgebra.checksquare(P)
    n_coords <= N || throw(error("number of requested coordinates, $n_coords, too large, only $N samples available"))
    Π = stationary_distribution(transpose(P))

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

    Σ .= sqrt.(abs.(Σ))
    Ψ = E[2]

    # Compute diffusion map Ψ and extract the diffusion coordinates
    rmul!(Ψ, Diagonal(Σ))
    @. Π = 1 / sqrt(Π)
    lmul!(Diagonal(Π), Ψ)
    return Σ, Ψ
end

 """
     diffusion_distance(diff_coord) -> SymmetricMatrix

 Returns the distance matrix of pairs of points whose diffusion distances
 correspond to the diffusion coordinates given by `diff_coord`.
 """
 function diffusion_distance(Ψ::AbstractMatrix)
     D = Dists.pairwise(Dists.Euclidean(), Ψ)
     return D
 end
