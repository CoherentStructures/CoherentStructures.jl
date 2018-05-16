# (c) 2018 Alvaro de Diego & Daniel Karrasch

Dists = Distances

doc"""
    sparse_diff_op(sol, [dim], k, ε; metric)

Return a list of sparse diffusion/Markov matrices `P`.

## Arguments
   * `sol`: 2D array with columns correspdonding to data points;
   * if `dim` is given, the columns are interpreted as concatenations of `dim`-
     dimensional points, to which `metric` is applied individually;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
     only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function sparse_diff_op( sols::AbstractArray{T, 2},
                            dim::Int,
                            kernel::Function,
                            ε::T;
                            # mapper::Function = pmap,
                            metric::Dists.PreMetric = Dists.Euclidean()) where T <: Number
    dimt, N = size(sols)
    (q, r) = divrem(dimt,dim)
    @assert r == 0 "first dimension of solution matrix is not a multiple of spatial dimension $(dim)"

    P = pmap(1:q) do t
        # @time Pₜ = sparse_diff_op( view(sols,:,t,:), kernel, ε; metric = metric )
        @time Pₜ = sparse_diff_op( sols[(t-1)*dim+1:t*dim,:], kernel, ε; metric = metric )
        println("Timestep $t/$q done")
        Pₜ
    end
end

function sparse_diff_op(sols::AbstractArray{T, 2},
                        kernel::Function,
                        ε::T;
                        metric::Dists.PreMetric = Dists.Euclidean()) where T <: Number

    P = sparseaffinitykernel(collect(sols), kernel, metric, ε)
    α_normalize!(P, 1)
    wLap_normalize!(P)
    return P
end

"""
    sparseaffinitykernel(A, k, metric=Euclidean(), ε=1e-3)

Return a sparse matrix `K` where ``k_{ij} = k(x_i, x_j)``.
The ``x_i`` are taken from the columns of `A`. Entries are
only calculated for pairs where ``metric(x_i, x_j)≦ε``.
Default metric is `Euclidean()`, default `ε` is 1e-3.
"""

function sparseaffinitykernel( A::Array{T, 2},
                               kernel::F,
                               metric::Dists.PreMetric = Dists.Euclidean(),
                               ε::S = convert(S,1e-3) ) where T <: Number where S <: Number where F <:Function
    dim, N = size(A)

    balltree = NearestNeighbors.BallTree(A, metric)
    idxs = NearestNeighbors.inrange(balltree, A, ε, false)
    Js::Vector{Int} = vcat(idxs...)
    Is::Vector{Int} = vcat([fill(i,length(idxs[i])) for i in eachindex(idxs)]...)
    Vs::Vector{T} = kernel.(Dists.colwise(metric, view(A,:,Is), view(A,:,Js)))
    return sparse(Is,Js,Vs,N,N)
end

doc"""
    α_normalize!(A, α = 0.5)
Normalize rows and columns of `A` in-place with the respective row-sum to the α-th power;
i.e., return $ a_{ij}:=a_{ij}/q_i^{\\alpha}/q_j^{\\alpha}$, where
$ q_k = \\sum_{\\ell} a_{k\\ell}$. Default for `α` is 0.5.
"""

function α_normalize!(A::AbstractMatrix{T}, α::S = 0.5) where T <: Real where S <: Real
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

function wLap_normalize!(A::AbstractMatrix)
    LinAlg.checksquare(A)
    dᵅ = 1./squeeze(sum(A, 2),2)
    scale!(dᵅ,A)
    return A
 end

 """
     multiply_diffusion!(v, Pt)

 Overwrite `v` with `Pt[n] * Pt[n-1] * ... * Pt[1] * v`.
 """
 function multiply_diffusion!(v,Pt)
     efficient_linalg_reduce!(v, Pt, A_mul_B!)
 end

 """
     multiply_markov!(v,Pt)

 Overwrite `v` with  `(Pt[n] * Pt[n-1] * ... * Pt[1])' * v`.
 """
 function multiply_markov!(v, Pt)
     efficient_linalg_reduce!(v, reverse(Pt), At_mul_B!)
 end

 function multiply_diffusion(v,Pt)
    multiply_diffusion!(copy(v), Pt)
 end

 function multiply_markov(v, Pt)
     multiply_markov!(copy(v), Pt)
 end

 function efficient_linalg_reduce!(v, As, la_func!::T) where T<:Function
 # take e.g. la_func! = A_mul_B!
 # --> v is overwritten with A[n] * ... * A[1] * v
     w1 = copy(v)
     w2 = copy(v)

     for A in As
         la_func!(w1, A, w2)
         # swap pointers
         tmp = w1; w1 = w2; w2 = tmp
     end
      v .= w1
 end

 """
     stationary_distribution(P,N)

 Compute the stationary distribution for a Markov transition operator.
 `P` may be dense or sparse, or a `LinearMap` matrix-vector multiplication
 is given by a function.
 """

 function stationary_distribution(P::Union{SparseMatrixCSC{T,Int64},LinearMaps.LinearMap{T},DenseMatrix{T}}) where T <: Real

     E   = eigs(P; nev=1, which=:LM)
     π   = squeeze(real(E[2]),2) # stationary distribution
     ext = extrema(π)
     prod(ext) < zero(eltype(ext)) && throw(error("Both signs in stationary distribution"))
     if any(ext .< zero(eltype(ext)))
         π .= -π
     end
     return π
 end

 function L_mul_Lt(L::LinearMaps.LinearMap{T},π::Vector{T})::LinearMaps.LinearMap{T} where T <: Real

     πsqrt  = Diagonal(sqrt.(π)) # alternavtive: Diagonal(sqrt.(π))
     πinv   = Diagonal(1./π) # alternative: Diagonal(inv.(π))
     return πsqrt * L * πinv * transpose(L) * πsqrt
 end

 function L_mul_Lt(L::AbstractArray{T},π::Vector{T})::LinearMaps.LinearMap{T} where T <: Real

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

 function diffusion_coordinates(Pᵗ::AbstractVector{SparseMatrixCSC{T,Int64}},n_coords::Int) where T <: Real

     Nt = unique(LinAlg.checksquare(Pᵗ...))
     length(Nt) == 1 ? N = Nt[1] : throw(error("List of matrices contains nonsquare or incompatible matrices."))
     function fDiff!(w,v)
         w .= v
         multiply_diffusion!(w,Pᵗ)
     end

     function fMarkov!(w,v)
         w .= v
         multiply_markov!(w,Pᵗ)
     end

     P = LinearMaps.LinearMap(fDiff!,fMarkov!,N)
     return diffusion_coordinates(P,n_coords)
 end

 function diffusion_coordinates(P::S,n_coords::Int) where S <: Union{SparseMatrixCSC,LinearMaps.LinearMap,DenseMatrix}

     N = LinAlg.checksquare(P)

     π = stationary_distribution(transpose(P))

     # Compute relevant SVD info for P by computing eigendecomposition of P*P'
     L = L_mul_Lt(P, π)
     E = eigs(L; nev=n_coords, which=:LM)
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
