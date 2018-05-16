# (c) 2018 Alvaro de Diego, with minor contributions by Daniel Karrasch

NN = NearestNeighbors
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

    balltree = NN.BallTree(A, metric)
    idxs = NN.inrange(balltree, A, ε, false)
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

function α_normalize!(A::AbstractSparseMatrix{T}, α::S = 0.5) where T <: Real where S <: Real
    LinAlg.checksquare(A)
    qₑ = spdiagm( (1./squeeze(sum(A, 2),2).^α))
    A .= qₑ * A * qₑ
    return A
end

function α_normalize!(A::DenseMatrix{T}, α::S = 0.5) where T <: Real where S <: Real
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

function wLap_normalize!(A::AbstractSparseMatrix)
    LinAlg.checksquare(A)
    dᵅ = spdiagm(1./squeeze(sum(A, 2),2))
    A .= dᵅ * A
    return A
end

function wLap_normalize!(A::DenseMatrix)
    LinAlg.checksquare(A)
    dᵅ = 1./squeeze(sum(A, 2),2)
    scale!(dᵅ,A)
    return A
 end
