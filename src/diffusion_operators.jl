# (c) 2018 Alvaro de Diego, with minor contributions by Daniel Karrasch

NN = NearestNeighbors
Dists = Distances

doc"""
    sparse_diff_op(sol, k, ε; metric)

Return a list of sparse diffusion/Markov matrices `P`.
   * `sol`: 2D or 3D array of trajectories of size `(dim,N)` and `(dim,q,N)`, resp.,
   where `dim` is the spatial dimension, `q` is the number of time steps,
   and `N` is the number of trajectories;
   * `k`: diffusion kernel, e.g., `x -> exp(-x*x/4σ)`;
   * `metric`: distance function w.r.t. which the kernel is computed, however,
   only for point pairs where $ metric(x_i, x_j)\leq \varepsilon$.
"""

function sparse_diff_op( sols::AbstractArray{T, 3},
                                   kernel::Function,
                                   ε::T;
                                   # mapper::Function = pmap,
                                   metric = Dists.Euclidean()) where T
    dim, q, N = size(sols)

    P = pmap(1:q) do t
        @time Pₜ = sparse_diff_op( sols[:,t,:], kernel, ε; metric = metric )
        println("Timestep $t/$q done")
        Pₜ
    end
end

function sparse_diff_op(sols::AbstractArray{T, 2},
                        kernel::Function,
                        ε::T;
                        metric = Dists.Euclidean()) where T
    dim, N = size(sols)

    # P = sparseaffinitykernel((@views sols[:,t,:]),
    P = sparseaffinitykernel( sols, kernel, metric, ε )
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

function sparseaffinitykernel( A::AbstractArray{T, 2},
                               kernel::F,
                               metric=Dists.Euclidean(),
                               ε::S=convert(S,1e-3) ) where T where S where F <:Function
    dim, N = size(A)

    # the "collect" is necessary if an array-view is passed because BallTree
    # currently cannot handle abstract arrays.
    # balltree = NN.BallTree(collect(A), metric)
    balltree = NN.BallTree(A, metric)

    # Get a Vector of tuples (I,J,V), one for each column i.
    # We hardcode the type because the compiler can't infer it (function barrier
    # and list comprehension don't help) -> check every once in a while, if this
    # has changed. (24.04.18)
    matrix_data_chunks = Array{Tuple{Vector{Int}, Vector{Int}, Vector{T}}}(N)
    matrix_data_chunks[:] = @views map(1:N) do i
        Js = NN.inrange(balltree, A[:,i], ε)
        Vs = kernel.(Dists.colwise(metric, A[:,i], A[:,Js]))
        Is = fill(i,length(Js))
        # return the resulting I,J,V for the sparse matrix
        Is, Js, Vs
    end

    # concatenate the lists of collected indices and values
    # reduce((x,y)-> vcat.(x,y), matrix_data) would do the same, but much slower
    Is, Js, Vs = collect_entries(matrix_data_chunks)

    return sparse(Is, Js, Vs, N, N)
end

doc"""
    α_normalize!(A, α = 0.5)
Normalize rows and columns of `A` in-place with the respective row-sum to the
α-th power; i.e., return $ a_{ij}:=a_{ij}/(q_i^{\alpha}q_j^{\alpha})$, where
$ q_k = \sum_{\ell} a_{k\ell}$. Default for `α` is 0.5.
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

# given a collection of Tuple{Vector, Vector, Vector },
# return a single Tuple{Vector, Vector, Vector}, concatenating
# entrywise.
function collect_entries(entry_lists::Vector{Tuple{Vector{Int}, Vector{Int}, Vector{T}}}) where T

    # check if each tuple has entries of matching lengths
    isdiag(tup) = (tup[1] == tup[2] == tup[3])
    isvalid(tup) = isdiag(length.(tup))
    @assert all(isvalid.(entry_lists))  "entry_lists contains a tuple with mismatching index counts"

    # get total number of entries
    counter = (lengths, tup) -> lengths .+ length.(tup)
    nI, nJ, nV = reduce(counter, (0,0,0), entry_lists)

    # allocate index and value lists
    I, J, V = zeros(Int, nI), zeros(Int, nJ), zeros(T, nV)

    # traverse the collection of Tuples while
    # filling the (linear) lists I,J and V

    k = 1 # current index in entry_lists
    offset = 1 # current index in entry_lists[k]
    for i = 1:nI
        tuple = entry_lists[k]
        I[i], J[i], V[i] = tuple[1][offset], tuple[2][offset], tuple[3][offset]
        offset += 1

        # move to the next entry of entry_lists
        if offset > length(tuple[1])
            offset = 1
            k += 1
        end
    end
    return I, J, V
end

"""
    meanmetric(F, av, metric)

For a set of trajectories ``x_i^t`` calculate an "averaged" distance matrix
``K ∈ R^{N×N}`` where ``k_{ij} = av(metric(x_i^1, x_j^1), … , k(x_i^T, x_j^T))``.
# Arguments
   * `F`: trajectory data with `size(F) = dim, T, N`
   * `av`: time-averaging function, see below for examples
   * `metric`: spatial distance function metric

## Examples for metrics
   * `Euclidean()`
   * `Haversine(r)`: geodesic distance on a sphere of radius `r` in the
        same units as `r`
   * `PeriodicEuclidean(L)`: Euclidean distance on a periodic domain, periods
        are contained in the vector `L`

## Examples for time averages
   * `av = mean`: arithmetic time-average, L¹ in time [Hadjighasem et al.]
   * `av = x->1/mean(inv,x)`: harmonic time-average [de Diego et al.]
   * `av = max`: sup/L\^infty in time [mentioned by Hadjighasem et al.]
   * `av = x->min(x)<=ε`: encounter adjacency [Rypina et al., Padberg-Gehle & Schneide]
   * `av = x->mean(abs2,x)`: Euclidean delay coordinate metric [cf. Froyland & Padberg-Gehle]
"""
function meanmetric(F::AbstractArray{T,3}, av, metric) where T <: Real
    dim, t, N = size(F)

    entries = div(N*(N+1),2)

    # linear storage of triangular matrix
    R = SharedArray{T,2}(N,N)

    # enumerate the entries linearily to distribute them
    # evenly over the workers
    dummy_value = av([evaluate(metric,F[:,1:1,1], F[:,1:1,1])])
    @everywhere dists = $(repeat([dummy_value],inner =t))

    @views @sync @parallel for n = 1:entries
       i, j = tri_indices(n)
       # fill_distances!(dists, F[:,:,i], F[:,:,j], metric, t)
       Distances.colwise!(dists, metric, F[:,:,i], F[:,:,j])
       R[i,j] = av(dists)
    end
    Symmetric(R,:L)
end

function fill_distances!(dists, xis, xjs, k, t)
    @views for l in 1:t
        dists[l] = k(xis[:,l], xjs[:,l])
    end
    return dists
end

function tri_indices(n::Int)
    i = floor(Int, 0.5*(1 + sqrt(8n-7)))
    j = n - div(i*(i-1),2)
    return i, j
end
