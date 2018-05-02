# (c) 2018 Alvaro de Diego, with minor contributions by Daniel Karrasch

NN = NearestNeighbors
Dists = Distances

"""
    sparse_time_coup_diff_op(sol, k, thresh; mapper, metric)

Return a list of sparse diffusion/Markov matrices ``P``.
`k` is the diffusion kernel, `metric` determines the notion of
distance w.r.t. which the kernel is computed, however, only for point
pairs where ``metric(x_i, x_j)≦ε``. `mapper` is either `pmap`
(for parallel computation) or `map`.
"""

function sparse_time_coup_diff_op( sols::AbstractArray{T, 3},
                                   kernel::Function,
                                   cutoff_distance:: T;
                                   mapper::Function = pmap,
                                   metric = Dists.Euclidean()) where T
    dim, q, N = size(sols)
    # diff_kernel = (x,y) -> kernel(evaluate(metric, x, y))

    P = mapper(1:q) do t
        tic()
        # Pₜ = sparseaffinitykernel((@views sols[:,t,:]),
        Pₜ = sparseaffinitykernel(sols[:,t,:],
                                 kernel,
                                 metric,
                                 cutoff_distance)
        α_normalize!(Pₜ, 1.0)
        wlap_normalize!(Pₜ)
        println("Timestep $t/$q done")
        toc()
        Pₜ
    end
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

    # the "collect" is necessary because BallTree currently cannot handle
    # abstract arrays.
    # balltree = NN.BallTree(collect(A), metric)
    balltree = NN.BallTree(A, metric)
    diff_kernel = (x,y) -> kernel(Dists.evaluate(metric, x, y))

    # Get a Vector of tuples (I,J,V), one for each column i.
    # We hardcode the type because the compiler can't infer it (function barrier
    # and list comprehension don't help) -> check every once in a while, if this
    # has changed. (24.04.18)
    matrix_data_chunks = Array{Tuple{Vector{Int}, Vector{Int}, Vector{T}}}(N)
    matrix_data_chunks[:] = @views map(1:N) do i
        Js = NN.inrange(balltree, A[:,i], ε)
        Vs = [diff_kernel(A[:,i], A[:,j]) for j in Js]
        Is = fill(i,length(Js))
        # return the resulting I,J,V for the sparse matrix
        Is, Js, Vs
    end

    # concatenate the lists of collected indices and values
    # reduce((x,y)-> vcat.(x,y), matrix_data) would do the same, but much slower
    Is, Js, Vs = collect_entries(matrix_data_chunks)

    return sparse(Is, Js, Vs, N, N)
end

"""
    α_normalize!(A, α = 0.5)
Normalize ``A`` in-place from left and right with the row-sum vector
``q.^α``. Default for ``α`` is 0.5.
"""

function α_normalize!(A::AbstractMatrix{T}, α::T = 0.5) where T<:Number
    qₑ = spdiagm( (1./squeeze(sum(A, 2),2).^α))
    A .= qₑ * A * qₑ
    return A
end

"""
    wlap_normalize!(A)
Normalize ``A`` in-place from left with the row-sum vector ``q``.
"""

function wlap_normalize!(A::AbstractMatrix)
    dᵅ = spdiagm(1./squeeze(sum(A, 2),2))
    A .= dᵅ * A
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
    I,J,V = zeros(Int, nI), zeros(Int, nJ), zeros(T, nV)

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

    return I,J,V
end
