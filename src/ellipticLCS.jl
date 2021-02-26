# (c) 2018-2020 Daniel Karrasch & Nathanael Schilling

const Ω = SMatrix{2,2}(0.0, 1.0, -1.0, 0.0)

"""
Container type for critical points of vector fields or singularities of line fields.

## Fields

* `coords::SVector{2}`: coordinates of the singularity
* `index::Rational`: index of the singularity
"""
struct Singularity{T<:Real}
    coords::SVector{2,T}
    index::Rational{Int}
end
function Singularity(coords::SVector{2}, index::Real)
    return Singularity(coords, convert(Rational, index))
end
function Singularity(coords::NTuple{2,Real}, index::Real)
    return Singularity(SVector{2}(coords), index)
end

"""
    getcoords(singularities)

Extracts the coordinates of `singularities`, a vector of `Singularity`s. Returns
a vector of `SVector`s.
"""
function getcoords(singularities::AbstractArray{<:Singularity})
    return [s.coords for s in singularities]
end

"""
    getindices(singularities)

Extracts the indices of `singularities`, a vector of `Singularity`s.
"""
function getindices(singularities::AbstractArray{<:Singularity})
    return [s.index for s in singularities]
end

abstract type MergeHeuristic end

"""
    Combine(dist) <: MergeHeuristic

Merge singularities that are within (Euclidean) distance `dist` from each other.
Upon merging, a virtual singularity is created with an averaged location.
"""
struct Combine <: MergeHeuristic
    distance::Float64
    function Combine(dist::Real)
        dist ≥ 0 || error("combine distance must be positive, got $dist")
        return new(convert(Float64, dist))
    end
end

"""
    Combine20 <: MergeHeuristic

Merge isolated singularity pairs that are mutually nearest neighbors.
"""
struct Combine20 <: MergeHeuristic end
"""
    Combine31 <: MergeHeuristic

Merge a trisector with three wedge singularities if those are its three nearest neighbors.
"""
struct Combine31 <: MergeHeuristic end
struct Combine20Aggressive <: MergeHeuristic end

"""
This is a container for coherent vortex boundaries. An object `barrier` of type
`EllipticBarrier` can be easily plotted by `plot(barrier.curve)`, or
`plot!([figure, ]barrier.curve)` if it is to be overlaid over an existing plot.

## Fields

* `curve`: a vector of tuples, contains the coordinates of coherent vortex boundary
  points;
* `core`: location of the vortex core;
* `p`: contains the parameter value of the direction field ``\\eta_{\\lambda}^{\\pm}``,
  for which the `curve` is a closed orbit;
* `s`: a `Bool` value, which encodes the sign in the formula of the direction
  field ``\\eta_{\\lambda}^{\\pm}`` via the formula ``(-1)^s``.
"""
struct EllipticBarrier{T<:Real}
    curve::Vector{SVector{2,T}}
    core::SVector{2,T}
    p::Float64
    s::Bool
end

"""
This is a container for an elliptic vortex, as represented by the vortex's `center`
and a list `barriers` of all computed [`EllipticBarrier`](@ref)s.

## Fields

* `center`: location of the vortex center;
* `barriers`: vector of `EllipticBarrier`s.
"""
struct EllipticVortex{T<:Real}
    center::SVector{2,T}
    barriers::Vector{EllipticBarrier{T}}
end

abstract type Parameters end

function Base.show(io::IO, p::T) where {T<:Parameters}
    keys = fieldnames(T)
    values = getfield.(Ref(p), keys)
    return show(io, (; zip(keys, values)...))
end

"""
Container for parameters used in elliptic LCS computations.

## Fields

* `boxradius`: "radius" of localization square for closed orbit detection
* `indexradius=1e-1boxradius`: radius for singularity type detection
* `merge_heuristics`: a list of [`MergeHeuristic`](@ref)s for combining singularities
* `n_seeds=100`: number of seed points on the Poincaré section
* `pmin=0.7`: lower bound on the parameter in the ``\\eta``-field
* `pmax=2.0`: upper bound on the parameter in the ``\\eta``-field
* `rdist=1e-4boxradius`: required return distances for closed orbits
* `tolerance_ode=1e-8boxradius`: absolute and relative tolerance in orbit integration
* `maxiters_ode::Int=2000`: maximum number of integration steps
* `max_orbit_length=8boxradius`: maximum length of orbit length
* `maxiters_bisection::Int=20`: maximum steps in bisection procedure
* `only_enclosing::Bool=true`: whether the orbit must enclose the starting point of the Poincaré section
* `only_smooth::Bool=true`: whether or not to reject orbits with "corners".
* `only_uniform::Bool=true`: whether or not to reject orbits that are not uniform

## Example

```repl
p = LCSParameters(2.5)
LCSParameters(2.5, 0.25, true, 100, 0.7, 2.0, 0.00025, 2.5e-8, 1000, 20.0, 30)
```
"""
struct LCSParameters{T<:Tuple{Vararg{MergeHeuristic}}} <: Parameters
    boxradius::Float64
    indexradius::Float64
    merge_heuristics::T
    n_seeds::Int
    pmin::Float64
    pmax::Float64
    rdist::Float64
    tolerance_ode::Float64
    maxiters_ode::Int
    max_orbit_length::Float64
    maxiters_bisection::Int
    only_enclosing::Bool
    only_smooth::Bool
    only_uniform::Bool

    function LCSParameters(
        boxradius::Real,
        indexradius::Real = 1e-1boxradius,
        merge_heuristics = (Combine20(),),
        n_seeds::Int = 100,
        pmin::Real = 0.7,
        pmax::Real = 2.0,
        rdist::Real = 1e-4boxradius,
        tolerance_ode::Real = 1e-8boxradius,
        maxiters_ode::Int = 1000,
        max_orbit_length::Real = 8boxradius,
        maxiters_bisection::Int = 30,
        only_enclosing::Bool = true,
        only_smooth::Bool = true,
        only_uniform::Bool = true,
    )
        mh = tuple(merge_heuristics...)
        return new{typeof(mh)}(
            float(boxradius),
            float(indexradius),
            mh,
            n_seeds,
            float(pmin),
            float(pmax),
            float(rdist),
            float(tolerance_ode),
            maxiters_ode,
            float(max_orbit_length),
            maxiters_bisection,
            only_enclosing,
            only_smooth,
            only_uniform,
        )
    end
end

function LCSParameters(;
    boxradius::Real = 1.0,
    indexradius::Real = 1e-1boxradius,
    merge_heuristics = (Combine20(),),
    n_seeds::Int = 100,
    pmin::Real = 0.7,
    pmax::Real = 2.0,
    rdist::Real = 1e-4boxradius,
    tolerance_ode::Real = 1e-8boxradius,
    maxiters_ode::Int = 1000,
    max_orbit_length::Real = 8boxradius,
    maxiters_bisection::Int = 30,
    only_enclosing::Bool = true,
    only_smooth::Bool = true,
    only_uniform::Bool = true,
)

    return LCSParameters(
        float(boxradius),
        float(indexradius),
        (merge_heuristics...,),
        n_seeds,
        float(pmin),
        float(pmax),
        float(rdist),
        float(tolerance_ode),
        maxiters_ode,
        float(max_orbit_length),
        maxiters_bisection,
        only_enclosing,
        only_smooth,
        only_uniform,
    )
end

struct LCScache{Ts<:Real,Tv<:SVector{2,<:Real}}
    λ₁::AxisArray{Ts,2}
    λ₂::AxisArray{Ts,2}
    Δ::AxisArray{Ts,2}
    α::AxisArray{Ts,2}
    β::AxisArray{Ts,2}
    ξ₁::AxisArray{Tv,2}
    ξ₂::AxisArray{Tv,2}
    η::AxisArray{Tv,2}
end

"""
    struct FlowGrowParams

Container for parameters used in point-inserting flow computations;
see [`flowgrow`](@ref).

## Fields

* `maxcurv=0.3`: maximal bound on the absolute value of the discrete curvature;
* `mindist=0.1`: least acceptable distance between two consecutive points;
* `maxdist=1.0`: maximal acceptable distance between two consecutive points.
"""
struct FlowGrowParams <: Parameters
    maxcurv::Float64
    mindist::Float64
    maxdist::Float64
    function FlowGrowParams(maxcurv, mindist, maxdist)
        all(isposdef, (maxcurv, mindist, maxdist))
        return new(Float64(maxcurv), Float64(mindist), Float64(maxdist))
    end
end

function FlowGrowParams(; maxcurv=0.3, mindist=0.1, maxdist=1.0)
    return FlowGrowParams(maxcurv, mindist, maxdist)
end

abstract type SignedAngleDist end
struct S1Dist <: SignedAngleDist end

"""
    S1Dist()(α, β)

Computes the signed length of the angle of the shortest circle segment going
from angle `β` to angle `α`, as computed on the full circle.

## Examples

```jldoctest
julia> dist = S1Dist();

julia> dist(π/2, 0)
1.5707963267948966

julia> dist(0, π/2)
-1.5707963267948966
```
"""
(::S1Dist)(x, y) = rem2pi(x - y, RoundNearest)
@deprecate s1dist(x, y) S1Dist()(x, y)

struct P1Dist <: SignedAngleDist end
"""
    P1Dist()(α, β)

Computes the signed length of the angle of the shortest circle segment going
from angle `β` to angle `α [± π]`, as computed on the half circle.

## Examples

```jldoctest
julia> dist = P1Dist();

julia> dist(π, 0)
0.0
```
"""
(::P1Dist)(x, y) = rem(float(x - y), float(π), RoundNearest)
@deprecate p1dist(x, y) P1Dist()(x, y)

##################### singularity/critical point detection #####################
"""
    compute_singularities(v, xspan, yspan, dist=S1Dist()) -> Vector{Singularity}    
    compute_singularities(v::AxisArray, dist=S1Dist()) -> Vector{Singularity}

Computes critical points and singularities of vector and line fields `v`,
respectively. The argument `dist` is a signed distance function for angles.
Choose `s1dist` (default) for vector fields, and `p1dist` for line fields.
"""
function compute_singularities(v::AbstractMatrix{<:SVector{2}},
    xspan::AbstractRange,
    yspan::AbstractRange,
    dist::SignedAngleDist = S1Dist(),
)
    axes(v) == (eachindex(xspan), eachindex(yspan)) || error("axes don't match")
    @inbounds α = map(u -> atan(u[2], u[1]), v)
    xstephalf = step(xspan) / 2
    ystephalf = step(yspan) / 2
    T = typeof(xstephalf)
    singularities = Singularity{T}[]
    # go counter-clockwise around each grid cell and add angles
    # for cells with non-vanishing index, collect cell midpoints
    @inbounds for (j, y) in enumerate(yspan[1:end-1]), (i, x) in enumerate(xspan[1:end-1])
        temp = dist(α[i+1, j], α[i, j]) # to the right
        temp += dist(α[i+1, j+1], α[i+1, j]) # to the top
        temp += dist(α[i, j+1], α[i+1, j+1]) # to the left
        temp += dist(α[i, j], α[i, j+1]) # to the bottom
        index = round(Int, temp / π)
        if !iszero(index)
            push!(singularities, Singularity((x + xstephalf, y + ystephalf), index // 2))
        end
    end
    return singularities
end
function compute_singularities(v::AxisArray{<:SVector{2},2}, dist::SignedAngleDist = S1Dist())
    xspan, yspan = v.axes
    return compute_singularities(v.data, xspan.val, yspan.val, dist)    
end

"""
    Combine(dist)(sing_coordinates) -> Vector{Singularity}

This function does the equivalent of: build a graph where singularities are vertices, and
two vertices share an edge iff the coordinates of the corresponding vertices (given by
`sing_coordinates`) have a distance leq `dist`. Find all connected components of
this graph, and return a list of their mean coordinate and sum of `sing_indices`.
"""
function (c::Combine)(singularities::Vector{<:Singularity})
    # Do a breath-first search of all singularities that are "connected" in the
    # sense that there is a path of singularities with each segment less than
    # `c.distance`
    # Average the coordinates, add the indices

    N = length(singularities)

    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    #Which singularities we've already dealt with
    sing_seen = falses(N)

    #Result will go here
    combined_singularities = eltype(singularities)[]

    #Iterate over all singularities
    for i in 1:N
        if sing_seen[i]
            continue
        end
        sing_seen[i] = true

        current_index = 0
        current_coords = SVector{2}((0.0, 0.0))
        num_combined = 0

        #Breadth-first-search
        stack = Int[]
        push!(stack, i)
        while !isempty(stack)
            current_singularity = pop!(stack)
            sing_seen[i] = true
            closeby_sings = NN.inrange(
                sing_tree,
                singularities[current_singularity].coords,
                c.distance,
            )
            for neighbour_index ∈ closeby_sings
                if !(sing_seen[neighbour_index])
                    sing_seen[neighbour_index] = true
                    push!(stack, neighbour_index)
                end
            end
            #Average coordinates & add indices
            current_index += singularities[current_singularity].index
            current_coords =
                (num_combined * current_coords + singularities[current_singularity].coords) / (num_combined + 1)
            num_combined += 1
        end
        if current_index != 0
            push!(
                combined_singularities,
                Singularity(current_coords, current_index),
            )
        end
    end

    return combined_singularities
end

"""
    Combine20()(singularities)

Determines singularities which are mutually closest neighbors and combines them as one,
while adding their indices.
"""
function (c::Combine20)(singularities::Vector{<:Singularity})
    N = length(singularities)
    N == 1 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    sing_seen = falses(N)

    new_singularities = eltype(singularities)[] # sing_out
    sing_out_weight = Int[]

    for i in 1:N
        if sing_seen[i]
            continue
        end
        sing_seen[i] = true

        if singularities[i].index != 1 // 2
            push!(new_singularities, singularities[i])
            continue
        end
        #We have an index +1/2 singularity
        idxs, dists = NN.knn(sing_tree, singularities[i].coords, 2, true)
        nn_idx = idxs[2]

        #We've already dealt with the nearest neighbor (but didn't find
        #this one as nearest neighbor), or it isn't a wedge
        if sing_seen[nn_idx] || singularities[nn_idx].index != 1 // 2
            push!(new_singularities, singularities[i])
            continue
        end

        #See if the nearest neighbor of the nearest neighbor is i
        idxs2, dists2 = NN.knn(sing_tree, singularities[nn_idx].coords, 2, true)
        if idxs2[2] != i
            push!(new_singularities, singularities[i])
            continue
        end

        sing_seen[nn_idx] = true
        push!(
            new_singularities,
            Singularity(
                0.5 * (singularities[i].coords + singularities[nn_idx].coords),
                2 // 2,
            ),
        )
    end
    return new_singularities
end
@deprecate combine_20(singularities) Combine20()(singularities)

"""
    Combine31()(singularities)

Takes the list of singularities in `singularities` and combines them
so that any -1/2 singularity whose three nearest neighbors are 1/2 singularities
becomes an elliptic region, provided that the -1/2 singularity
is in the triangle spanned by the wedges. This configuration
is common for OECS, applying to material barriers on a large
turbulent example yielded only about an additional 1% material barriers.
"""
function (c::Combine31)(singularities::Vector{<:Singularity})
    N = length(singularities)
    N <= 2 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    sing_seen = falses(N)

    new_singularities = eltype(singularities)[] # sing_out
    sing_out_weight = Int[]

    #Iterate over all trisector-type singularities
    for i in 1:N
        if singularities[i].index != -1 // 2
            continue
        end

        idxs, dists = NN.knn(sing_tree, singularities[i].coords, 4, true)
        correct_configuration = true
        for j in 1:3
            if singularities[idxs[j+1]].index != 1 // 2
                correct_configuration = false
            end
        end
        if !in_triangle(
            singularities[i].coords,
            singularities[idxs[2]].coords,
            singularities[idxs[3]].coords,
            singularities[idxs[4]].coords,
        )
            correct_configuration = false
        end
        if !correct_configuration
            continue
        end

        for j in 1:3
            sing_seen[idxs[j+1]] = true
        end
        sing_seen[i] = true
        push!(new_singularities, Singularity(singularities[i].coords, 1 // 1))
    end

    #Add whatever singularities are left over
    for i in 1:N
        if !sing_seen[i]
            push!(new_singularities, singularities[i])
        end
    end
    return new_singularities
end
@deprecate combine_31(singularities) Combine31()(singularities)

"""
    Combine20Aggressive()(singularities)

A heuristic for combining singularities which is likely to have a lot of false positives.
"""
function (c::Combine20Aggressive)(singularities::Vector{<:Singularity})
    N = length(singularities)
    N == 1 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    combined_with = [Int[] for i in 1:N]
    sing_seen = falses(N)

    new_singularities = eltype(singularities)[] # sing_out
    sing_out_weight = Int[]

    for i in 1:N
        cur_sing = singularities[i]

        if cur_sing.index != 1 // 2
            continue
        end

        #We have an index +1/2 singularity
        idxs, dists = NN.knn(sing_tree, cur_sing.coords, 2, true)
        nn_idx = idxs[2]
        nn_sing = singularities[nn_idx]

        if nn_idx ∈ combined_with[i]
            continue
        end

        #See if its nearest neighbor is a wedge
        if nn_sing.index != 1 // 2
            continue
        end

        midpoint = 0.5 .* (cur_sing.coords .+ nn_sing.coords)
        width = norm(cur_sing.coords .- midpoint)
        idxs2 = NN.inrange(sing_tree, midpoint, width)

        function in_rect(p, p1, p2)
            xmax = max(p1[1], p2[1])
            ymax = max(p1[2], p2[2])
            xmin = min(p1[1], p2[1])
            ymin = min(p1[2], p2[2])

            return (xmin <= p[1] <= xmax) && (ymin <= p[2] <= ymax)
        end
        found_in_rect = false
        for j in idxs2
            if j == i || j == idxs[2]
                continue
            end
            if in_rect(singularities[j].coords, cur_sing.coords, nn_sing.coords)
                found_in_rect = true
                found_in_rect = false
            end
        end

        found_in_rect && continue

        sing_seen[nn_idx] = true
        sing_seen[i] = true
        push!(
            new_singularities,
            Singularity(
                0.5 * (singularities[i].coords + singularities[nn_idx].coords),
                2 // 2,
            ),
        )
        push!(combined_with[nn_idx], i)
    end

    for i in 1:N
        if !sing_seen[i]
            push!(new_singularities, singularities[i])
        end
    end
    return new_singularities
end
@deprecate combine_20_aggressive(singularities) Combine20Aggressive()(singularities)

"""
    critical_point_detection(vs, xspan, yspan, combine_distance, dist=S1Dist();
        merge_heuristics=(combine_20,)) -> Vector{Singularity}
    critical_point_detection(vs::AxisArray, combine_distance, dist=S1Dist();
        merge_heuristics=(combine_20,)) -> Vector{Singularity}

Computes critical points of a vector/line field `vs`, potentially given as an `AxisArray`.
Critical points with distance less or equal to `combine_distance` are
combined by averaging the coordinates and adding the respective indices. The
argument `dist` is a signed distance function for angles: choose [`S1Dist()`](@ref)
for vector fields, and [`P1Dist()`](@ref) for line fields; cf. [`compute_singularities`](@ref).
[`MergeHeuristic`](@ref)s listed in `merge_heuristics`, cf. [`LCSParameters`](@ref),
are applied to combine singularities.

Returns a vector of [`Singularity`](@ref)s.
"""
function critical_point_detection(
    vs::AbstractMatrix{<:SVector{2}},
    xspan::AbstractRange,
    yspan::AbstractRange,
    combine_distance::Real,
    dist::SignedAngleDist = S1Dist();
    merge_heuristics = (Combine20(),),
)
    singularities = compute_singularities(vs, xspan, yspan, dist)
    new_singularities = Combine(combine_distance)(singularities)
    for f in merge_heuristics
        new_singularities = f(new_singularities)
    end
    return new_singularities
end
@inline function critical_point_detection(
    vs::AxisArray{<:SVector{2},2},
    combine_distance::Real,
    dist::SignedAngleDist = S1Dist();
    merge_heuristics = (Combine20(),),
)
    xspan, yspan = vs.axes
    return critical_point_detection(vs.data, xspan.val, yspan.val, combine_distance, dist;
        merge_heuristics=merge_heuristics)
end

"""
    singularity_detection(T, xspan, yspan, combine_distance; merge_heuristics=(Combine20(),)) -> Vector{Singularity}
    singularity_detection(T::AxisArray, combine_distance; merge_heuristics=(Combine20(),)) -> Vector{Singularity}

Calculate line-field singularities of the first eigenvector of `T` by taking a discrete
differential-geometric approach. Singularities are calculated on each cell. Singularities
with distance less or equal to `combine_distance` are combined by averaging the coordinates
and adding the respective indices. The heuristics listed in `merge_heuristics` are used to
merge singularities, cf. [`LCSParameters`](@ref).

Return a vector of [`Singularity`](@ref)s.
"""
function singularity_detection(
    T::AbstractMatrix{<:SymmetricTensor{2,2}},
    xspan::AbstractRange,
    yspan::AbstractRange,
    combine_distance::Real;
    merge_heuristics = (Combine20(),),
)
    ξ = map(T) do t
        @inbounds v = eigvecs(t)[:, 1]
        @inbounds SVector{2}((v[1], v[2]))
    end
    critical_point_detection(ξ, xspan, yspan, combine_distance, P1Dist(); merge_heuristics=merge_heuristics)
end
@inline function singularity_detection(
    T::AxisArray{<:SymmetricTensor{2,2},2},
    combine_distance::Real;
    merge_heuristics = (Combine20(),),
)
    xspan, yspan = vs.axes
    return singularity_detection(T, xspan.val, yspan.val, combine_distance;
        merge_heuristics=merge_heuristics)
end

######################## closed orbit computations #############################
"""
    compute_returning_orbit(vf, seed::SVector{2}, save::Bool=false, maxiters=2000, tolerance=1e-8, max_orbit_length=20.0)

Computes returning orbits under the velocity field `vf`, originating from `seed`.
The optional argument `save` controls whether intermediate locations of the
returning orbit should be saved.
Returns a tuple of `orbit` and `statuscode` (`0` for success, `1` for `maxiters`
reached, `2` for out of bounds error, 3 for other error).
"""
function compute_returning_orbit(
    vf::ODE.ODEFunction,
    seed::T,
    save::Bool = false,
    maxiters::Int = 2000,
    tolerance::Float64 = 1e-8,
    max_orbit_length::Float64 = 20.0,
)::Tuple{Vector{T},Symbol} where {T<:SVector{2}} # TODO: type assertion necessary on Julia v1.0
    dir = vf(seed, nothing, 0.0)[2] < 0 ? -1 : 1 # Whether orbits initially go upwards
    condition(u, t, integrator) = dir * (seed[2] - u[2])
    affect!(integrator) = ODE.terminate!(integrator)
    cb = ODE.ContinuousCallback(condition, nothing, affect!)
    prob = ODE.ODEProblem(vf, seed, (0.0, max_orbit_length))
    try
        sol = ODE.solve(
            prob,
            ODE.Tsit5(),
            maxiters = maxiters,
            dense = false,
            save_everystep = save,
            reltol = tolerance,
            abstol = tolerance,
            callback = cb,
            verbose = false,
        )
        return (sol.u, sol.retcode)
    catch e
        if e isa BoundsError
            return ([SVector{2,eltype(T)}(NaN, NaN)], :BoundsError)
        end
        rethrow(e)
    end
end

function Poincaré_return_distance(
    vf,
    seed::SVector{2,T},
    save::Bool = false;
    tolerance_ode::Float64 = 1e-8,
    maxiters_ode::Int = 2000,
    max_orbit_length::Float64 = 20.0,
) where {T<:Real}
    sol, retcode = compute_returning_orbit(
        vf,
        seed,
        save,
        maxiters_ode,
        tolerance_ode,
        max_orbit_length,
    )
    # check if result due to callback
    if retcode === :Terminated
        return sol[end][1] - seed[1]
    else
        return T(NaN)
    end
end

function orient(T::AxisArray{<:SymmetricTensor{2,2},2}, center::SVector{2})
    xspan, yspan = T.axes
    λ₁, λ₂, ξ₁, ξ₂, _, _ = tensor_invariants(T)
    Δλ = AxisArray(λ₂ .- λ₁, T.axes)
    star = AxisArray(
        [SVector{2}(x, y) - center for x in xspan.val, y in yspan.val],
        T.axes,
    )
    c1 = AxisArray(sign.(dot.((Ω,) .* star, ξ₁)), T.axes)
    ξ₁ .*= c1
    c2 = AxisArray(sign.(dot.(star, ξ₂)), T.axes)
    ξ₂ .*= c2
    return LCScache(λ₁, λ₂, Δλ, c1, c2, ξ₁, ξ₂, star)
end
function orient(T::AbstractMatrix{<:SymmetricTensor{2,2}}, xspan, yspan, center::SVector{2})
    return orient(AxisArray(T, xspan, yspan), center)
end

"""
    compute_closed_orbits(ps, ηfield, cache; kwargs...)

Compute the (outermost) closed orbit for a given Poincaré section `ps`, a vector field
constructor `ηfield`, and an LCScache `cache`.

## Keyword arguments

* `rev=true`: determines whether closed orbits are sought from the outside inwards (`true`)
  or from the inside outwards (`false`);
* `pmin=0.7`, `pmax=1.5`: correspond to the range of shift parameters in which closed orbits are sought;
* `rdist=1e-4` sets the required return distance for an orbit to be considered as closed

rev=true, pmin=0.7, pmax=1.5, rdist=1e-4, tolerance_ode=1e-8, maxiters_ode=2000, maxiters_bisection=20


. . The parameter `maxiters_ode` gives the maximum number
of steps taken by the ODE solver when computing the closed orbit, the ode solver uses tolerance
given by `tolerance_ode`. The parameter `maxiters_bisection` gives the maximum number of iterations
used by the bisection algorithm to find closed orbits.
"""
function compute_closed_orbits(
    ps::AbstractVector{SVector{2,S1}},
    ηfield,
    cache;
    rev::Bool = true,
    p::LCSParameters = LCSParameters()
) where {S1<:Real}
    if cache isa LCScache # tensor-based LCS computation
        l1itp = ITP.LinearInterpolation(cache.λ₁)
        l2itp = ITP.LinearInterpolation(cache.λ₂)
    else # vector-field-based LCS computation
        nitp = ITP.LinearInterpolation(map(v -> norm(v)^2, cache))
    end
    # define local helper functions for the η⁺/η⁻ closed orbit detection
    prd(λ::Float64, σ::Bool, seed::SVector{2}, cache) =
        let tol = p.tolerance_ode,
            maxode = p.maxiters_ode,
            maxorbit = p.max_orbit_length

            Poincaré_return_distance(
                ηfield(λ, σ, cache),
                seed;
                tolerance_ode = tol,
                maxiters_ode = maxode,
                max_orbit_length = maxorbit,
            )
        end

    # VERSION 2: 3D-interpolant
    # η(λ::Float64, signum::Bool) = begin
    #     α = min.(sqrt.(max.(λ₂.vals .- λ, 0) ./ Δλ.vals), 1)
    #     β = min.(sqrt.(max.(λ .- λ₁.vals, 0) ./ Δλ.vals), 1)
    #     return α .* ξ₁.vecs .+ (-1)^signum .* β .* ξ₂.vecs
    # end
    # λrange = range(pmin, stop=pmax, length=20)
    # ηdata = cat([η(λ, false) for λ in λrange]..., dims=3)
    # ηitp = ITP.scale(ITP.interpolate(ηdata,
    #         (ITP.BSpline(ITP.Cubic(ITP.Natural(ITP.OnGrid()))),
    #          ITP.BSpline(ITP.Cubic(ITP.Natural(ITP.OnGrid()))),
    #          ITP.BSpline(ITP.Linear()))),
    #                 xspan, yspan, λrange)
    # ηfield(λ::Float64) = ODE.ODEFunction{false}((u, p, t) -> ηitp(u[1], u[2], λ))
    # prd(λ::Float64, seed::SVector{2,S}) = Poincaré_return_distance(ηfield(λ), seed)
    # END OF VERSION 2

    # go along the Poincaré section and solve for λ⁰ such that orbits close up
    vortices = EllipticBarrier{S1}[]
    idxs = rev ? (length(ps):-1:2) : (2:length(ps))
    for i in idxs
        if cache isa LCScache && p.only_uniform
            pmin_local = max(p.pmin, l1itp(ps[i][1], ps[i][2]))
            pmax_local = min(p.pmax, l2itp(ps[i][1], ps[i][2]))
            margin_step = (pmax_local - pmin_local) / 20
            if !(margin_step > 0)
                continue
            end
        else #TODO: can something like the above be done for the constrained LCS setting too?
            pmin_local = p.pmin
            pmax_local = p.pmax
            margin_step = (pmax_local - pmin_local) / 20
        end

        σ = false
        bisection_retcode, λ⁰ = bisection(
            let σ = σ, ps = ps[i], cache = cache
                λ -> prd(λ, σ, ps, cache)
            end,
            pmin_local,
            pmax_local,
            p.rdist,
            p.maxiters_bisection,
            margin_step,
        )
        if bisection_retcode != zero_found
            σ = true
            bisection_retcode, λ⁰ = bisection(
                let σ = σ, ps = ps[i], cache = cache
                    λ -> prd(λ, σ, ps, cache)
                end,
                pmin_local,
                pmax_local,
                p.rdist,
                p.maxiters_bisection,
                margin_step,
            )
        end
        if bisection_retcode == zero_found
            orbit, retcode = compute_returning_orbit(
                ηfield(λ⁰, σ, cache),
                ps[i],
                true,
                p.maxiters_ode,
                p.tolerance_ode,
                p.max_orbit_length,
            )
            if retcode === :Terminated
                closed = norm(orbit[1] - orbit[end]) <= p.rdist
                if cache isa LCScache
                    in_well_defined_squares =
                        !(p.only_smooth) || in_defined_squares(orbit, cache)
                    uniform =
                        !(p.only_uniform) || in_uniform_squares(orbit, λ⁰, cache)
                else
                    predicate = let λ = λ⁰
                        qs -> nitp(qs[1], qs[2]) >= λ^2
                    end
                    in_well_defined_squares = true
                    uniform = !(p.only_uniform) || all(predicate, orbit)
                end

                contains_singularity =
                    !(p.only_enclosing) || contains_point(orbit, ps[1])

                if (
                    closed &&
                    uniform && in_well_defined_squares && contains_singularity
                )
                    push!(vortices, EllipticBarrier(orbit, ps[1], λ⁰, σ))
                    rev && break
                end
            end
        end
    end
    return vortices
end

"""
    ellipticLCS(T::AbstractMatrix, xspan, yspan, p; kwargs...)
    ellipticLCS(T::AxisArray, p; kwargs...)

Computes elliptic LCSs as null-geodesics of the Lorentzian metric tensor
field given by shifted versions of `T` on the 2D computational grid spanned
by `xspan` and `yspan`. `p` is a [`LCSParameters`](@ref)-type container of
computational parameters. Returns a list of `EllipticBarrier`-type objects.

## Keyword arguments

* `outermost=true`: only the outermost barriers, i.e., the vortex
  boundaries are returned, otherwise all detected transport barrieres;
* `verbose=true`: show intermediate computational information;
* `unique_vortices=true`: filter out vortices enclosed by other vortices;
* `suggested_centers=[]`: suggest vortex centers (of type [`Singularity`](@ref));
* `debug=false`: whether to use the debug mode, which avoids parallelization
  for more precise error messages;
* `singularity_predicate = nothing`: provide an optional callback to reject certain
  singularity candidates.
"""
function ellipticLCS(
    T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
    xspan::AbstractRange,
    yspan::AbstractRange,
    p::LCSParameters = LCSParameters();
    verbose::Bool = true,
    unique_vortices = true,
    singularity_predicate = nothing,
    suggested_centers = Singularity{S}[],
    kwargs...,
) where {S<:Real}
    # detect centers of elliptic (in the index sense) regions
    singularities = singularity_detection(T, xspan, yspan, p.indexradius;
        merge_heuristics = p.merge_heuristics,
    )
    append!(singularities, suggested_centers)
    if singularity_predicate !== nothing
        singularities = filter(singularity_predicate, singularities)
    end
    verbose && @info "Found $(length(singularities)) singularities..."
    vortexcenters = filter(s -> s.index == 1, singularities)
    verbose && @info "Defined $(length(vortexcenters)) Poincaré sections..."

    vortices = getvortices(T, xspan, yspan, vortexcenters, p; verbose = verbose, kwargs...)
    if unique_vortices
        vortices = makeVortexListUnique(vortices, p.indexradius)
    end
    return vortices, singularities
end
@inline function ellipticLCS(
    T::AxisArray{<:SymmetricTensor{2,2,S},2},
    p::LCSParameters = LCSParameters();
    kwargs...,
) where {S<:Real}
    xspan, yspan = T.axes
    return ellipticLCS(T.data, xspan.val, yspan.val, p; kwargs...)
end

function debugAt(
    T::AxisArray{<:SymmetricTensor{2,2,S},2},
    orientaround,
    startwhere = (orientaround .+ (0.0, 0.1));
    p::LCSParameters = LCSParameters(),
) where {S}
    cache = orient(T[:, :], SVector{2}(orientaround[1], orientaround[2]))
    l1itp = ITP.LinearInterpolation(cache.λ₁)
    l2itp = ITP.LinearInterpolation(cache.λ₂)
    result = []
    for σ ∈ [true, false]
        for λ ∈ range(p.pmin, stop = p.pmax, length = 50)
            sol, _ = compute_returning_orbit(
                ηfield(λ, σ, cache),
                SVector{2,Float64}(startwhere[1], startwhere[2]),
                true,
                p.maxiters_ode,
                p.tolerance_ode,
                p.max_orbit_length,
            )
            push!(result, sol)
        end
    end

    prd(λ::Float64, σ::Bool, seed, cache) =
        let tol = p.tolerance_ode,
            maxode = p.maxiters_ode,
            maxorbit = p.max_orbit_length

            Poincaré_return_distance(
                ηfield(λ, σ, cache),
                seed;
                tolerance_ode = tol,
                maxiters_ode = maxode,
                max_orbit_length = maxorbit,
            )
        end
    result2 = []
    for σ in (true, false)
        lamrange = range(p.pmin, stop = p.pmax, length = 30)
        push!(
            result2,
            (
                lamrange,
                map(
                    x -> prd(
                        x,
                        σ,
                        SVector{2,Float64}(startwhere[1], startwhere[2]),
                        cache,
                    ),
                    lamrange,
                ),
            ),
        )
    end

    pmin_local = max(p.pmin, l1itp(startwhere[1], startwhere[2]))
    pmax_local = min(p.pmax, l2itp(startwhere[1], startwhere[2]))
    println("pmin_local is $pmin_local and pmax_local is $pmax_local")
    margin_step = (pmax_local - pmin_local) / 20

    result3 = (
        bisection(
            λ -> prd(
                λ,
                true,
                SVector{2,Float64}(startwhere[1], startwhere[2]),
                cache,
            ),
            pmin_local,
            pmax_local,
            p.rdist,
            p.maxiters_bisection,
            margin_step,
        ),
        bisection(
            λ -> prd(
                λ,
                false,
                SVector{2,Float64}(startwhere[1], startwhere[2]),
                cache,
            ),
            pmin_local,
            pmax_local,
            p.rdist,
            p.maxiters_bisection,
            margin_step,
        ),
    )
    return result, result2, result3
end

# vector field constructor function
function ηfield(λ::Float64, σ::Bool, c::LCScache)
    @. c.α = min(sqrt(max(c.λ₂ - λ, eps()) / c.Δ), 1.0)
    @. c.β = min(sqrt(max(λ - c.λ₁, eps()) / c.Δ), 1.0)
    @. c.η = c.α * c.ξ₁ + ((-1)^σ) * c.β * c.ξ₂

    itp = ITP.LinearInterpolation(c.η)

    function unit_length_itp(u, p, t)
        result = itp(u[1], u[2])
        normresult = sqrt(result[1]^2 + result[2]^2)
        return normresult == 0 ? result : result / normresult
    end
    return ODE.ODEFunction{false}(unit_length_itp)
end

"""
    getvortices(T, centers, p = LCSParameters(); outermost=true, verbose=true, debug=false)

Compute [elliptic vortices](@ref EllipticVortex) from the symmetric tensor field
`T` with Poincaré sections placed at locations listed in `centers`. The argument
`p` is of type [`LCSParameters`](@ref) and contains parameters used in the
vortex detection.

## Keyword arguments
* `outermost=true`: whether closed orbits should be computed from the outside
  inwards until the first closed orbit is found; otherwise, closed orbits are
  computed from the center outward.
* `verbose=true`: whether feedback on the progress should be given.
* `debug=false`: whether parallel computation should be used. Set to `true` to
  turn off parallel computation and to obtain more useful error messages.
"""
function getvortices(
    T::AxisArray{SymmetricTensor{2,2,S,3},2},
    vortexcenters::AbstractVector{<:Singularity},
    p::LCSParameters = LCSParameters();
    outermost::Bool = true,
    verbose::Bool = true,
    debug::Bool = false,
) where {S<:Real}
    xspan = T.axes[1]
    xmax = xspan[end]

    #This is where results go
    vortices = EllipticVortex{S}[]

    #Type of restricted field is quite complex, therefore make a variable for it here
    Ttype = typeof(T)

    # We make two remote channels. The master process pushes to jobs_rc in order
    # (vx, vy, vr, p, outermost, T_local):
    #     * vx::S,vy::S (coordinates of vortex center)
    #     * vr::S (length of Poincaré section)
    #     * p::LCSParameters
    #     * outermost::Bool (whether to only search for outermost barriers)
    #     * T_local (A local copy of the tensor field)
    # Worker processes/tasks take elements from jobs_rc, calculate barriers, and put
    # the results in results_rc

    #How many vortex centers we have
    num_jobs = length(vortexcenters)

    jobs_queue_length = debug ? num_jobs : nprocs()
    results_queue_length = debug ? num_jobs : 2 * nprocs()

    jobs_rc = RemoteChannel(
        () -> Channel{Tuple{S,S,S,LCSParameters,Bool,Ttype}}(jobs_queue_length),
    )
    results_rc = RemoteChannel(
        () -> Channel{Tuple{S,S,Vector{EllipticBarrier{S}}}}(
            results_queue_length,
        ),
    )

    #Producer job
    function makejob(vc)
        # set up Poincaré section
        vx = vc.coords[1]
        vy = vc.coords[2]
        vr = xspan[findlast(x -> x <= vx + p.boxradius, xspan.val)]
        # localize tensor field
        T_local = T[
            ClosedInterval(vx - p.boxradius, vx + p.boxradius),
            ClosedInterval(vy - p.boxradius, vy + p.boxradius),
        ]
        put!(jobs_rc, (vx, vy, vr, p, outermost, T_local))
    end

    #Start an asynchronous producer task that puts stuff onto jobs_rc
    if !debug
        producer_task = @async try
            map(makejob, vortexcenters)
            isopen(jobs_rc) && close(jobs_rc)
        catch e
            print("Error in producing jobs for workers: ")
            println(e)
            flush(stdout)
            isopen(results_rc) && close(results_rc)
            isopen(jobs_rc) && close(jobs_rc)
        end
    else
        map(makejob, vortexcenters)
        close(jobs_rc)
    end

    #This is run as consumer job on workers
    function consumer_job()
        error_on_take = false
        try
            num_processed = 0
            while true
                if debug && num_processed == num_jobs
                    close(results_rc)
                    return 0
                end

                error_on_take = true
                vx, vy, vr, p, outermost, T_local = take!(jobs_rc)
                error_on_take = false
                #Setup seed points, if we are close to the right boundary then fewer points are used.
                vs = range(
                    vx,
                    stop = vr,
                    length = 1 + ceil(Int, (vr - vx) / p.boxradius * p.n_seeds),
                )

                cache = orient(T_local[:, :], SVector{2}(vx, vy))
                ps = SVector{2}.(vs, vy)

                result = compute_closed_orbits(ps, ηfield, cache; rev = outermost, p = p)
                put!(results_rc, (vx, vy, result))
                num_processed += 1
            end
        catch e
            if debug
                rethrow(e)
            end

            if error_on_take && !isopen(jobs_rc)
                return 0
            else
                print("Worker: ")
                println(e)
                flush(stdout)
                isopen(results_rc) && close(results_rc)
                return 1
            end
        end
    end # consumer_job

    #Start the consumer jobs
    if debug
        @info "Starting vortex-calculation"
        consumer_job()
        @info "Done with vortex-calculation"
    else
        consumer_jobs = map(p -> remotecall(consumer_job, p), workers())
    end

    num_barriers = 0
    if verbose
        pm = Progress(num_jobs, desc = "Detecting vortices")
    end
    foreach(1:num_jobs) do i
        vx, vy, barriers = take!(results_rc)
        num_barriers += length(barriers)
        if verbose
            ProgressMeter.next!(
                pm;
                showvalues = [(:num_barriers, num_barriers)],
            )
        end
        push!(vortices, EllipticVortex(SVector{2}(vx, vy), barriers))
    end

    if !debug
        #Cleanup, make sure everything finished etc...
        wait(producer_task)
        isopen(jobs_rc) && close(jobs_rc)
        isopen(results_rc) && close(results_rc)
        if 1 ∈ wait.(consumer_jobs)
            raise(AssertionError("Caught error on worker"))
        end
    end

    #Get rid of vortices without barriers
    vortexlist = filter(v -> !isempty(v.barriers), vortices)
    verbose &&
    @info "Found $(sum(map(v -> length(v.barriers), vortexlist))) elliptic barriers in total."
    return vortexlist
end
function getvortices(
    T::AbstractMatrix{<:SymmetricTensor{2,2}},
    xspan::AbstractRange,
    yspan::AbstractRange,
    vortexcenters::AbstractVector{<:Singularity},
    p::LCSParameters = LCSParameters();
    kwargs...
) where {S<:Real}
    return getvortices(AxisArray(T, xspan, yspan), vortexcenters, p; kwargs...)
end

#TODO: Document this more etc...
function makeVortexListUnique(vortices::Vector{<:EllipticVortex}, indexradius)
    N = length(vortices)
    if N == 0
        return vortices
    end
    which_not_to_add = falses(N)
    vortexcenters = [v.center for v in vortices]
    vortexcenters_tree = NN.KDTree(vortexcenters, Dists.Euclidean())
    result = typeof(vortices[1])[]
    for i in 1:N
        which_not_to_add[i] && continue
        idxs2 = NN.inrange(vortexcenters_tree, vortexcenters[i], 2 * indexradius)
        for j in idxs2
            j == i && continue
            c1 = [SVector{2}(p[1], p[2]) for p in vortices[j].barriers[1].curve]
            c2 = [SVector{2}(p[1], p[2]) for p in vortices[i].barriers[1].curve]
            which_not_to_add[j] = contains_point(c1, vortexcenters[i]) ||
                contains_point(c2, vortexcenters[j])
        end
        push!(result, vortices[i])
    end
    return result
end

"""
    constrainedLCS(T::AbstractMatrix, xspan, yspan, p; kwargs...)
    constrainedLCS(T::AxisArray, p; kwargs...)

Computes constrained transport barriers as closed orbits of the transport vector
field on the 2D computational grid spanned by `xspan` and `yspan`.
`p` is an [`LCSParameters`](@ref)-type container of computational parameters.
Returns a list of `EllipticBarrier`-type objects.

The keyword arguments and their default values are:
*   `outermost=true`: only the outermost barriers, i.e., the vortex
    boundaries are returned, otherwise all detected transport barrieres;
*   `verbose=true`: show intermediate computational information
*   `debug=false`: whether to use the debug mode, which avoids parallelization
    for more precise error messages.
"""
function constrainedLCS(
    q::AbstractMatrix{<:SVector{2}},
    xspan::AbstractRange,
    yspan::AbstractRange,
    p::LCSParameters = LCSParameters();
    kwargs...,
)
    constrainedLCS(AxisArray(q, xspan, yspan), p; kwargs...)
end
function constrainedLCS(
    q::AxisArray{<:SVector{2,S},2},
    p::LCSParameters = LCSParameters();
    outermost::Bool = true,
    verbose::Bool = true,
    debug = false,
    singularity_predicate = nothing,
    suggested_centers = Singularity{S}[],
) where {S}
    # detect centers of elliptic (in the index sense) regions
    xspan = q.axes[1]
    xmax = xspan[end]
    # TODO: unfortunately, this type assertion is required for inferrability
    critpts::Vector{Singularity{S}} = critical_point_detection(
        q,
        p.indexradius;
        merge_heuristics=p.merge_heuristics,
    )
    append!(critpts, suggested_centers)
    verbose && @info "Found $(length(critpts)) critical points..."
    vortexcenters = critpts[getindices(critpts) .== 1]
    verbose && @info "Defined $(length(vortexcenters)) Poincaré sections..."

    vortices = EllipticVortex{S}[]

    #Type of restricted field is quite complex, therefore make a variable for it here
    qType = AxisArrays.AxisArray{
        SVector{2,S},
        2,
        Array{SVector{2,S},2},
        Tuple{
            AxisArrays.Axis{
                :row,
                StepRangeLen{S,Base.TwicePrecision{S},Base.TwicePrecision{S}},
            },
            AxisArrays.Axis{
                :col,
                StepRangeLen{S,Base.TwicePrecision{S},Base.TwicePrecision{S}},
            },
        },
    }

    # We make two remote channels. The master process pushes to jobs_rc in order
    # (vx, vy, vr, p, outermost, T_local):
    #     * vx::S,vy::S (coordinates of vortex center)
    #     * vr::S (length of Poincaré section)
    #     * p::LCSParameters
    #     * q_local (A local copy of the vector field)
    #     * outermost::Bool (whether to only search for outermost barriers)
    # Worker processes/tasks take elements from jobs_rc, calculate barriers, and put
    # the results in results_rc
    num_jobs = length(vortexcenters)

    jobs_queue_length = debug ? num_jobs : nprocs()
    results_queue_length = debug ? num_jobs : 2 * nprocs()


    jobs_rc = RemoteChannel(
        () -> Channel{Tuple{S,S,S,LCSParameters,Bool,qType}}(nprocs()),
    )
    results_rc = RemoteChannel(
        () -> Channel{Tuple{S,S,Vector{EllipticBarrier{S}}}}(2 * nprocs()),
    )

    #Producer job
    function makejob(vc)
        # set up Poincaré section
        vx = vc.coords[1]
        vy = vc.coords[2]
        vr = xspan[findlast(x -> x <= vx + p.boxradius, xspan.val)]
        # localize tensor field
        q_local = q[
            ClosedInterval(vx - p.boxradius, vx + p.boxradius),
            ClosedInterval(vy - p.boxradius, vy + p.boxradius),
        ]
        put!(jobs_rc, (vx, vy, vr, p, outermost, q_local))
    end

    #Start an asynchronous producer task that puts stuff onto jobs_rc
    if !debug
        producer_task = @async try
            map(makejob, vortexcenters)
            isopen(jobs_rc) && close(jobs_rc)
        catch e
            print("Error in producing jobs for workers: ")
            println(e)
            flush(stdout)
            close(jobs_rc)
            close(results_rc)
        end
    else
        foreach(makejob, vortexcenters)
        close(jobs_rc)
    end

    #This is run as consumer job on workers
    function consumer_job()
        error_on_take = false #See whether the take call failed
        try
            num_processed = 0
            while true
                if debug && num_processed == num_jobs
                    close(results_rc)
                    return 0
                end

                error_on_take = true
                vx, vy, vr, p, outermost, q_local = take!(jobs_rc)
                error_on_take = false

                #Setup seed points, if we are close to the right boundary then fewer points are used.
                vs = range(
                    vx,
                    stop = vr,
                    length = 1 + ceil(Int, (vr - vx) / p.boxradius * p.n_seeds),
                )
                ps = SVector{2}.(vs, vy)

                cache = deepcopy(q_local)
                normsqq = map(v -> norm(v)^2, q_local)
                nitp = ITP.LinearInterpolation(normsqq)
                invnormsqq = map(x -> iszero(x) ? one(x) : inv(x), normsqq)
                q1 = invnormsqq .* q_local
                function constrainedLCSηfield(λ, s, cache)
                    cache .=
                        sqrt.(max.(normsqq .- (λ^2), 0)) .* q1 +
                        ((-1)^s * λ) .* (Ω,) .* q1
                    itp = ITP.LinearInterpolation(cache)
                    return ODE.ODEFunction{false}(
                        (u, p, t) -> itp(u[1], u[2]),
                    )
                end

                result = compute_closed_orbits(ps, constrainedLCSηfield, cache; rev = outermost, p = p)
                put!(results_rc, (vx, vy, result))
                num_processed += 1
            end
        catch e
            if debug
                rethrow(e)
            end
            if !isopen(jobs_rc) && error_on_take
                return 0
            else
                print("Worker: ")
                println(e)
                flush(stdout)
                isopen(results_rc) && close(results_rc)
                return 1
            end
        end
    end # consumer_job

    if debug
        @info "Starting vortex-calculation"
        consumer_job()
        @info "Done with vortex-calculation"
    else
        consumer_jobs = map(p -> remotecall(consumer_job, p), workers())
    end

    num_barriers = 0
    if verbose
        pm = Progress(num_jobs, desc = "Detecting vortices")
    end
    map(1:num_jobs) do i
        vx, vy, barriers = take!(results_rc)
        num_barriers += length(barriers)
        if verbose
            ProgressMeter.next!(
                pm;
                showvalues = [(:num_barriers, num_barriers)],
            )
        end
        push!(vortices, EllipticVortex(SVector{2}(vx, vy), barriers))
    end

    if !debug
        #Cleanup, make sure everything finished etc...
        wait(producer_task)
        isopen(jobs_rc) && close(jobs_rc)
        isopen(results_rc) && close(results_rc)
        if 1 ∈ wait.(consumer_jobs)
            raise(AssertionError("Caught error on worker"))
        end
    end

    #Get rid of vortices without barriers
    vortexlist = filter(v -> !isempty(v.barriers), vortices)
    verbose &&
    @info "Found $(sum(map(v -> length(v.barriers), vortexlist))) elliptic barriers in total."
    return vortexlist, critpts
end

"""
    in_defined_squares(xs, cache)

Check whether a vector of points `xs`, interpreted as a curve, passes only cells of the
grid on which the line/vector field is well-behaved. Well-behavior is tested by checking
whether any pair of vectors at the corner points point in opposite directions, i.e.,
``v_i \\cdot v_j < 0`` for corner vectors ``v_i`` and ``v_j``.
"""
function in_defined_squares(xs, cache)
    xspan = cache.η.axes[1].val
    yspan = cache.η.axes[2].val
    nx = length(xspan)
    ny = length(yspan)

    for x in xs
        xid = floor(
            Int,
            (nx - 1) * (x[1] - xspan[1]) /
            (xspan[end] - xspan[1] + step(xspan)),
        )
        yid = floor(
            Int,
            (ny - 1) * (x[2] - yspan[1]) /
            (yspan[end] - yspan[1] + step(xspan)),
        )

        xid = xid == nx ? nx - 1 : xid
        yid = yid == ny ? ny - 1 : yid

        ps = [cache.η[xid+di+1, yid+dj+1] for di in (0, 1), dj in (0, 1)]

        for i in 1:4, j in (i+1):4
            ps[i] ⋅ ps[j] < 0 && return false
        end
    end
    return true
end

"""
    in_uniform_squares(xs, λ⁰, cache)

Check whether a vector of points `xs`, interpreted as a curve, passes only cells of the
grid on which the η-field equation is well-posed. Well-posedness is tested by checking
whether the parameter `λ⁰` lies between `λ₁` and `λ₂` at the corner points.
"""
function in_uniform_squares(xs, λ⁰, cache)
    xspan = cache.η.axes[1].val
    yspan = cache.η.axes[2].val
    nx = length(xspan)
    ny = length(yspan)

    for x in xs
        xid = floor(
            Int,
            (nx - 1) * (x[1] - xspan[1]) /
            (xspan[end] - xspan[1] + step(xspan)),
        )
        yid = floor(
            Int,
            (ny - 1) * (x[2] - yspan[1]) /
            (yspan[end] - yspan[1] + step(yspan)),
        )

        xid = xid == nx ? nx - 1 : xid
        yid = yid == ny ? ny - 1 : yid

        l1s = [cache.λ₁[xid+di+1, yid+dj+1] for di in (0, 1), dj in (0, 1)]
        l2s = [cache.λ₂[xid+di+1, yid+dj+1] for di in (0, 1), dj in (0, 1)]

        for i in 1:4
            !(l1s[i] <= λ⁰ <= l2s[i]) && return false
        end
    end
    return true
end

function contains_point(xs, point_to_check)
    points_to_center = [x - point_to_check for x in xs]
    angles = [atan(x[2], x[1]) for x in points_to_center]
    lx = length(xs)
    res = 0.0
    for i in 0:(lx-1)
        res += s1dist(angles[i+1], angles[((i+1)%lx)+1])
    end
    res /= 2π
    return !iszero(round(res))
end

function materialbarriersTensors(
    odefun,
    xspan,
    yspan,
    tspan,
    lcsp;
    δ = 1e-6,
    tolerance = 1e-6,
    p = nothing,
    kwargs...,
)
    P0 = AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
    Tfun = let tspan = tspan, δ = δ, p = p, tol = tolerance, rhs = odefun
        u -> av_weighted_CG_tensor(
            rhs,
            u,
            tspan,
            δ;
            p = p,
            tolerance = tol,
            kwargs...,
        )
    end
    return pmap(
        Tfun,
        P0;
        batch_size = div(length(P0), Distributed.nprocs()^2),
    )
end

"""
    materialbarriers(odefun, xspan, yspan, tspan, lcsp;
        on_torus=false, δ=1e-6, tolerance=1e-6, p=nothing, kwargs...)

Calculate material barriers to diffusive and stochastic transport on the material domain
spanned by `xspan` and `yspan`, where the averaged weighted CG tensor is computed at the
time instance contained in `tspan`. The argument `lcsp` must be of type
[`LCSParameters`](@ref), and contains parameters used for the elliptic vortex detection.
"""
function materialbarriers(
    odefun,
    xspan,
    yspan,
    tspan,
    lcsp;
    δ = 1e-6,
    tolerance = 1e-6,
    p = nothing,
    on_torus = false,
    kwargs...,
)
    T0 = materialbarriersTensors(
        odefun,
        xspan,
        yspan,
        tspan,
        lcsp;
        δ = δ,
        tolerance = tolerance,
        p = p,
    )
    bg = map(tr, T0)

    if !on_torus
        vortices, singularities =
            ellipticLCS(T0, lcsp; singularity_predicate = (_ -> true), kwargs...)
        return vortices, singularities, bg
    else
        xmin = xspan[1]
        xmax = xspan[end] + step(xspan)
        xdiff = xmax - xmin
        ymin = yspan[1]
        ymax = yspan[end] + step(yspan)
        ydiff = ymax - ymin
        nx = length(xspan)
        ny = length(yspan)
        xrange =
            range(xmin - xdiff, stop = xmax + xdiff, length = 3 * nx + 1)[1:end-1]
        yrange =
            range(ymin - ydiff, stop = ymax + ydiff, length = 3 * ny + 1)[1:end-1]
        predicate = let xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
            x -> (xmin <= x.coords[1] < xmax) && (ymin <= x.coords[2] < ymax)
        end
        T = AxisArray(
            hvcat((3, 3, 3), T0, T0, T0, T0, T0, T0, T0, T0, T0),
            xrange,
            yrange,
        )
        vortices, singularities =
            ellipticLCS(T, lcsp; singularity_predicate = predicate, kwargs...)
        return vortices, singularities, bg
    end
end

### Some convenience functions
function flow(odefun::ODE.ODEFunction, u::Singularity, tspan; kwargs...)
    newcoords = flow(odefun, u.coords, tspan; kwargs...)
    return Singularity.(newcoords, u.index)
end
function flow(odefun::ODE.ODEFunction, curve::Vector{<:SVector}, tspan; kwargs...)
    newcurves = [flow(odefun, x, tspan; kwargs...) for x in curve]
    return map(eachindex(tspan)) do t
        [nc[t] for nc in newcurves]
    end
end
function flow(odefun::ODE.ODEFunction, barrier::EllipticBarrier{T}, tspan; kwargs...) where {T<:Real}
    newcurves = flow(odefun, barrier.curve, tspan; kwargs...)
    newcores = flow(odefun, barrier.core, tspan; kwargs...)
    return EllipticBarrier{T}.(newcurves, newcores, barrier.p, barrier.s)
end
function flow(odefun::ODE.ODEFunction, vortex::EllipticVortex{T}, tspan; kwargs...) where {T}
    newcenters  = flow(odefun, vortex.center, tspan; kwargs...)
    newbarriers = [flow(odefun, barrier, tspan; kwargs...) for barrier in vortex.barriers]
    newbarriers2 = map(eachindex(tspan)) do t
        [barrier[t] for barrier in newbarriers]
    end
    return EllipticVortex{T}.(newcenters, newbarriers2)
end

function refine!(ccurve::Vector{T}, ncurve::Vector{T}, odefun, tspan, params; kwargs...) where {T<:SVector}
    i = firstindex(ncurve)
    while i < lastindex(ncurve)-1
        if i > firstindex(ncurve)
            κ = curvature(ncurve[i-1], ncurve[i], ncurve[i+1])
        else
            κ = 0.0
        end
        d = norm(ncurve[i+1] - ncurve[i])
        if (d > params.maxdist) || ((κ > params.maxcurv) && (d > params.mindist))
            if (i > firstindex(ncurve)) && (i < lastindex(ncurve)-1)
                cadd = cubicinterp(ccurve[i-1], ccurve[i], ccurve[i+1], ccurve[i+2])
            else
                cadd = (ccurve[i] + ccurve[i+1])/2
            end
            if any(isnan, cadd)
                i += 1
            else # point insertion is potentially suboptimal
                insert!(ccurve, i+1, cadd)
                insert!(ncurve, i+1, flow(odefun, cadd, tspan; kwargs...)[2])
            end
        else
            i += 1
        end
    end
    return ccurve
end

"""
    flowgrow(odefun, curve, tspan, params; kwargs...)

Advect `curve` with point insertion by the ODE with right hand side `odefun` over
the time interval `tspan`, evaluated at each element of `tspan`. This method is
known in oceanography as Dritschel's method. The point insertion method is controlled
by the parameters stored in `params::FlowGrowParams`; cf. [`FlowGrowParams`](@ref).
Keyword arguments `kwargs` are passed to the [`flow`](@ref) function.

Convenience methods for [`EllipticBarrier`](@ref) and [`EllipticVortex`] objects
in place of `curve` exist. In this case, the method returns a vector of length
`length(tspan)` of corresponding objects.
"""
function flowgrow(odefun, curve::Vector{<:SVector}, tspan, params; kwargs...)
    nt = length(tspan)
    newcurves = Vector{typeof(curve)}(undef, nt)
    newcurves[1] = deepcopy(curve)
    for i in 1:nt-1
        ts = tspan[i:i+1]
        newcurves[i+1] = flow(odefun, newcurves[i], ts; kwargs...)[2]
        refine!(newcurves[i], newcurves[i+1], odefun, ts, params; kwargs...)
    end
    return newcurves
end
function flowgrow(odefun, barrier::EllipticBarrier, tspan, params::FlowGrowParams=FlowGrowParams(); kwargs...)
    newcores  = flow(odefun, barrier.core, tspan; kwargs...)
    newcurves = flowgrow(odefun, barrier.curve, tspan, params; kwargs...)
    return typeof(barrier).(newcurves, newcores, barrier.p, barrier.s)
end
function flowgrow(odefun, vortex::EllipticVortex{T}, tspan, params::FlowGrowParams=FlowGrowParams(); kwargs...) where {T}
    newcenters  = flow(odefun, vortex.center, tspan; kwargs...)
    newbarriers = [flowgrow(odefun, barrier, tspan, params; kwargs...) for barrier in vortex.barriers]
    newbarriers2 = map(eachindex(tspan)) do t
        [nb[t] for nb in newbarriers]
    end
    return typeof(vortex).(newcenters, newbarriers2)
end

"""
    area(polygon)

Compute the enclosed area of `polygon`, which can be of type `Vector{SVector{2}}`,
`EllipticBarrier` or `EllipticVortex`. In the latter case, the enclosed area of
the outermost (i.e., the last `EllipticBarrier` in the `barriers` field) is computed.
"""
function area(poly::Vector{SVector{2,T}}) where {T}
    a = zero(T)
    @inbounds @simd for i in 1:length(poly)-1
        p1 = poly[i]
        p2 = poly[i+1]
        a += p1[1]*p2[2]-p2[1]*p1[2]
    end
    p1 = last(poly)
    p2 = first(poly)
    a += p1[1]*p2[2]-p2[1]*p1[2]
    return 0.5*a
end
area(barrier::EllipticBarrier) = area(barrier.curve)
area(vortex::EllipticVortex)   = area(last(vortex.barriers))
