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

"""
This is a container for coherent vortex boundaries. An object `vortex` of type
`EllipticBarrier` can be easily plotted by `plot(vortex.curve)`, or
`plot!([figure, ]vortex.curve)` if it is to be overlaid over an existing plot.

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
    curve::Vector{Tuple{T,T}}
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

"""
Container for parameters used in elliptic LCS computations.

## Fields
* `boxradius`: "radius" of localization square for closed orbit detection
* `indexradius=1e-1boxradius`: radius for singularity type detection
* `merge_heuristics`: a list of heuristics for combining singularities, supported are
* * `combine_20`: merge isolated singularity pairs that are mutually nearest neighbors
* * `combine_31`: merge 1 trisector + nearest-neighbor 3 wedge configurations.
* * `combine_20_aggressive`: an additional wedge combination heuristic
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
```
julia> p = LCSParameters(2.5)
LCSParameters(2.5, 0.25, true, 100, 0.7, 2.0, 0.00025, 2.5e-8, 1000, 20.0, 30)
```
"""
struct LCSParameters
    boxradius::Float64
    indexradius::Float64
    merge_heuristics::Vector{Any}
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
        merge_heuristics = [combine_20],
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
        return new(
            float(boxradius),
            float(indexradius),
            merge_heuristics,
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
    merge_heuristics = [combine_20],
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
        merge_heuristics,
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
    s1dist(α, β)

Computes the signed length of the angle of the shortest circle segment going
from angle `β` to angle `α`, as computed on the full circle.

# Examples
```jldoctest
julia> s1dist(π/2, 0)
1.5707963267948966

julia> s1dist(0, π/2)
-1.5707963267948966
```
"""
@inline s1dist(x::Real, y::Real) = rem2pi(float(x - y), RoundNearest)

"""
    p1dist(α, β)

Computes the signed length of the angle of the shortest circle segment going
from angle `β` to angle `α [± π]`, as computed on the half circle.

# Examples
```jldoctest
julia> p1dist(π, 0)
0.0
```
"""
@inline p1dist(x::Real, y::Real) = rem(float(x - y), float(π), RoundNearest)

##################### singularity/critical point detection #####################
"""
    compute_singularities(v, dist=s1dist) -> Vector{Singularity}

Computes critical points and singularities of vector and line fields `v`,
respectively. The argument `dist` is a signed distance function for angles.
Choose `s1dist` (default) for vector fields, and `p1dist` for line fields.
"""
function compute_singularities(
    v::AxisArray{<:SVector{2},2},
    dist::Function = s1dist,
)
    xspan, yspan = v.axes
    α = map(u -> atan(u[2], u[1]), v)
    xstephalf = step(xspan.val) / 2
    ystephalf = step(yspan.val) / 2
    T = typeof(xstephalf)
    singularities = Singularity{T}[]
    # go counter-clockwise around each grid cell and add angles
    # for cells with non-vanishing index, collect cell midpoints
    for (j, y) in enumerate(yspan[1:end-1]), (i, x) in enumerate(xspan[1:end-1])
        temp = dist(α[i+1, j], α[i, j]) # to the right
        temp += dist(α[i+1, j+1], α[i+1, j]) # to the top
        temp += dist(α[i, j+1], α[i+1, j+1]) # to the left
        temp += dist(α[i, j], α[i, j+1]) # to the bottom
        index = round(Int, temp / π) // 2
        if !iszero(index)
            push!(
                singularities,
                Singularity((x + xstephalf, y + ystephalf), index),
            )
        end
    end
    return singularities
end

"""
    combine_singularities(sing_coordinates, sing_indices, combine_distance) -> Vector{Singularity}

This function does the equivalent of: build a graph where singularities are vertices, and
two vertices share an edge iff the coordinates of the corresponding vertices (given by
`sing_coordinates`) have a distance leq `combine_distance`. Find all connected components of
this graph, and return a list of their mean coordinate and sum of `sing_indices`.
"""
function combine_singularities(
    singularities::Vector{Singularity{T}},
    combine_distance::Real,
)::Vector{Singularity{T}} where {T}
    # Do a breath-first search of all singularities that are "connected" in the
    # sense that there is a path of singularities with each segment less than
    # `combine_distance`
    # Average the coordinates, add the indices

    N = length(singularities)

    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    #Which singularities we've already dealt with
    sing_seen = falses(N)

    #Result will go here
    combined_singularities = Singularity{T}[]

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
                combine_distance,
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
                (
                    num_combined * current_coords +
                    singularities[current_singularity].coords
                ) / (num_combined + 1)
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
    combine_20(singularities)

Determines singularities which are mutually closest neighbors and combines them as one,
while adding their indices.
"""
function combine_20(
    singularities::Vector{Singularity{T}},
)::Vector{Singularity{T}} where {T}
    N = length(singularities)
    N == 1 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    sing_seen = falses(N)

    new_singularities = Singularity{T}[] # sing_out
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

"""
    combine_31(singularities)

Takes the list of singularities in `singularities` and combines them
so that any -1/2 singularity whose three nearest neighbors are 1/2 singularities
becomes an elliptic region, provided that the -1/2 singularity
is in the triangle spanned by the wedges. This configuration
is common for OECS, applying to material barriers on a large
turbulent example yielded only about an additional 1% material barriers.
"""
function combine_31(
    singularities::Vector{Singularity{T}},
)::Vector{Singularity{T}} where {T}
    N = length(singularities)
    N <= 2 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    sing_seen = falses(N)

    new_singularities = Singularity{T}[] # sing_out
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

"""
    combine_20_aggressive(singularities)

A heuristic for combining singularities which is likely to have a lot of false positives.
"""
function combine_20_aggressive(
    singularities::Vector{Singularity{T}},
)::Vector{Singularity{T}} where {T}
    N = length(singularities)
    N == 1 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    combined_with = [Int[] for i in 1:N]
    sing_seen = falses(N)

    new_singularities = Singularity{T}[] # sing_out
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


"""
    critical_point_detection(vs, combine_distance, dist=s1dist; merge_heuristics=[combine_20]) -> Vector{Singularity}

Computes critical points of a vector/line field `vs`, given as an `AxisArray`.
Critical points with distance less or equal to `combine_distance` are
combined by averaging the coordinates and adding the respective indices. The
argument `dist` is a signed distance function for angles: choose [`s1dist`](@ref)
for vector fields, and [`p1dist`](@ref) for line fields; cf. [`compute_singularities`](@ref).
Heuristics listed as functions in `merge_heuristics`, cf. [`LCSParameters`](@ref),
are applied to combine singularities.

Returns a vector of [`Singularity`](@ref)s.
"""
function critical_point_detection(
    vs::AxisArray{<:SVector{2,T},2},
    combine_distance::Real,
    dist = s1dist;
    merge_heuristics = [combine_20],
)::Vector{Singularity{T}} where {T<:Real}
    singularities = compute_singularities(vs, dist)
    new_singularities = combine_singularities(singularities, combine_distance)
    for f in merge_heuristics
        new_singularities = f(new_singularities)
    end
    return new_singularities
end

"""
    singularity_detection(T, combine_distance; merge_heuristics=[combine_20]) -> Vector{Singularity}

Calculate line-field singularities of the first eigenvector of `T` by taking a discrete
differential-geometric approach. Singularities are calculated on each cell. Singularities
with distance less or equal to `combine_distance` are combined by averaging the coordinates
and adding the respective indices. The heuristics listed in `merge_heuristics` are used to
merge singularities, cf. [`LCSParameters`](@ref).

Return a vector of [`Singularity`](@ref)s.
"""
function singularity_detection(
    T::AxisArray{S,2},
    combine_distance::Real;
    merge_heuristics = [combine_20],
) where {S<:SymmetricTensor{2,2}}
    ξ = map(t -> convert(SVector{2}, eigvecs(t)[:, 1]), T)
    critical_point_detection(
        ξ,
        combine_distance,
        p1dist;
        merge_heuristics = merge_heuristics,
    )
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
    vf,
    seed::SVector{2,T},
    save::Bool = false,
    maxiters::Int = 2000,
    tolerance::Float64 = 1e-8,
    max_orbit_length::Float64 = 20.0,
) where {T<:Real}
    dir = vf(seed, nothing, 0.0)[2] < 0 ? -1 : 1 # Whether orbits initially go upwards
    condition(u, t, integrator) = dir * (seed[2] - u[2])
    affect!(integrator) = OrdinaryDiffEq.terminate!(integrator)
    cb = OrdinaryDiffEq.ContinuousCallback(condition, nothing, affect!)
    prob = OrdinaryDiffEq.ODEProblem(vf, seed, (0.0, max_orbit_length))
    try
        sol = OrdinaryDiffEq.solve(
            prob,
            OrdinaryDiffEq.Tsit5(),
            maxiters = maxiters,
            dense = false,
            save_everystep = save,
            reltol = tolerance,
            abstol = tolerance,
            callback = cb,
            verbose = false,
        )
        if sol.retcode == :Terminated
            retcode = 0
        elseif sol.retcode == :MaxIters
            retcode = 1
        else
            retcode = 3
        end
        return (sol.u, retcode)
    catch e
        if e isa BoundsError
            return (SVector{2,T}(NaN, NaN), 2)
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
    if retcode == 0
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
    c1 = AxisArray(sign.(dot.([Ω] .* star, ξ₁)), T.axes)
    ξ₁ .*= c1
    c2 = AxisArray(sign.(dot.(star, ξ₂)), T.axes)
    ξ₂ .*= c2
    return LCScache(λ₁, λ₂, Δλ, c1, c2, ξ₁, ξ₂, star)
end

"""
    compute_closed_orbits(ps, ηfield, cache; rev=true, pmin=0.7, pmax=1.5, rdist=1e-4, tolerance_ode=1e-8, maxiters_ode=2000, maxiters_bisection=20)

Compute the (outermost) closed orbit for a given Poincaré section `ps`, a vector field
constructor `ηfield`, and an LCScache `cache`. Keyword arguments `pmin` and `pmax`
correspond to the range of shift parameters in which closed orbits are sought;
`rev` determines whether closed orbits are sought from the outside inwards (`true`)
or from the inside outwards (`false`). `rdist` sets the required return distance for
an orbit to be considered as closed. The parameter `maxiters_ode` gives the maximum number
of steps taken by the ODE solver when computing the closed orbit, the ode solver uses tolerance
given by `tolerance_ode`. The parameter `maxiters_bisection` gives the maximum number of iterations
used by the bisection algorithm to find closed orbits.
"""
function compute_closed_orbits(
    ps::AbstractVector{SVector{2,S1}},
    ηfield,
    cache;
    rev::Bool = true,
    pmin::Real = 0.7,
    pmax::Real = 1.5,
    rdist::Real = 1e-4,
    tolerance_ode::Float64 = 1e-8,
    maxiters_ode::Int = 2000,
    max_orbit_length::Float64 = 20.0,
    maxiters_bisection::Int = 20,
    only_enclosing = true,
    only_smooth = true,
    only_uniform = true,
) where {S1<:Real}
    if cache isa LCScache # tensor-based LCS computation
        l1itp = ITP.LinearInterpolation(cache.λ₁)
        l2itp = ITP.LinearInterpolation(cache.λ₂)
    else # vector-field-based LCS computation
        nitp = ITP.LinearInterpolation(map(v -> norm(v)^2, cache))
    end
    # define local helper functions for the η⁺/η⁻ closed orbit detection
    prd(λ::Float64, σ::Bool, seed::SVector{2}, cache) =
        let tol = tolerance_ode,
            maxode = maxiters_ode,
            maxorbit = max_orbit_length

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
    # ηfield(λ::Float64) = OrdinaryDiffEq.ODEFunction{false}((u, p, t) -> ηitp(u[1], u[2], λ))
    # prd(λ::Float64, seed::SVector{2,S}) = Poincaré_return_distance(ηfield(λ), seed)
    # END OF VERSION 2

    # go along the Poincaré section and solve for λ⁰ such that orbits close up
    vortices = EllipticBarrier{S1}[]
    idxs = rev ? (length(ps):-1:2) : (2:length(ps))
    for i in idxs
        if cache isa LCScache && only_uniform
            pmin_local = max(pmin, l1itp(ps[i][1], ps[i][2]))
            pmax_local = min(pmax, l2itp(ps[i][1], ps[i][2]))
            margin_step = (pmax_local - pmin_local) / 20
            if !(margin_step > 0)
                continue
            end
        else #TODO: can something like the above be done for the constrained LCS setting too?
            pmin_local = pmin
            pmax_local = pmax
            margin_step = (pmax_local - pmin_local) / 20
        end

        σ = false
        bisection_retcode, λ⁰ = bisection(
            let σ = σ, ps = ps[i], cache = cache
                λ -> prd(λ, σ, ps, cache)
            end,
            pmin_local,
            pmax_local,
            rdist,
            maxiters_bisection,
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
                rdist,
                maxiters_bisection,
                margin_step,
            )
        end
        if bisection_retcode == zero_found
            orbit, retcode = compute_returning_orbit(
                ηfield(λ⁰, σ, cache),
                ps[i],
                true,
                maxiters_ode,
                tolerance_ode,
                max_orbit_length,
            )
            if retcode == 0
                closed = norm(orbit[1] - orbit[end]) <= rdist
                if cache isa LCScache
                    in_well_defined_squares =
                        !only_smooth || in_defined_squares(orbit, cache)
                    uniform =
                        !only_uniform || in_uniform_squares(orbit, λ⁰, cache)
                else
                    predicate = let λ = λ⁰
                        qs -> nitp(qs[1], qs[2]) >= λ^2
                    end
                    in_well_defined_squares = true
                    uniform = !only_uniform || all(predicate, orbit)
                end

                contains_singularity =
                    !only_enclosing || contains_point(orbit, ps[1])

                if (
                    closed &&
                    uniform && in_well_defined_squares && contains_singularity
                )
                    push!(
                        vortices,
                        EllipticBarrier(
                            [qs.data for qs in orbit],
                            ps[1],
                            λ⁰,
                            σ,
                        ),
                    )
                    rev && break
                end
            end
        end
    end
    return vortices
end

"""
    ellipticLCS(T::AbstractArray, xspan, yspan, p; kwargs...)
    ellipticLCS(T::AxisArray, p; kwargs...)

Computes elliptic LCSs as null-geodesics of the Lorentzian metric tensor
field given by shifted versions of `T` on the 2D computational grid spanned
by `xspan` and `yspan`. `p` is a [`LCSParameters`](@ref)-type container of
computational parameters. Returns a list of `EllipticBarrier`-type objects.

The keyword arguments and their default values are:
*   `outermost=true`: only the outermost barriers, i.e., the vortex
    boundaries are returned, otherwise all detected transport barrieres;
*   `verbose=true`: show intermediate computational information;
*   `debug=false`: whether to use the debug mode, which avoids parallelization
    for more precise error messages.
*   `singularity_predicate = nothing`: provide an optional callback to reject certain singularity candidates.
"""
function ellipticLCS(
    T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
    xspan::AbstractRange{S},
    yspan::AbstractRange{S},
    p::LCSParameters = LCSParameters();
    kwargs...,
) where {S<:Real}
    ellipticLCS(AxisArray(T, xspan, yspan), p; kwargs...)
end
function ellipticLCS(
    T::AxisArray{<:SymmetricTensor{2,2,S},2},
    p::LCSParameters = LCSParameters();
    verbose::Bool = true,
    unique_vortices = true,
    singularity_predicate = nothing,
    suggested_centers = Singularity{S}[],
    kwargs...,
) where {S<:Real}
    # detect centers of elliptic (in the index sense) regions
    singularities = append!(
        suggested_centers,
        singularity_detection(
            T,
            p.indexradius;
            merge_heuristics = p.merge_heuristics,
        ),
    )
    if singularity_predicate !== nothing
        singularities = filter(singularity_predicate, singularities)
    end
    verbose && @info "Found $(length(singularities)) singularities..."
    vortexcenters = filter(s -> s.index == 1, singularities)
    verbose && @info "Defined $(length(vortexcenters)) Poincaré sections..."

    vortices = getvortices(T, vortexcenters, p; verbose = verbose, kwargs...)
    if unique_vortices
        vortices = makeVortexListUnique(vortices, p.indexradius)
    end
    return vortices::Vector{<:EllipticVortex},
    singularities::Vector{<:Singularity}
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
            sol, retcode = compute_returning_orbit(
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
    return OrdinaryDiffEq.ODEFunction{false}(unit_length_itp)
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
* `debug=true`: whether parallel computation should be used. Set to `false` to
  turn off parallel computation and to obtain more useful error messages.
"""
function getvortices(
    T::AxisArray{SymmetricTensor{2,2,S,3},2},
    vortexcenters::Vector{<:Singularity},
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
    Ttype = AxisArrays.AxisArray{
        SymmetricTensor{2,2,S,3},
        2,
        Array{SymmetricTensor{2,2,S,3},2},
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
    #     * T_local (A local copy of the tensor field)
    #     * outermost::Bool (whether to only search for outermost barriers)
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

                result = compute_closed_orbits(
                    ps,
                    ηfield,
                    cache;
                    rev = outermost,
                    pmin = p.pmin,
                    pmax = p.pmax,
                    rdist = p.rdist,
                    tolerance_ode = p.tolerance_ode,
                    maxiters_ode = p.maxiters_ode,
                    max_orbit_length = p.max_orbit_length,
                    maxiters_bisection = p.maxiters_bisection,
                    only_enclosing = p.only_enclosing,
                    only_smooth = p.only_smooth,
                    only_uniform = p.only_uniform,
                )
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
    return vortexlist::Vector{<:EllipticVortex}
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
        idxs2 =
            NN.inrange(vortexcenters_tree, vortexcenters[i], 2 * indexradius)
        for j in idxs2
            j == i && continue
            c1 = [
                SVector{2}(p[1], p[2]) for p in vortices[j].barriers[1].curve
            ]
            c2 = [
                SVector{2}(p[1], p[2]) for p in vortices[i].barriers[1].curve
            ]
            if contains_point(c1, vortexcenters[i]) ||
               contains_point(c2, vortexcenters[j])
                which_not_to_add[j] = true
            end
        end
        push!(result, vortices[i])
    end
    return result::Vector{<:EllipticVortex}
end

"""
    constrainedLCS(T::AbstractArray, xspan, yspan, p; kwargs...)
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
    q::AbstractMatrix{SVector{2,<:Real}},
    xspan::AbstractRange{<:Real},
    yspan::AbstractRange{<:Real},
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
) where {S<:Real}
    # detect centers of elliptic (in the index sense) regions
    xspan = q.axes[1]
    xmax = xspan[end]
    critpts = critical_point_detection(
        q,
        p.indexradius;
        merge_heuristics = p.merge_heuristics,
    )
    verbose && @info "Found $(length(critpts)) critical points..."
    vortexcenters = critpts[getindices(critpts).==1]
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
        map(makejob, vortexcenters)
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
                        ((-1)^s * λ) .* [Ω] .* q1
                    itp = ITP.LinearInterpolation(cache)
                    return OrdinaryDiffEq.ODEFunction{false}(
                        (u, p, t) -> itp(u[1], u[2]),
                    )
                end

                result = compute_closed_orbits(
                    ps,
                    constrainedLCSηfield,
                    cache;
                    rev = outermost,
                    pmin = p.pmin,
                    pmax = p.pmax,
                    rdist = p.rdist,
                    tolerance_ode = p.tolerance_ode,
                    maxiters_ode = p.maxiters_ode,
                    max_orbit_length = p.max_orbit_length,
                    maxiters_bisection = p.maxiters_bisection,
                    only_enclosing = p.only_enclosing,
                    only_smooth = p.only_smooth,
                )
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
    return vortexlist::Vector{<:EllipticVortex}, critpts::Vector{<:Singularity}
end


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
    on_torus = false,
    kwargs...,
)
    P0 = AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
    Tfun = let tspan = tspan, δ = δ, p = p, tol = tolerance
        u -> av_weighted_CG_tensor(
            odefun,
            u,
            tspan,
            δ;
            p = p,
            tolerance = tolerance,
        )
    end
    T0 = pmap(
        Tfun,
        P0;
        batch_size = div(length(xspan) * length(yspan), Distributed.nprocs()),
    )
    if !on_torus
        T = T0
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
        T = AxisArray(
            hvcat((3, 3, 3), T0, T0, T0, T0, T0, T0, T0, T0, T0),
            xrange,
            yrange,
        )
    end
    return T, T0
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
    T, T0 = materialbarriersTensors(
        odefun,
        xspan,
        yspan,
        tspan,
        lcsp;
        δ = δ,
        tolerance = tolerance,
        p = p,
        on_torus = on_torus,
    )
    if !on_torus
        predicate = x -> true
    else
        xmin = xspan[1]
        xmax = xspan[end] + step(xspan)
        ymin = yspan[1]
        ymax = yspan[end] + step(yspan)
        predicate =
            x -> (xmin <= x.coords[1] < xmax) && (ymin <= x.coords[2] < ymax)
    end
    vortices, singularities =
        ellipticLCS(T, lcsp; singularity_predicate = predicate, kwargs...)
    return vortices, singularities, tensor_invariants(T0)[5]
end

### Some convenience functions

function flow(odefun, u::EllipticBarrier{T}, tspan; kwargs...) where {T<:Real}
    nt = length(tspan)
    nc = length(u.curve)
    newcurves = map(
        let odefun = odefun, tspan = tspan, kwargs = kwargs
            x -> flow(odefun, SVector{2}(x[1], x[2]), tspan; kwargs...)
        end,
        u.curve,
    )
    newcurves2 = Vector{Tuple{T,T}}[]
    for i in 1:nt
        curcurve = Tuple{T,T}[]
        for j in 1:nc
            push!(curcurve, (newcurves[j][i][1], newcurves[j][i][2]))
        end
        push!(newcurves2, curcurve)
    end
    newcores = flow(odefun, u.core, tspan; kwargs...)
    return [
        EllipticBarrier{T}(newcurves2[i], newcores[i], u.p, u.s) for i in 1:nt
    ]
end

function flow(odefun, u::Singularity, tspan; kwargs...)
    nt = length(tspan)
    newcoords = flow(odefun, u.coords, tspan; kwargs...)
    return [Singuarity(newcoords[i], u.index) for i in 1:nt]
end

function flow(odefun, u::EllipticVortex{T}, tspan; kwargs...) where {T}
    newbarriers = [flow(odefun, b, tspan; kwargs...) for b in u.barriers]
    newbarriers2 = Vector{EllipticBarrier{T}}[]
    nt = length(tspan)
    nb = length(u.barriers)
    for i in 1:nt
        curbarriers = EllipticBarrier{T}[]
        for j in 1:nb
            push!(curbarriers, newbarriers[j][i])
        end
        push!(newbarriers2, curbarriers)
    end
    newcenters = flow(odefun, u.center, tspan; kwargs...)
    return [EllipticVortex{T}(newcenters[i], newbarriers2[i]) for i in 1:nt]
end
