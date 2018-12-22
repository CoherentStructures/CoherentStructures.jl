# (c) 2018 Daniel Karrasch & Nathanael Schilling

"""
    struct Singularity

## Fields
* `coords::SVector{2,Float64}`: coordinates of the singularity
* `index::Int`: index of the singularity
"""
struct Singularity{T <: Real}
    coords::SVector{2,T}
    index::Int64
end

"""
    getcoords(singularities)
Extracts the coordinates of `singularities`, a vector of `Singularity`s. Returns
a vector of `SVector`s.
"""
function getcoords(singularities::Vector{Singularity{T}}) where T
    return [s.coords for s in singularities]
end

"""
    getindices(singularities)
Extracts the indices of `singularities`, a vector of `Singularity`s.
"""
function getindices(singularities::Vector{Singularity{T}}) where T
    return [s.index for s in singularities]
end

"""
    struct EllipticBarrier

This is a container for coherent vortex boundaries. An object `vortex` of type
`EllipticBarrier` can be easily plotted by `plot(vortex.curve)`, or
`plot!([figure, ]vortex.curve)` if it is to be overlaid over an existing plot.

## Fields
* `curve`: a list of tuples, contains the coordinates of coherent vortex boundary
  points;
* `core`: location of the vortex core;
* `p`: contains the parameter value of the direction field ``\\eta_{\\lambda}^{\\pm}``,
  for the `curve` is a closed orbit;
* `s`: a `Bool` value, which encodes the sign in the formula of the direction
  field ``\\eta_{\\lambda}^{\\pm}`` via the formula ``(-1)^s``.
"""
struct EllipticBarrier{T <: Real}
    curve::Vector{Tuple{T,T}}
    core::SVector{2,T}
    p::Float64
    s::Bool
end

"""
    struct LCSParameters

Container for parameters used in elliptic LCS computations.

## Fields
* `indexradius::Float64=0.1`: radius for singularity type detection
* `boxradius::Float64=0.5`: "radius" of localization square for closed orbit detection
* `combine_pairs=true`: whether isolated singularity pairs should be merged
* `n_seeds::Int64=60`: number of seed points on the Poincaré section
* `pmin::Float64=0.7`: lower bound on the parameter in the ``\\eta``-field
* `pmax::Float64=1.5`: upper bound on the parameter in the ``\\eta``-field
* `rdist::Float64=1e-4`: required return distances for closed orbits
"""
struct LCSParameters
    indexradius::Float64
    boxradius::Float64
    combine_pairs::Bool
    n_seeds::Int64
    pmin::Float64
    pmax::Float64
    rdist::Float64
end
function LCSParameters(;
            indexradius::Float64=0.1,
            boxradius::Float64=0.5,
            combine_pairs::Bool=true,
            n_seeds::Int64=60,
            pmin::Float64=0.7,
            pmax::Float64=1.5,
            rdist::Float64=1e-4)

    LCSParameters(indexradius, boxradius, combine_pairs, n_seeds, pmin, pmax, rdist)
end

struct LCScache{Ts <: Real, Tv <: SVector{2,<: Real}}
    λ₁::AxisArray{Ts,2}
    λ₂::AxisArray{Ts,2}
    Δ::AxisArray{Ts,2}
    α::AxisArray{Ts,2}
    β::AxisArray{Ts,2}
    ξ₁::AxisArray{Tv,2}
    ξ₂::AxisArray{Tv,2}
    η::AxisArray{Tv,2}
end

##################### singularity/critical point detection #####################
"""
    compute_singularities(α, modulus) -> Vector{Singularity}

Computes critical points/singularities of vector and line fields, respectively.
`α` is a scalar field (array) which is assumed to contain some consistent angle
representation of the vector/line field. Choose `modulus=2π` for vector
fields, and `modulus=π` for line fields.
"""
function compute_singularities(α::AxisArray{<:Real,2}, modulus)
    xspan, yspan = α.axes
    singularities = Singularity{typeof(step(xspan.val) / 2)}[] # sing_out
    xstephalf = step(xspan.val) / 2
    ystephalf = step(yspan.val) / 2
    # go counter-clockwise around each grid cell and add angles
    # for cells with non-vanishing index, collect cell midpoints
    for (j,y) in enumerate(yspan[1:end-1]), (i,x) in enumerate(xspan[1:end-1])
        temp  = periodic_diff(α[i+1,j], α[i,j], modulus) # to the right
        temp += periodic_diff(α[i+1,j+1], α[i+1,j], modulus) # to the top
        temp += periodic_diff(α[i,j+1], α[i+1,j+1], modulus) # to the left
        temp += periodic_diff(α[i,j], α[i,j+1], modulus) # to the bottom
        index = round(Int, temp/modulus)
        if index != 0
            push!(singularities, Singularity(SVector{2}(x + xstephalf, y + ystephalf), index))
        end
    end
    return singularities
end

"""
    combine_singularities(sing_coordinates, sing_indices, combine_distance) -> Vector{Singularity}

This function does the equivalent of:
Build a graph where singularities are vertices, and two vertices share
an edge iff the coordinates of the corresponding vertices (given by `sing_coordinates`)
have a distance leq `combine_distance`. Find all connected components of this graph,
and return a list of their mean coordinate and sum of `sing_indices`
"""
function combine_singularities(singularities::Vector{Singularity{T}}, combine_distance::Real) where {T}

    #Do a breath-first search of all singularities
    #that are "connected" in the sense of
    #there being a path of singularities with each
    #segment less than `combine_distance` to it
    #Average the coordinates, add the indices

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
        current_coords = @SVector [0.0, 0.0]
        num_combined = 0

        #Breadth-first-search
        stack = Int64[]
        push!(stack, i)
        while !isempty(stack)
            current_singularity = pop!(stack)
            sing_seen[i] = true
            closeby_sings = NN.inrange(sing_tree, singularities[current_singularity].coords, combine_distance)
            for neighbour_index ∈ closeby_sings
                if !(sing_seen[neighbour_index])
                    sing_seen[neighbour_index] = true
                    push!(stack, neighbour_index)
                end
            end
            #Average coordinates & add indices
            current_index += singularities[current_singularity].index
            current_coords = (num_combined * current_coords +
                    singularities[current_singularity].coords) / (num_combined + 1)
            num_combined += 1
        end
        if current_index != 0
            push!(combined_singularities, Singularity(current_coords, current_index))
        end
    end

    return combined_singularities
end

"""
    combine_isolated_pairs(singularities)
Determines singularities which are mutually closest neighbors and combines them
as one, while adding their indices.
"""
function combine_isolated_pairs(singularities::Vector{Singularity{T}}) where T
    N = length(singularities)
    N == 1 && return singularities
    sing_tree = NN.KDTree(getcoords(singularities), Dists.Euclidean())
    sing_seen = falses(N)

    new_singularities = Singularity{T}[] # sing_out
    sing_out_weight = Int64[]

    for i in 1:N
        if sing_seen[i] == true
            continue
        end
        sing_seen[i] = true

        if singularities[i].index != 1
            push!(new_singularities, singularities[i])
            continue
        end
        #We have an index +1/2 singularity
        idxs, dists = NN.knn(sing_tree, singularities[i].coords, 2, true)
        nn_idx = idxs[2]

        #We've already dealt with the nearest neighbor (but didn't find
        #this one as nearest neighbor), or it isn't a wedge
        if sing_seen[nn_idx] || singularities[nn_idx].index != 1
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
        push!(new_singularities, Singularity(0.5 * (singularities[i].coords + singularities[nn_idx].coords), 2))
    end
    return new_singularities
end

"""
    critical_point_detection(vs, combine_distance, γ; combine_pairs=true)

Computes critical points of a vector/line field `vs`, given as an `AxisArray`.
Critical points with distance less or equal to `combine_distance` are
combined by averaging the coordinates and adding the respective indices. The parameter
`γ` should be chosen `π` for line fields and `2π` for vector fields; cf.
[`compute_singularities`](@ref). If `combine_pairs is `true, pairs of singularities
that are mutually the closest ones are included in the final list.
"""
function critical_point_detection(vs::AxisArray{<: SVector{2,<:Real},2},
                                    combine_distance::Real,
                                    γ::Real;
                                    combine_pairs=true)
    α = map(v -> atan(v[2], v[1]), vs)
    singularities = compute_singularities(α, γ)
    new_singularities = combine_singularities(singularities, combine_distance)
    if combine_pairs
        #There could still be wedge-singularities that
        #are separated by more than combine_distance
        return combine_isolated_pairs(new_singularities)
    else
        return new_singularities
    end
end

"""
    singularity_detection(T, combine_distance; combine_isolated_wedges=true) -> Vector{Singularity}

Calculates line-field singularities of the first eigenvector of `T` by taking
a discrete differential-geometric approach. Singularities are calculated on each
cell. Singularities with distance less or equal to `combine_distance` are
combined by averaging the coordinates and adding the respective indices. If
`combine_pairs` is `true, pairs of singularities that are mutually the
closest ones are included in the final list.

Returns a vector of [`Singularity`](@ref)s. Returned indices correspond to doubled
indices to get integer values.
"""
function singularity_detection(T::AxisArray{S,2},
                                combine_distance::Float64;
                                combine_pairs=true) where {S <: SymmetricTensor{2,2,<:Real,3}}
    ξ = map(t -> convert(SVector{2}, eigvecs(t)[:,1]), T)
    critical_point_detection(ξ, combine_distance, π; combine_pairs=combine_pairs)
end

######################## closed orbit computations #############################
"""
    compute_returning_orbit(vf, seed::SVector{2}, save::Bool=false)
Computes returning orbits under the velocity field `vf`, originating from `seed`.
The optional argument `save` controls whether intermediate locations of the
returning orbit should be saved.
"""
function compute_returning_orbit(vf, seed::SVector{2,T}, save::Bool=false) where T <: Real

    condition(u, t, integrator) = u[2] - seed[2]
    affect!(integrator) = OrdinaryDiffEq.terminate!(integrator)
    cb = OrdinaryDiffEq.ContinuousCallback(condition, nothing, affect!)
    # return _flow(vf, seed, range(0., stop=20., length=200); tolerance=1e-8, callback=cb, verbose=false)
    prob = OrdinaryDiffEq.ODEProblem(vf, seed, (0., 20.))
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(), maxiters=2e3,
            dense=false, save_everystep=save, reltol=1e-8, abstol=1e-8,
            callback=cb, verbose=false).u
end

function Poincaré_return_distance(vf, seed::SVector{2,T}, save::Bool=false) where T <: Real

    sol = compute_returning_orbit(vf, seed, save)
    # check if result due to callback
    if abs(sol[end][2] - seed[2]) <= 1e-1
        return sol[end][1] - seed[1]
    else
        return NaN
    end
end

function bisection(f, a::T, b::T, tol::Real=1e-4, maxiter::Int=15) where T <: Real
    fa, fb = f(a), f(b)
    fa*fb <= 0 || error("No real root in [a,b]")
    i = 0
    local c
    while b-a > tol
        i += 1
        i != maxiter || error("Max iteration exceeded")
        c = (a + b) / 2 # bisection
        # c = (a*fb-b*fa)/(fb-fa) # regula falsi
        fc = f(c)
        if abs(fc) < tol
            break
        elseif fa * fc > 0
            a = c  # Root is in the right half of [a,b].
            fa = fc
        else
            b = c  # Root is in the left half of [a,b].
        end
    end
    # @info "needed $(2+i) function evaluations"
    return c
end

function orient(T::AxisArray{SymmetricTensor{2,2,S1,3},2}, center::SVector{2,S2}) where {S1 <: Real, S2 <: Real}
    xspan, yspan = T.axes
    λ₁, λ₂, ξ₁, ξ₂, _, _ = tensor_invariants(T)
    Δλ = AxisArray(λ₂ .- λ₁, T.axes)
    Ω = SMatrix{2,2}(0., -1., 1., 0.)
    star = AxisArray([SVector{2}(x, y) - center for x in xspan.val, y in yspan.val], T.axes)
    c1 = AxisArray(sign.(dot.([Ω] .* star, ξ₁)), T.axes)
    ξ₁ .*= c1
    c2 = AxisArray(sign.(dot.(star, ξ₂)), T.axes)
    ξ₂ .*= c2
    LCScache(λ₁, λ₂, Δλ, c1, c2, ξ₁, ξ₂, star)
end

"""
    compute_closed_orbits(ps, ηfield, cache; rev=true, pmin=0.7, pmax=1.5, rdist=1e-4)

Compute the outermost closed orbit for a given Poincaré section `ps`, a vector field
constructor `ηfield`, and an LCScache `cache`. Keyword arguments `pmin` and `pmax`
correspond to the range of shift parameters in which closed orbits are sought;
`rev` determines whether closed orbits are sought from the outside inwards (`true`)
or from the inside outwards (`false`). `rdist` sets the required return distance for
an orbit to be considered as closed.
"""
function compute_closed_orbits(ps::AbstractVector{SVector{2,S1}},
                                ηfield,
                                cache;
                                rev::Bool=true,
                                pmin::Real=0.7,
                                pmax::Real=1.5,
                                rdist::Real=1e-4
                                ) where {S1 <: Real}
    if cache isa LCScache
        l1itp = ITP.LinearInterpolation(cache.λ₁)
        l2itp = ITP.LinearInterpolation(cache.λ₂)
    else
        nitp = ITP.LinearInterpolation(map(v -> norm(v)^2, cache))
    end
    # define local helper functions for the η⁺/η⁻ closed orbit detection
    prd(λ::Float64, σ::Bool, seed::SVector{2,S1}, cache) =
            Poincaré_return_distance(ηfield(λ, σ, cache), seed)

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
    # ηfield(λ::Float64) = OrdinaryDiffEq.ODEFunction((u, p, t) -> ηitp(u[1], u[2], λ))
    # prd(λ::Float64, seed::SVector{2,S}) = Poincaré_return_distance(ηfield(λ), seed)
    # END OF VERSION 2

    # go along the Poincaré section and solve for λ⁰ such that orbits close up
    vortices = EllipticBarrier[]
    idxs = rev ? (length(ps):-1:2) : (2:length(ps))
    for i in idxs
        λ⁰ = 0.0
        try
            global σ = false
            λ⁰ = bisection(λ -> prd(λ, σ, ps[i], cache), pmin, pmax, rdist)
        catch
            global σ = true
            try
                λ⁰ = bisection(λ -> prd(λ, σ, ps[i], cache), pmin, pmax, rdist)
            catch
            end
        end
        if !iszero(λ⁰)
            orbit = compute_returning_orbit(ηfield(λ⁰, σ, cache), ps[i], true)
            closed = norm(orbit[1] - orbit[end]) <= rdist
            predicate = qs -> cache isa LCScache ?
                l1itp(qs[1], qs[2]) <= λ⁰ <= l2itp(qs[1], qs[2]) :
                nitp(qs[1], qs[2]) >= λ⁰^2
            uniform = all(predicate, orbit)
            if (closed && uniform)
                push!(vortices, EllipticBarrier([qs.data for qs in orbit], ps[1], λ⁰, σ))
                rev && break
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
*   `verbose=true`: show intermediate computational information
"""
function ellipticLCS(T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                        xspan::AbstractRange{S},
                        yspan::AbstractRange{S},
                        p::LCSParameters=LCSParameters();
                        kwargs...) where S <: Real
    ellipticLCS(AxisArray(T, xspan, yspan), p; kwargs...)
end
function ellipticLCS(T::AxisArray{SymmetricTensor{2,2,S,3},2},
                        p::LCSParameters=LCSParameters();
                        outermost::Bool=true,
                        verbose::Bool=true) where S <: Real
    # detect centers of elliptic (in the index sense) regions
    xspan = T.axes[1]
    xmax = xspan[end]
    singularities = singularity_detection(T, p.indexradius; combine_pairs=p.combine_pairs)
    verbose && @info "Found $(length(singularities)) singularities..."
    vortexcenters = singularities[getindices(singularities) .== 2]
    verbose && @info "Defined $(length(vortexcenters)) Poincaré sections..."

    # loop over potential vortex centers, return detected closed orbits
    vortexlists = pmap(vortexcenters) do vc
        # set up Poincaré section
        vx = vc.coords[1]
        vy = vc.coords[2]
        vr = xspan[findlast(x -> x <= vx + p.boxradius, xspan.val)]
        vs = range(vx, stop=vr, length=1+ceil(Int, (vr - vx) / p.boxradius * p.n_seeds))
        ps = SVector{2}.(vs, vy)

        # localize tensor field
        T_local = T[ClosedInterval(vx - p.boxradius, vx + p.boxradius), ClosedInterval(vy - p.boxradius, vy + p.boxradius)]

        # for computational tractability, pre-orient the eigenvector fields
        # restrict search to star-shaped coherent vortices
        # ξ₁ is oriented counter-clockwise, ξ₂ is oriented outwards
        cache = orient(T_local, vc.coords)

        # vector field constructor function
        @inline ηfield(λ::Float64, σ::Bool, c::LCScache) = begin
            @. c.α = min(sqrt(max(c.λ₂ - λ, 0) / c.Δ), 1)
            @. c.β = min(sqrt(max(λ - c.λ₁, 0) / c.Δ), 1)
            @. c.η = c.α * c.ξ₁ + ((-1) ^ σ) * c.β * c.ξ₂
            itp = ITP.LinearInterpolation(c.η)
            return OrdinaryDiffEq.ODEFunction((u, p, t) -> itp(u[1], u[2]))
        end

        # closed orbits extraction
        if verbose
            result, t, _ = @timed compute_closed_orbits(ps, ηfield, cache;
                    rev=outermost, pmin=p.pmin, pmax=p.pmax, rdist=p.rdist)
            @info "Vortex candidate $(ps[1]) was finished in $t seconds and " *
                "yielded $(length(result)) transport barrier" *
                (length(result) > 1 ? "s." : ".")
            return result
        else
            return compute_closed_orbits(ps, ηfield, cache;
                    rev=outermost, pmin=p.pmin, pmax=p.pmax, rdist=p.rdist)
        end
    end

    vortexlist = vcat(vortexlists...)
    verbose && @info "Found $(length(vortexlist)) elliptic barriers in total."
    return vortexlist, singularities
end

function constrainedLCS(q::AbstractMatrix{SVector{2,<:Real}},
                        xspan::AbstractRange{<:Real},
                        yspan::AbstractRange{<:Real},
                        p::LCSParameters=LCSParameters();
                        kwargs...)
    constrainedLCS(AxisArray(q, xspan, yspan), p; kwargs...)
end
function constrainedLCS(q::AxisArray{SVector{2,S},2},
                        p::LCSParameters=LCSParameters();
                        outermost::Bool=true,
                        verbose::Bool=true) where S <: Real
    # detect centers of elliptic (in the index sense) regions
    xspan = q.axes[1]
    xmax = xspan[end]
    critpts = critical_point_detection(q, p.indexradius, 2π; combine_pairs=p.combine_pairs)
    verbose && @info "Found $(length(critpts)) critical points..."
    vortexcenters = critpts[getindices(critpts) .== 1]
    verbose && @info "Defined $(length(vortexcenters)) Poincaré sections..."

    # loop over potential vortex centers, return detected closed orbits
    vortexlists = pmap(vortexcenters) do vc
        # set up Poincaré section
        vx = vc.coords[1]
        vy = vc.coords[2]
        vr = xspan[findlast(x -> x <= vx + p.boxradius, xspan.val)]
        vs = range(vx, stop=vr, length=1+ceil(Int, (vr - vx) / p.boxradius * p.n_seeds))
        ps = SVector{2}.(vs, vy)

        # localize tensor field
        q_local = q[ClosedInterval(vx - p.boxradius, vx + p.boxradius), ClosedInterval(vy - p.boxradius, vy + p.boxradius)]

        # vector field constructor function
        Ω = SMatrix{2,2}(0., -1., 1., 0.)
        cache = deepcopy(q_local)
        normsqq = map(v -> norm(v)^2, q_local)
        nitp = ITP.LinearInterpolation(normsqq)
        invnormsqq = map(x -> iszero(x) ? one(x) : inv(x), normsqq)
        @inline function ηfield(λ, s, cache)
            cache .= sqrt.(max.(normsqq .- (λ^2), 0)) .* invnormsqq .* q_local +
                            ((-1)^s * λ) .* invnormsqq .* [Ω] .* q_local
            itp = ITP.LinearInterpolation(cache)
            return OrdinaryDiffEq.ODEFunction((u, p ,t) -> itp(u[1], u[2]))
        end

        # closed orbits extraction
        if verbose
            result, t, _ = @timed compute_closed_orbits(ps, ηfield, cache;
                    rev=outermost, pmin=p.pmin, pmax=p.pmax, rdist=p.rdist)
            @info "Vortex candidate $(ps[1]) was finished in $t seconds and " *
                "yielded $(length(result)) transport barrier" *
                (length(result) > 1 ? "s." : ".")
            return result
        else
            return compute_closed_orbits(ps, ηfield, cache;
                    rev=outermost, pmin=p.pmin, pmax=p.pmax, rdist=p.rdist)
        end
    end

    vortexlist = vcat(vortexlists...)
    verbose && @info "Found $(length(vortexlist)) elliptic barriers in total."
    return vortexlist, critpts
end
