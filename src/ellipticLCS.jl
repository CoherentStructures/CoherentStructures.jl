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

function get_coords(singularities::Vector{Singularity{T}}) where T
    return [s.coords for s in singularities]
end
function get_indices(singularities::Vector{Singularity{T}}) where T
    return [s.index for s in singularities]
end

"""
    struct EllipticBarrier

This is a container for coherent vortex boundaries. An object `vortex` of type
`EllipticBarrier` can be easily plotted by `plot(vortex.curve)`, or
`plot!([figure, ]vortex.curve)` if it is to be overlaid over an existing plot.

## Fields
* `curve`: a list of tuples; contains the coordinates of coherent vortex boundary
  points.
* `p`: contains the parameter value of the direction field ``\\eta_{\\lambda}^{\\pm}``,
  for the `curve` is a closed orbit.
* `s`: a `Bool` value, which encodes the sign in the formula of the direction
  field ``\\eta_{\\lambda}^{\\pm}`` via the formula ``(-1)^s``.
"""
struct EllipticBarrier{T <: Real}
    curve::Vector{Tuple{T,T}}
    p::Float64
    s::Bool
end

"""
    struct LCSParameters

Container for parameters used in elliptic LCS computations.

## Fields
* `radius::Float64=0.1`: radius for singularity type detection
* `p_length::Float64=0.5`: lenght of the Poincaré section
* `n_seeds::Int64=40`: number of seed points on the Poincaré section
* `pmin::Float64=0.7`: lower bound on the parameter in the ``\\eta``-field
* `pmax::Float64=1.3`: upper bound on the parameter in the ``\\eta``-field
* `rdist::Float64=1e-4`: required return distances for closed orbits
"""
struct LCSParameters
    radius::Float64
    # MaxWedgeDist`::Float64 # maximum distance to closest wedge
    # MinWedgeDist::Float64 # minimal distance to closest wedge
    # Min2ndDist::Float64 # minimal distance to second closest wedge
    p_length::Float64
    n_seeds::Int64
    pmin::Float64
    pmax::Float64
    rdist::Float64
end
function LCSParameters(;
            radius::Float64=0.1,
            # MaxWedgeDist::Float64=0.5,
            # MinWedgeDist::Float64=0.04,
            # Min2ndDist::Float64=0.5,
            p_length::Float64=0.5,
            n_seeds::Int64=60,
            pmin::Float64=0.7,
            pmax::Float64=1.5,
            rdist::Float64=1e-4)

    LCSParameters(radius,
        # MaxWedgeDist, MinWedgeDist, Min2ndDist,
        p_length, n_seeds, pmin, pmax, rdist)
end

struct LCScache{Ts <: Real, Tv <: SVector{2,<:Real}}
    λ₁::Array{Ts, 2}
    λ₂::Array{Ts, 2}
    Δ::Array{Ts, 2}
    α::Array{Ts, 2}
    β::Array{Ts, 2}
    ξ₁::Array{Tv, 2}
    ξ₂::Array{Tv, 2}
    η::Array{Tv, 2}
end

# """
#     singularity_location_detection(T, xspan, yspan)
#
# Detects tensor singularities of the tensor field `T`, given as a matrix of
# `SymmetricTensor{2,2}`. `xspan` and `yspan` correspond to the uniform
# grid vectors over which `T` is given. Returns a list of static 2-vectors.
# """
# function singularity_location_detection(T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
#                                         xspan::AbstractVector{S},
#                                         yspan::AbstractVector{S}) where S
#
#     z1 = [c[1]-c[4] for c in T]
#     z2 = [c[2] for c in T]
#     zdiff = z1-z2
#     # C = Contour.contours(xspan,yspan,zdiff,[0.])
#     cl = Contour.levels(Contour.contours(yspan, xspan, zdiff, [0.]))[1]
#     sitp = ITP.LinearInterpolation((xspan, yspan), permutedims(z1))
#     # itp = ITP.interpolate(z1, ITP.BSpline(ITP.Linear()))
#     # sitp = ITP.extrapolate(ITP.scale(itp, yspan, xspan), (ITP.Reflect(), ITP.Reflect(), ITP.Reflect()))
#     Xs, Ys = Float64[], Float64[]
#     for line in Contour.lines(cl)
#         yL, xL = Contour.coordinates(line)
#         zL = [sitp(xL[i], yL[i]) for i in eachindex(xL, yL)]
#         ind = findall(zL[1:end-1] .* zL[2:end] .<= 0)
#         zLind = -zL[ind] ./ (zL[ind .+ 1] - zL[ind])
#         Xs = append!(Xs, xL[ind] + (xL[ind .+ 1] - xL[ind]) .* zLind)
#         Ys = append!(Ys, yL[ind] + (yL[ind .+ 1] - yL[ind]) .* zLind)
#     end
#     return [SVector{2}(Xs[i], Ys[i]) for i in eachindex(Xs, Ys)]
# end
#
# """
#     singularity_type_detection(singularity, ξ, radius)
#
# Determines the singularity type of the singularity candidate `singularity`
# by querying the direction field `ξ` in a circle of radius `radius` around the
# singularity. Returns `1` for a trisector, `-1` for a wedge, and `0` otherwise.
# """
# function singularity_type_detection(singularity::SVector{2,S}, ξ, radius::Real) where {S <: Real}
#
#     Ntheta = 360   # number of points used to construct a circle around each singularity
#     θ = range(-π, stop=π, length=Ntheta)
#     circle = map(t -> SVector{2,S}(radius * cos(t), radius * sin(t)), θ)
#     pnts = [singularity] .+ circle
#     radVals = [ξ(p[1], p[2]) for p in pnts]
#     singularity_type = 0
#     if (sum(diff(radVals) .< 0) / Ntheta > 0.62)
#         singularity_type = -1  # trisector
#     elseif (sum(diff(radVals) .> 0) / Ntheta > 0.62)
#         singularity_type = 1  # wedge
#     end
#     return singularity_type
# end
"""
    compute_singularities(α::ScalarField{2}, modulus) -> Vector{Singularity}

Computes critical points/singularities of vector and line fields, respectively.
`α` is a `ScalarField`, which is assumed to contain some consistent angle
representation of the vector/line field. Choose `modulus` as `2π` for vector
fields, and as `π` for line fields.
"""
function compute_singularities(α::ScalarField{2}, modulus)
    xspan, yspan = α.grid_axes
    singularities = Singularity{typeof(step(xspan) / 2)}[] # sing_out
    xstephalf = step(xspan) / 2
    ystephalf = step(yspan) / 2
    # go counter-clockwise around each grid cell and add angles
    # for cells with non-vanishing index, collect cell midpoints
    for (j,y) in enumerate(yspan[1:end-1]), (i,x) in enumerate(xspan[1:end-1])
        temp  = periodic_diff(α[j,i+1], α[j,i], modulus) # to the right
        temp += periodic_diff(α[j+1,i+1], α[j,i+1], modulus) # to the top
        temp += periodic_diff(α[j+1,i], α[j+1,i+1], modulus) # to the left
        temp += periodic_diff(α[j,i], α[j+1,i], modulus) # to the bottom
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

    sing_tree = NN.KDTree(get_coords(singularities), Dists.Euclidean())
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

function combine_isolated_wedge_pairs(singularities::Vector{Singularity{T}}) where T
    N = length(singularities)
    sing_tree = NN.KDTree(get_coords(singularities), Dists.Euclidean())
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
    discrete_singularity_detection(T, combine_distance; combine_isolated_wedges=true) -> Vector{Singularity}

Calculates line-field singularities of the first eigenvector of `T` by taking
a discrete differential-geometric approach. Singularities are calculated on each
cell. Singularities with distance less or equal to `combine_distance` are
combined by averaging the coordinates and adding the respective indices. If
`combine_isolated_wedges` is `true, pairs of indices that are mutually the
closest singularities are included in the final list.

Returns a vector of [`Singularity`](@ref)s. Indices are multiplied by 2 to get
integer values.
"""
function discrete_singularity_detection(T::SymmetricTensorField{2},
                                        combine_distance::Float64;
                                        combine_isolated_wedges=true)
    ξ = [eigvecs(t)[:,1] for t in T.tensors]
    α = ScalarField(T.grid_axes, [atan(v[2], v[1]) for v in ξ])
    singularities = compute_singularities(α, π)
    new_singularities = combine_singularities(singularities, combine_distance)
    if combine_isolated_wedges
        #There could still be wedge-singularities that
        #are separated by more than combine_distance
        return combine_isolated_wedge_pairs(new_singularities)
    else
        return new_singularities
    end
end

# """
#     detect_elliptic_region(singularities, singularityTypes, p)
#
# Determines candidate regions for closed tensor line orbits.
#    * `singularities`: list of all singularities
#    * `singularityTypes`: list of corresponding singularity types
#    * `p`: parameter container of type `LCSParameters`
# Returns a list of vortex centers.
# """
# function detect_elliptic_region(singularities::AbstractVector{SVector{2,S}},
#                                 singularity_types::AbstractVector{Int},
#                                 p::LCSParameters=LCSParameters()) where S <: Number
#
#     indWedges = findall(singularity_types .== 1)
#     wedges = singularities[indWedges]
#     wedgeDist = Dists.evaluate.([Dists.Euclidean()], wedges, wedges')
#     idx = zeros(Int64, size(wedgeDist,1), 2)
#     pairs = Vector{Int}[]
#     for i=1:size(wedgeDist, 1)
#         idx = partialsortperm(wedgeDist[i,:], 2:3)
#         if (wedgeDist[i,idx[1]] <= p.MaxWedgeDist &&
#             wedgeDist[i,idx[1]] >= p.MinWedgeDist &&
#             wedgeDist[i,idx[2]] >= p.Min2ndDist)
#             push!(pairs, [i, idx[1]])
#         end
#     end
#     pairind = unique(sort!.(intersect(pairs, reverse.(pairs, dims=1))))
#     centers = [mean(singularities[indWedges[p]]) for p in pairind]
#     return SVector{2}.(centers)
# end

"""
    set_Poincaré_section(vc, xspan, yspan, p_length=1.0, n_seeds=60)

Generates a horizontal Poincaré section, centered at the vortex center `vc`
of length `p.p_length` consisting of `p.n_seeds` starting at `0.2*p_length`
eastwards. All points are guaranteed to lie in the computational domain given
by `xspan` and `yspan`.
"""
function set_Poincaré_section(vc::SVector{2,S},
                                xspan::AbstractVector{S},
                                yspan::AbstractVector{S},
                                p_length::Real=1.0,
                                n_seeds::Int=60) where S <: Real

    xmin, xmax = extrema(xspan)
    ymin, ymax = extrema(yspan)
    p_section::Vector{SVector{2,S}} = [vc]
    eₓ = SVector{2,S}(1., 0.)
    pspan = range(vc + .2p_length*eₓ, stop=vc + p_length*eₓ, length=n_seeds)
    idxs = [all(ps .<= [xmax, ymax]) && all(ps .>= [xmin, ymin]) for ps in pspan]
    append!(p_section, pspan[idxs])
    return p_section
end

function compute_returning_orbit(vf, seed::SVector{2,T}, save::Bool=false) where T <: Real

    condition(u,t,integrator) = u[2] - seed[2]
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
    if abs(sol[end][2] - seed[2]) <= 1e-2
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

function orient(T::SymmetricTensorField{2}, center::SVector{2,S}) where {S <: Real}
    xspan, yspan = T.grid_axes
    λ₁, λ₂, ξ₁, ξ₂, _, _ = tensor_invariants(T)
    Δλ = λ₂ - λ₁
    Ω = SMatrix{2,2}(0., -1., 1., 0.)
    star = VectorField(T.grid_axes, [SVector{2}(x, y) - center for y in yspan, x in xspan])
    c1 = sign.(dot.([Ω] .* star.vecs, ξ₁.vecs))
    ξ₁.vecs .= c1 .* ξ₁.vecs
    c2 = sign.(dot.(star.vecs, ξ₂.vecs))
    ξ₂.vecs .= c2 .* ξ₂.vecs
    LCScache(λ₁.vals, λ₂.vals, Δλ.vals, c1, c2, ξ₁.vecs, ξ₂.vecs, star.vecs)
end

"""
    compute_closed_orbits(pSection, T[, xspan, yspan]; rev=true, pmin=0.7, pmax=1.5, rdist=1e-4)

Compute the outermost closed orbit for a given Poincaré section `pSection`,
tensor field `T`, where the total computational domain is spanned by `xspan`
and `yspan`. Keyword arguments `pmin` and `pmax` correspond to the range of
shift parameters in which closed orbits are sought; `rev` determines whether
closed orbits are sought from the outside inwards (`true`) or from the inside
outwards (`false`). `rdist` sets the required return distance for an orbit to be
considered as closed.
"""
function compute_closed_orbits(pSection, T, xspan, yspan; rev=true, pmin=0.7, pmax=1.5, rdist=1e-4)
    compute_closed_orbits(pSection, SymmetricTensorField((xspan, yspan), T);
                                    rev=rev, pmin=pmin, pmax=pmax, rdist=rdist)
end
function compute_closed_orbits(pSection::Vector{SVector{2,S}},
                                        T::SymmetricTensorField{2};
                                        rev::Bool=true,
                                        pmin::Real=0.7,
                                        pmax::Real=1.5,
                                        rdist::Real=1e-4
                                        ) where S <: Real

    xspan, yspan = T.grid_axes
    # for computational tractability, pre-orient the eigenvector fields
    # restrict search to star-shaped coherent vortices
    # ξ₁ is oriented counter-clockwise, ξ₂ is oriented outwards
    cache = orient(T, pSection[1])
    l1itp = ITP.scale(ITP.interpolate(permutedims(cache.λ₁), ITP.BSpline(ITP.Linear())),
                        xspan, yspan)
    l2itp = ITP.scale(ITP.interpolate(permutedims(cache.λ₂), ITP.BSpline(ITP.Linear())),
                        xspan, yspan)

    # define local helper functions for the η⁺/η⁻ closed orbit detection
    @inline ηfield(λ::Float64, σ::Bool, c::LCScache) = begin
        c.α .= min.(sqrt.(max.(c.λ₂ .- λ, 0) ./ c.Δ), 1)
        c.β .= min.(sqrt.(max.(λ .- c.λ₁, 0) ./ c.Δ), 1)
        c.η .= c.α .* c.ξ₁ .+ ((-1) ^ σ) .* c.β .* c.ξ₂
        # itp = ITP.CubicSplineInterpolation(T.grid_axes, permutedims(c.η))
        itp = ITP.scale(ITP.interpolate(permutedims(c.η), ITP.BSpline(ITP.Cubic(ITP.Natural(ITP.OnGrid())))),
                            xspan, yspan)
        return OrdinaryDiffEq.ODEFunction((u, p, t) -> itp(u[1], u[2]))
    end
    prd(λ::Float64, σ::Bool, seed::SVector{2,S}, cache::LCScache) =
            Poincaré_return_distance(ηfield(λ, σ, cache), seed)

    # VERSION 2: 3D-interpolant
    # η(λ::Float64, signum::Bool) = begin
    #     α = min.(sqrt.(max.(λ₂.vals .- λ, 0) ./ Δλ.vals), 1)
    #     β = min.(sqrt.(max.(λ .- λ₁.vals, 0) ./ Δλ.vals), 1)
    #     return α .* ξ₁.vecs .+ (-1)^signum .* β .* ξ₂.vecs
    # end
    # λrange = range(pmin, stop=pmax, length=20)
    # ηdata = cat([η(λ, false) for λ in λrange]..., dims=3)
    # ηitp = ITP.scale(ITP.interpolate(permutedims(ηdata, (2, 1, 3)),
    #         (ITP.BSpline(ITP.Cubic(ITP.Natural(ITP.OnGrid()))),
    #          ITP.BSpline(ITP.Cubic(ITP.Natural(ITP.OnGrid()))),
    #          ITP.BSpline(ITP.Linear()))),
    #                 xspan, yspan, λrange)
    # ηfield(λ::Float64) = OrdinaryDiffEq.ODEFunction((u, p, t) -> ηitp(u[1], u[2], λ))
    # prd(λ::Float64, seed::SVector{2,S}) = Poincaré_return_distance(ηfield(λ), seed)
    # END OF VERSION 2

    # go along the Poincaré section and solve for λ
    # first, define a nonlinear root finding problem
    vortices = EllipticBarrier[]
    idxs = rev ? (length(pSection):-1:2) : (2:length(pSection))
    for i in idxs
        λ⁰ = 0.0
        try
            global σ = false
            λ⁰ = bisection(λ -> prd(λ, σ, pSection[i], cache), pmin, pmax, rdist)
        catch
            global σ = true
            try
                λ⁰ = bisection(λ -> prd(λ, σ, pSection[i], cache), pmin, pmax, rdist)
            catch
            end
        end
        if !iszero(λ⁰)
            orbit = compute_returning_orbit(ηfield(λ⁰, σ, cache), pSection[i], true)
            closed = norm(orbit[1] - orbit[end]) <= rdist
            uniform = all([l1itp(ps[1], ps[2]) <= λ⁰ <= l2itp(ps[1], ps[2]) for ps in orbit])
            # @show (closed, uniform)
            # @show length(orbit)
            if (closed && uniform)
                push!(vortices, EllipticBarrier([ps.data for ps in orbit], λ⁰, σ))
                rev && break
            end
        end
    end
    return vortices
end

"""
    ellipticLCS(T, xspan, yspan, p; outermost=true)

Computes elliptic LCSs as null-geodesics of the Lorentzian metric tensor
field given by shifted versions of `T` on the 2D computational grid spanned
by `xspan` and `yspan`. `p` is a [`LCSParameters`](@ref)-type container of
computational parameters.

Returns a list of `EllipticBarrier`-type objects: if the optional keyword
argument `outermost` is true, then only the outermost barriers, i.e., the vortex
boundaries, otherwise all detected transport barrieres are returned.
"""
function ellipticLCS(T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                        xspan::AbstractRange{S},
                        yspan::AbstractRange{S},
                        p::LCSParameters=LCSParameters();
                        outermost::Bool=true) where S <: Real
    F = SymmetricTensorField((xspan, yspan), T)
    return ellipticLCS(F, p, outermost=outermost)
end
"""
    function ellipticLCS(T, p; outermost=true)

Computes elliptic LCSs as null-geodesics of the Lorentzian metric tensor
field given by (pointwise) shifted versions of the [`SymmetricTensorField`](@ref)
`T`. `p` is a [`LCSParameters`](@ref)-type container of computational parameters.

Returns a list of `EllipticBarrier`-type objects: if the optional keyword
argument `outermost` is true, then only the outermost barriers, i.e., the vortex
boundaries, otherwise all detected transport barrieres are returned.
"""
function ellipticLCS(T::SymmetricTensorField{2},
                        p::LCSParameters=LCSParameters();
                        outermost::Bool=true,
                        verbose::Bool=true) where S <: Real

    # EVERYTHING FROM HERE ...
    # singularities = singularity_location_detection(T, xspan, yspan)
    # @info "Detected $(length(singularities)) singularity candidates..."
    #
    # ξ = [eigvecs(t)[:,1] for t in T]
    # ξrad = atan.([v[2]./v[1] for v in ξ])
    # ξraditp = ITP.LinearInterpolation((xspan, yspan), permutedims(ξrad);
    #                                             extrapolation_bc=ITP.Line())
    # # ξraditp = ITP.extrapolate(ITP.scale(ITP.interpolate(ξrad,
    # #                     ITP.BSpline(ITP.Linear())),
    # #                     yspan,xspan), ITP.Reflect())
    # singularitytypes = map(singularities) do s
    #     singularity_type_detection(s, ξraditp, p.radius)
    # end
    # @info "Determined $(sum(abs.(singularitytypes))) nondegenerate singularities..."
    #
    # vortexcenters = detect_elliptic_region(singularities, singularitytypes, p)
    # ... TO HERE SHOULD BE REPLACED BY THE NEW METHOD
    xspan, yspan = T.grid_axes
    singularities = discrete_singularity_detection(T, p.radius;
                                            combine_isolated_wedges=true)
    @info "Found $(length(singularities)) singularities..."

    vortexcenters = singularities[get_indices(singularities) .== 2]
    p_section = map(vortexcenters) do vc
        set_Poincaré_section(vc.coords, xspan, yspan, p.p_length, p.n_seeds)
    end
    @info "Defined $(length(vortexcenters)) Poincaré sections..."

    vortexlists = pmap(p_section) do ps
        if verbose
            result, t, _ = @timed compute_closed_orbits(ps, T;
                    rev=outermost, pmin=p.pmin, pmax=p.pmax, rdist=p.rdist)
            @info "Vortex candidate $(ps[1]) was finished in $t seconds and " *
                "yielded $(length(result)) transport barrier" *
                (length(result) > 1 ? "s." : ".")
            return result
        else
            return compute_closed_orbits(ps, T;
                    rev=outermost, pmin=p.pmin, pmax=p.pmax, rdist=p.rdist)
        end
    end

    # closed orbits extraction
    vortexlist = vcat(vortexlists...)
    @info "Found $(length(vortexlist)) elliptic barriers in total."
    return vortexlist, singularities
end
