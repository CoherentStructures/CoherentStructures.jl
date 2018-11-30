# (c) 2018 Daniel Karrasch & Nathanael Schilling

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
            n_seeds::Int64=40,
            pmin::Float64=0.7,
            pmax::Float64=1.3,
            rdist::Float64=1e-4)

    LCSParameters(radius,
        # MaxWedgeDist, MinWedgeDist, Min2ndDist,
        p_length, n_seeds, pmin, pmax, rdist)
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
    combine_singularities(sing_coordinates, sing_indices, combine_distance)

This function does the equivalent of:
Build a graph where singularities are vertices, and two vertices share
an edge iff the coordinates of the corresponding vertices (given by `sing_coordinates`)
have a distance leq `combine_distance`. Find all connected components of this graph,
and return a list of their mean coordinate and sum of `sing_indices`
"""
function combine_singularities(sing_coordinates, sing_indices, combine_distance)

    #Do a breath-first search of all singularities
    #that are "connected" in the sense of
    #there being a path of singularities with each
    #segment less than `combine_distance` to it
    #Average the coordinates, add the indices


    n_sing = length(sing_coordinates)

    sing_tree = NN.KDTree(sing_coordinates, Dists.Euclidean())
    #Which singularities we've already dealt with
    sing_seen = falses(n_sing)

    #Result will go here
    sing_out = SVector{2,Float64}[]
    sing_out_weight = Int64[]

    #Iterate over all singularities
    for i in 1:n_sing
        if sing_seen[i]
            continue
        end
        sing_seen[i] = true

        current_weight = 0
        current_coords = @SVector [0.0, 0.0]
        num_combined = 0

        #Breadth-first-search
        stack = Int64[]
        push!(stack, i)
        while !isempty(stack)
            current_singularity = pop!(stack)
            sing_seen[i] = true
            closeby_sings = NN.inrange(sing_tree, sing_coordinates[current_singularity], combine_distance)
            for neighbour_index ∈ closeby_sings
                if !(sing_seen[neighbour_index])
                    sing_seen[neighbour_index] = true
                    push!(stack, neighbour_index)
                end
            end
            #Average coordinates & add indices
            current_weight += sing_indices[current_singularity]
            current_coords = (num_combined * current_coords + sing_coordinates[current_singularity]) /
                                    (num_combined + 1)
            num_combined += 1
        end
        if current_weight != 0
            push!(sing_out, current_coords)
            push!(sing_out_weight, current_weight)
        end
    end

    return sing_out, sing_out_weight
end

function combine_isolated_wedge_pairs(sing_coordinates, sing_indices)
    n_sing = length(sing_coordinates)
    sing_tree = NN.KDTree(sing_coordinates, Dists.Euclidean())
    sing_seen = falses(n_sing)

    sing_out = SVector{2,Float64}[]
    sing_out_weight = Int64[]

    for i in 1:n_sing
        if sing_seen[i] == true
            continue
        end
        sing_seen[i] = true

        if sing_indices[i] != 1
            push!(sing_out, sing_coordinates[i])
            push!(sing_out_weight, sing_indices[i])
            continue
        end
        #We have an index +1/2 singularity
        idxs, dists = NN.knn(sing_tree, sing_coordinates[i], 2, true)
        nn_idx = idxs[2]

        #We've already dealt with the nearest neighbor (but didn't find
        #this one as nearest neighbor), or it isn't a wedge
        if sing_seen[nn_idx] || sing_indices[nn_idx] != 1
            push!(sing_out, sing_coordinates[i])
            push!(sing_out_weight, sing_indices[i])
            continue
        end

        #See if the nearest neighbor of the nearest neighbor is i
        idxs2, dists2 = NN.knn(sing_tree, sing_coordinates[nn_idx], 2, true)
        if idxs2[2] != i
            push!(sing_out, sing_coordinates[i])
            push!(sing_out_weight, sing_indices[i])
            continue
        end

        sing_seen[nn_idx] = true
        push!(sing_out, 0.5 * (sing_coordinates[i] + sing_coordinates[nn_idx]))
        push!(sing_out_weight, 2)
    end
    return sing_out, sing_out_weight
end

"""
    discrete_singularity_detection(T, combine_distance,[xspan,yspan])

Calculates line-field singularities of the first eigenvector of `T` by taking
a discrete differential-geometric approach. Singularities are calculated on each
cell. Singularities with distance less or equal to `combine_difference` are
combined by averaging the location and adding the index. Returns a vector of
coordinates and a vector of corresponding indices. Indices are multiplied by 2
to get integer values.
"""
function discrete_singularity_detection(T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                                        combine_distance::Float64,
                                        xspan=1:(size(T)[2]), yspan=1:(size(T)[1]);
                                        combine_isolated_wedges=true
                                        ) where S
    ξ = [eigvecs(t)[:,1] for t in T]
    ξrad = [atan(v[2], v[1]) for v in ξ] .+ π/2

    nx, ny = size(T)
    cell_contents = zeros(Int, nx-1, ny-1)
    for j in 1:(ny-1), i in 1:(nx-1)
        toadd = 0.0
        toadd -= periodic_diff(ξrad[i+1,j], ξrad[i,j], π)
        toadd -= periodic_diff(ξrad[i+1,j+1], ξrad[i+1,j], π)
        toadd -= periodic_diff(ξrad[i,j+1], ξrad[i+1,j+1], π)
        toadd -= periodic_diff(ξrad[i,j], ξrad[i,j+1], π)
        cell_contents[i,j] = round(Int, toadd/π)
    end

    sing_locations = findall(!iszero, cell_contents)
    sing_indices = cell_contents[sing_locations]

    sing_i, sing_j = unzip(Tuple.(sing_locations))

    #"coordinates" of singularities at cell-midpoints
    sing_y = yspan[sing_i] .+ 0.5 * (yspan[sing_i .+ 1] - yspan[sing_i])
    sing_x = xspan[sing_j] .+ 0.5 * (xspan[sing_j .+ 1] - xspan[sing_j])
    sing_coordinates = SVector{2}.(sing_x, sing_y)

    sing_combined, sing_combined_weights = combine_singularities(
            sing_coordinates,
            sing_indices,
            combine_distance
            )

    if combine_isolated_wedges
        #There could still be wedge-singularities that
        #are separated by more than combine_distance
        #It would be a shame if we missed these
        return combine_isolated_wedge_pairs(sing_combined, sing_combined_weights)
    else
        return sing_combined, sing_combined_weights
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
    set_Poincaré_section(vc, xspan, yspan, p::LCSParameters)

Generates a horizontal Poincaré section, centered at the vortex center `vc`
of length `p.p_length` consisting of `p.n_seeds` starting at `0.2*p_length`
eastwards. All points are guaranteed to lie in the computational domain given
by `xspan` and `yspan`.
"""
function set_Poincaré_section(vc::SVector{2,S},
                                xspan::AbstractVector{S},
                                yspan::AbstractVector{S},
                                p::LCSParameters=LCSParameters()) where S <: Real

    xmin, xmax = extrema(xspan)
    ymin, ymax = extrema(yspan)
    p_section::Vector{SVector{2,S}} = [vc]
    eₓ = SVector{2,S}(1., 0.)
    pspan = range(vc + .2p.p_length*eₓ, stop=vc + p.p_length*eₓ, length=p.n_seeds)
    idxs = [all(ps .<= [xmax, ymax]) && all(ps .>= [xmin, ymin]) for ps in pspan]
    append!(p_section, pspan[idxs])
    return p_section
end

function compute_returning_orbit(vf, seed::SVector{2,T}) where T <: Real

    condition(u,t,integrator) = u[2] - seed[2]
    affect!(integrator) = OrdinaryDiffEq.terminate!(integrator)
    cb = OrdinaryDiffEq.ContinuousCallback(condition, nothing, affect!)
    # return _flow(vf, seed, range(0., stop=20., length=200); tolerance=1e-8, callback=cb, verbose=false)
    prob = OrdinaryDiffEq.ODEProblem(vf, seed, (0., 20.))
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(), maxiters=2e3,
            dense=false, reltol=1e-8, abstol=1e-8, callback=cb, verbose=false).u
end

function Poincaré_return_distance(vf, seed::SVector{2,T}) where T <: Real

    sol = compute_returning_orbit(vf, seed)
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
    return c
end

"""
    compute_closed_orbits(pSection, T, xspan, yspan; rev=true, p=LCSParameters())

Compute the outermost closed orbit for a given Poincaré section `pSection`,
tensor field `T`, where the total computational domain is spanned by `xspan`
and `yspan`. Keyword arguments `pmin` and `pmax` correspond to the range of
shift parameters in which closed orbits are sought; `rev` determines whether
closed orbits are sought from the outside inwards (`true`) or from the inside
outwards (`false`). `rdist` sets the required return distance for an orbit to be
considered as closed.
"""
function compute_closed_orbits(pSection::Vector{SVector{2,S}},
                                        T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                                        xspan::AbstractVector{S},
                                        yspan::AbstractVector{S};
                                        rev::Bool=true,
                                        p::LCSParameters=LCSParameters()
                                        ) where S <: Real

    λ₁, λ₂, ξ₁, ξ₂, _, _ = tensor_invariants(T)
    Δλ = λ₂ - λ₁
    l1itp = ITP.LinearInterpolation((xspan, yspan), permutedims(λ₁))
    # l1itp = ITP.scale(ITP.interpolate(λ₁, ITP.BSpline(ITP.Linear())),
    #                     yspan,xspan)
    l2itp = ITP.LinearInterpolation((xspan, yspan), permutedims(λ₂))
    # l2itp = ITP.scale(ITP.interpolate(λ₂, ITP.BSpline(ITP.Linear())),
    #                     yspan,xspan)

    # for computational tractability, pre-orient the eigenvector fields
    Ω = SMatrix{2,2}(0., -1., 1., 0.)
    relP = [SVector{2}(x, y) - pSection[1] for y in yspan, x in xspan]
    n = [Ω] .* relP
    ξ₁ .= sign.(n .⋅ ξ₁) .* ξ₁
    ξ₂ .= sign.(relP .⋅ ξ₂) .* ξ₂
    ηfield(calT::Float64, signum::Bool) = begin
        α = real.(sqrt.(complex.((λ₂ .- calT) ./ Δλ)))
        β = real.(sqrt.(complex.((calT .- λ₁) ./ Δλ)))
        η = α .* ξ₁ .+ (-1) ^ signum * β .* ξ₂
        ηitp = ITP.CubicSplineInterpolation((xspan, yspan), permutedims(η))
        # ηitp = ITP.scale(ITP.interpolate(η, ITP.BSpline(ITP.Cubic(ITP.Natural(ITP.OnGrid())))),
        #                     yspan, xspan)
        return OrdinaryDiffEq.ODEFunction((u, p, t) -> ηitp(u[1], u[2]))
    end
    prd(calT::Float64, signum::Bool, seed::SVector{2,S}) = begin
        vf = ηfield(calT, signum)
        Poincaré_return_distance(vf, seed)
    end

    # go along the Poincaré section and solve for T
    # first, define a nonlinear root finding problem
    vortices = EllipticBarrier[]
    idxs = rev ? (length(pSection):-1:2) : (2:length(pSection))
    for i in idxs
        Tsol = zero(Float64)
        try
            Tsol = bisection(λ -> prd(λ, false, pSection[i]), p.pmin, p.pmax, p.rdist)
            orbit = compute_returning_orbit(ηfield(Tsol, false), pSection[i])
            closed = norm(orbit[1] - orbit[end]) <= p.rdist
            uniform = all([l1itp(ps[1], ps[2]) <= Tsol <= l2itp(ps[1], ps[2]) for ps in orbit])
            # @show (closed, uniform)
            # @show length(orbit)
            if (closed && uniform)
                push!(vortices, EllipticBarrier([ps.data for ps in orbit], Tsol, false))
                rev && break
            end
        catch
        end
        if iszero(Tsol)
            try
                Tsol = bisection(λ -> prd(λ, true, pSection[i]), p.pmin, p.pmax, p.rdist)
                orbit = compute_returning_orbit(ηfield(Tsol, true), pSection[i])
                closed = norm(orbit[1] - orbit[end]) <= p.rdist
                uniform = all([l1itp(ps[1], ps[2]) <= Tsol <= l2itp(ps[1], ps[2]) for ps in orbit])
                # @show (closed, uniform)
                # @show length(orbit)
                if (closed && uniform)
                    push!(vortices, EllipticBarrier([ps.data for ps in orbit], Tsol, true))
                    rev && break
                end
            catch
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
                        xspan::AbstractVector{S},
                        yspan::AbstractVector{S},
                        p::LCSParameters=LCSParameters();
                        outermost::Bool=true) where S <: Real
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
    singularities, singularitytypes = discrete_singularity_detection(T, p.radius,
                                            xspan, yspan;
                                            combine_isolated_wedges=true)
    @info "Found $(length(singularities)) interesting singularities..."

    vortexcenters = SVector{2}.(singularities[singularitytypes .== 2])
    p_section = map(vortexcenters) do vc
        set_Poincaré_section(vc, xspan, yspan, p)
    end
    @info "Defined $(length(vortexcenters)) Poincaré sections..."

    vortexlists = pmap(p_section) do ps
        compute_closed_orbits(ps, T, xspan, yspan; rev=outermost, p=p)
    end

    # closed orbits extraction
    vortexlist = vcat(vortexlists...)
    @info "Found $(length(vortexlist)) elliptic barriers."
    return vortexlist
end
