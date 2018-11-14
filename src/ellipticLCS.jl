# (c) 2018 Daniel Karrasch

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
* `MaxWedgeDist`::Float64=0.5`: maximum distance to closest wedge
* `MinWedgeDist`::Float64=0.04`: minimal distance to closest wedge
* `Min2ndDist::Float64=0.5`: minimal distance to second closest wedge
* `p_length::Float64=0.5`: lenght of the Poincaré section
* `n_seeds::Int64=40`: number of seed points on the Poincaré section
* `pmin::Float64=0.7`: lower bound on the parameter in the ``\\eta``-field
* `pmax::Float64=1.3`: upper bound on the parameter in the ``\\eta``-field
"""
struct LCSParameters
    radius::Float64
    MaxWedgeDist::Float64
    MinWedgeDist::Float64
    Min2ndDist::Float64
    p_length::Float64
    n_seeds::Int64
    pmin::Float64
    pmax::Float64
end
function LCSParameters(;
            radius::Float64=0.1,
            MaxWedgeDist::Float64=0.5,
            MinWedgeDist::Float64=0.04,
            Min2ndDist::Float64=0.5,
            p_length::Float64=0.5,
            n_seeds::Int64=40,
            pmin::Float64=0.7,
            pmax::Float64=1.3)

    LCSParameters(radius, MaxWedgeDist, MinWedgeDist, Min2ndDist, p_length, n_seeds, pmin, pmax)
end

"""
    singularity_location_detection(T, xspan, yspan)

Detects tensor singularities of the tensor field `T`, given as a matrix of
`SymmetricTensor{2,2}`. `xspan` and `yspan` correspond to the uniform
grid vectors over which `T` is given. Returns a list of static 2-vectors.
"""
function singularity_location_detection(T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                                        xspan::AbstractVector{S},
                                        yspan::AbstractVector{S}) where S

    z1 = [c[1]-c[4] for c in T]
    z2 = [c[2] for c in T]
    zdiff = z1-z2
    # C = Contour.contours(xspan,yspan,zdiff,[0.])
    cl = Contour.levels(Contour.contours(yspan, xspan, zdiff, [0.]))[1]
    sitp = ITP.LinearInterpolation((xspan, yspan), permutedims(z1))
    # itp = ITP.interpolate(z1, ITP.BSpline(ITP.Linear()))
    # sitp = ITP.extrapolate(ITP.scale(itp, yspan, xspan), (ITP.Reflect(), ITP.Reflect(), ITP.Reflect()))
    Xs, Ys = Float64[], Float64[]
    for line in Contour.lines(cl)
        yL, xL = Contour.coordinates(line)
        zL = [sitp(xL[i], yL[i]) for i in eachindex(xL, yL)]
        ind = findall(zL[1:end-1] .* zL[2:end] .<= 0)
        zLind = -zL[ind] ./ (zL[ind .+ 1] - zL[ind])
        Xs = append!(Xs, xL[ind] + (xL[ind .+ 1] - xL[ind]) .* zLind)
        Ys = append!(Ys, yL[ind] + (yL[ind .+ 1] - yL[ind]) .* zLind)
    end
    return [SVector{2}(Xs[i], Ys[i]) for i in eachindex(Xs, Ys)]
end

"""
    singularity_type_detection(singularity, T, radius, xspan, yspan)

Determines the singularity type of the singularity candidate `singularity`
by querying the tensor eigenvector field of `T` in a circle of radius `radius`
around the singularity. `xspan` and `yspan` correspond to the computational grid.
Returns `1` for a trisector, `-1` for a wedge, and `0` otherwise.
"""
function singularity_type_detection(singularity::SVector{2,S}, ξ, radius::Real) where {S <: Real}

    Ntheta = 360   # number of points used to construct a circle around each singularity
    circle = [SVector{2,S}(radius*cos(t), radius*sin(t)) for t in range(-π, stop=π, length=Ntheta)]
    pnts = [singularity + c for c in circle]
    radVals = [ξ(p[2], p[1]) for p in pnts]
    singularity_type = 0
    if (sum(diff(radVals) .< 0) / Ntheta > 0.62)
        singularity_type = -1  # trisector
    elseif (sum(diff(radVals) .> 0) / Ntheta > 0.62)
        singularity_type = 1  # wedge
    end
    return singularity_type
end

"""
    detect_elliptic_region(singularities, singularityTypes, MaxWedgeDist, MinWedgeDist, Min2ndDist)

Determines candidate regions for closed tensor line orbits.
   * `singularities`: list of all singularities
   * `singularityTypes`: list of corresponding singularity types
   * `MaxWedgeDist`: maximum distance to closest wedge
   * `MinWedgeDist`: minimal distance to closest wedge
   * `Min2ndDist`: minimal distance to second closest wedge
Returns a list of vortex centers.
"""
function detect_elliptic_region(singularities::AbstractVector{SVector{2,S}},
                                singularity_types::AbstractVector{Int},
                                MaxWedgeDist::Float64,
                                MinWedgeDist::Float64,
                                Min2ndDist::Float64) where S <: Number

    indWedges = findall(singularity_types .== 1)
    wedgeDist = Distances.pairwise(Distances.Euclidean(), hcat(singularities[indWedges]...))
    idx = zeros(Int64, size(wedgeDist,1), 2)
    pairs = Vector{Int}[]
    for i=1:size(wedgeDist,1)
        idx = partialsortperm(wedgeDist[i,:], 2:3)
        if (wedgeDist[i,idx[1]] <= MaxWedgeDist &&
            wedgeDist[i,idx[1]] >= MinWedgeDist &&
            wedgeDist[i,idx[2]]>=Min2ndDist)
            push!(pairs,[i, idx[1]])
        end
    end
    pairind = unique(sort!.(intersect(pairs, reverse.(pairs, dims=1))))
    centers = [mean(singularities[indWedges[p]]) for p in pairind]
    return SVector{2}.(centers)
end

"""
    set_Poincaré_section(vc, p_length, n_seeds, xspan, yspan)

Generates a horizontal Poincaré section, centered at the vortex center `vc`
of length `p_length` consisting of `n_seeds` starting at `0.2*p_length`
eastwards. All points are guaranteed to lie in the computational domain given
by `xspan` and `yspan`.
"""
function set_Poincaré_section(vc::SVector{2,S},
                                p_length::Float64,
                                n_seeds::Int,
                                xspan::AbstractVector{S},
                                yspan::AbstractVector{S}) where S <: Real

    xmin, xmax = extrema(xspan)
    ymin, ymax = extrema(yspan)
    p_section::Vector{SVector{2,S}} = [vc]
    eₓ = SVector{2,S}(1., 0.)
    pspan = range(vc + .2p_length*eₓ, stop=vc + p_length*eₓ, length=n_seeds)
    idxs = [all(p .<= [xmax, ymax]) && all(p .>= [xmin, ymin]) for p in pspan]
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

function bisection(f, a::T, b::T, tol::Float64=1.e-4, maxiter::Integer=15) where T <: Real
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
    compute_outermost_closed_orbit(pSection, T, xspan, yspan; pmin = .7, pmax = 1.3)

Compute the outermost closed orbit for a given Poincaré section `pSection`,
tensor field `T`, where the total computational domain is spanned by `xspan`
and `yspan`. Keyword arguments `pmin` and `pmax` correspond to the range of
shift parameters in which closed orbits are sought.
"""
function compute_outermost_closed_orbit(pSection::Vector{SVector{2,S}},
                                        T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                                        xspan::AbstractVector{S},
                                        yspan::AbstractVector{S};
                                        pmin::Float64=.7,
                                        pmax::Float64=1.3) where S <: Real

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
    Tval = zeros(length(pSection) - 1)
    s = falses(length(pSection) - 1)
    orbits = Vector{Vector{Tuple{S,S}}}(undef, length(pSection) - 1)
    for i in eachindex(pSection[2:end])
        Tsol = zero(Float64)
        try
            Tsol = bisection(λ -> prd(λ, false, pSection[i+1]), pmin, pmax)
            orbit = compute_returning_orbit(ηfield(Tsol, false), pSection[i+1])
            closed = norm(orbit[1] - orbit[end]) <= 1e-2
            uniform = all([l1itp(p[1], p[2]) <= Tsol <= l2itp(p[1], p[2]) for p in orbit])
            # @show (closed, uniform)
            # @show length(orbit)
            if (closed && uniform)
                Tval[i] = Tsol
                orbits[i] = [p.data for p in orbit]
                s[i] = false
            end
        catch
        end
        if iszero(Tsol)
            try
                Tsol = bisection(λ -> prd(λ, true, pSection[i+1]), pmin, pmax)
                orbit = compute_returning_orbit(ηfield(Tsol, true), pSection[i+1])
                closed = norm(orbit[1] - orbit[end]) <= 1e-2
                uniform = all([l1itp(p[1], p[2]) <= Tsol <= l2itp(p[1], p[2]) for p in orbit])
                # @show (closed, uniform)
                # @show length(orbit)
                if (closed && uniform)
                    Tval[i] = Tsol
                    orbits[i] = [p.data for p in orbit]
                    s[i] = true
                end
            catch
            end
        end
    end
    outerInd = findlast(!iszero, Tval)
    if outerInd !== nothing
        return EllipticBarrier(orbits[outerInd], Tval[outerInd], s[outerInd])
    else
        return nothing
    end
end

"""
    ellipticLCS(T, xspan, yspan, p)

Computes elliptic LCSs as null-geodesics of the Lorentzian metric tensor
field given by shifted versions of `T` on the 2D computational grid spanned
by `xspan` and `yspan`. `p` is a tuple of the following parameters (in that order):
   * radius: radius in tensor singularity type detection,
   * MaxWdgeDist: maximal distance to nearest wedge-type singularity,
   * MinWedgeDist: minimal distance to nearest wedge-type singularity,
   * Min2ndDist: minimal distance to second-nearest wedge-type singularity,
   * p_length: length of Poincaré section,
   * n_seeds: number of seeding points along the Poincaré section,
Returns a list of tuples, each tuple containing
   * the parameter value λ in the η-formula,
   * the sign used in the η-formula,
   * the outermost closed orbit for the corresponding λ and sign.
"""
function ellipticLCS(T::AbstractMatrix{SymmetricTensor{2,2,S,3}},
                        xspan::AbstractVector{S},
                        yspan::AbstractVector{S},
                        p::LCSParameters) where S <: Real

    singularities = singularity_location_detection(T, xspan, yspan)
    println("Detected $(length(singularities)) singularity candidates...")

    ξ = [eigvecs(t)[:,1] for t in T]
    ξrad = atan.([v[2]./v[1] for v in ξ])
    ξraditp = ITP.LinearInterpolation((yspan, xspan), ξrad, extrapolation_bc = ITP.Line())
    # ξraditp = ITP.extrapolate(ITP.scale(ITP.interpolate(ξrad,
    #                     ITP.BSpline(ITP.Linear())),
    #                     yspan,xspan), ITP.Reflect())
    singularitytypes = map(singularities) do s
        singularity_type_detection(s, ξraditp, p.radius)
    end
    println("Determined $(sum(abs.(singularitytypes))) nondegenerate singularities...")

    vortexcenters = detect_elliptic_region(singularities, singularitytypes,
                                p.MaxWedgeDist, p.MinWedgeDist, p.Min2ndDist)
    println("Defined $(length(vortexcenters)) Poincaré sections...")

    p_section = map(vortexcenters) do vc
        set_Poincaré_section(vc, p.p_length, p.n_seeds, xspan, yspan)
    end

    closedorbits = pmap(p_section) do ps
        compute_outermost_closed_orbit(ps, T, xspan, yspan; pmin=p.pmin, pmax=p.pmax)
    end

    # closed orbits extraction
    vortices = Vector{EllipticBarrier}()
    for co in closedorbits
        if co !== nothing
            push!(vortices, co)
        end
    end
    println("Found $(length(vortices)) vortices.")
    return vortices
end
