# (c) 2018 Daniel Karrasch

const ITP = Interpolations

"""
    singularity_location_detection(T,xspan,yspan)

Detects tensor singularities of the tensor field `T`, given as a matrix of
`SymmetricTensor{2,2}`. `xspan` and `yspan` correspond to the uniform
grid vectors over which `T` is given. Returns a list of static 2-vectors.
"""
function singularity_location_detection(T::AbstractMatrix{Tensors.SymmetricTensor{2,2,S,3}},
                                        xspan::AbstractVector{S},
                                        yspan::AbstractVector{S}) where S

    z1 = [c[1]-c[4] for c in T]
    z2 = [c[2] for c in T]
    zdiff = z1-z2
    # C = Contour.contours(xspan,yspan,zdiff,[0.])
    cl = Contour.levels(Contour.contours(yspan, xspan, zdiff, [0.]))[1]
    itp = ITP.interpolate(z1, ITP.BSpline(ITP.Linear()), ITP.OnGrid())
    sitp = ITP.scale(itp, yspan, xspan)
    Xs, Ys = Float64[], Float64[]
    for line in Contour.lines(cl)
        yL, xL = Contour.coordinates(line)
        zL = [sitp[yL[i], xL[i]] for i in eachindex(xL, yL)]
        ind = findall(zL[1:end-1] .* zL[2:end].<=0.)
        zLind = -zL[ind] ./ (zL[ind .+ 1] - zL[ind])
        Xs = append!(Xs, xL[ind] + (xL[ind .+ 1] - xL[ind]) .* zLind)
        Ys = append!(Ys, yL[ind] + (yL[ind .+ 1] - yL[ind]) .* zLind)
    end
    return [[Xs[i], Ys[i]] for i in eachindex(Xs,Ys)]
end

"""
    singularity_type_detection(singularity,T,radius,xspan,yspan)

Determines the singularity type of the singularity candidate `singularity`
by querying the tensor eigenvector field of `T` in a circle of radius `radius`
around the singularity. `xspan` and `yspan` correspond to the computational grid.
Returns `1` for a trisector, `-1` for a wedge, and `0` otherwise.
"""
function singularity_type_detection(singularity::AbstractVector{S},
                                    ξ::ITP.ScaledInterpolation,
                                    radius::Float64) where S

    Ntheta = 360   # number of points used to construct a circle around each singularity
    circle = [StaticArrays.SVector{2,S}(radius*cos(t), radius*sin(t)) for t in range(-π, stop=π, length=Ntheta)]
    pnts = [singularity + c for c in circle]
    radVals = [ξ[p[2], p[1]] for p in pnts]
    singularity_type = 0
    if (sum(diff(radVals) .< 0) / Ntheta > 0.62)
        singularity_type = -1  # trisector
    elseif (sum(diff(radVals) .> 0) / Ntheta > 0.62)
        singularity_type = 1  # wedge
    end
    return singularity_type
end

"""
    detect_elliptic_region(singularities,singularityTypes,MaxWedgeDist,MinWedgeDist,Min2ndDist)

Determines candidate regions for closed tensor line orbits.
   * `singularities`: list of all singularities
   * `singularityTypes`: list of corresponding singularity types
   * `MaxWedgeDist`: maximum distance to closest wedge
   * `MinWedgeDist`: minimal distance to closest wedge
   * `Min2ndDist`: minimal distance to second closest wedge
Returns a list of vortex centers.
"""
function detect_elliptic_region(singularities::AbstractVector{Vector{S}},
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
    return [Statistics.mean(singularities[indWedges[p]]) for p in pairind]
end

"""
    set_Poincaré_section(vc,p_length,n_seeds,xspan,yspan)

Generates a horizontal Poincaré section, centered at the vortex center `vc`
of length `p_length` consisting of `n_seeds` starting at `0.2*p_length`
eastwards. All points are guaranteed to lie in the computational domain given
by `xspan` and `yspan`.
"""
function set_Poincaré_section(vc::AbstractVector{S},
                                p_length::Float64,
                                n_seeds::Int,
                                xspan::AbstractVector{S},
                                yspan::AbstractVector{S}) where S <: Real

    xmin, xmax = extrema(xspan)
    ymin, ymax = extrema(yspan)
    p_section::Vector{Vector{S}} = [vc]
    eₓ = [1., 0.]
    pspan = range(vc + .2p_length*eₓ, stop=vc + p_length*eₓ, length=n_seeds)
    idxs = [all(p .<= [xmax, ymax]) && all(p .>= [xmin, ymin]) for p in pspan]
    append!(p_section, pspan[idxs])
    return p_section
end

function compute_returning_orbit(calT::Float64,
                                 seed::AbstractVector{T},
                                 λ₁::AbstractMatrix{T},
                                 λ₂::AbstractMatrix{T},
                                 ξ₁::AbstractMatrix{Tensors.Tensor{1,2,T,2}},
                                 ξ₂::AbstractMatrix{Tensors.Tensor{1,2,T,2}},
                                 s::Int,
                                 xspan::AbstractVector{T},
                                 yspan::AbstractVector{T}) where T <: Real

    Δλ = λ₂ - λ₁
    α = real.(sqrt.(Complex.((λ₂ .- calT) ./ Δλ)))
    β = real.(sqrt.(Complex.((calT .- λ₁) ./ Δλ)))
    η = isposdef(s) ? α .* ξ₁ + β .* ξ₂ : α .* ξ₁ - β .* ξ₂
    η = [StaticArrays.SVector{2,T}(n[1],n[2]) for n in η]
    ηitp = ITP.scale(ITP.interpolate(η, ITP.BSpline(ITP.Cubic(ITP.Natural())), ITP.OnGrid()),
                        yspan, xspan)
    ηfield = (u,p,t) -> ηitp[u[2], u[1]]

    prob = OrdinaryDiffEq.ODEProblem(ηfield, StaticArrays.SVector{2}(seed[1], seed[2]), (0.,20.))
    condition(u,t,integrator) = u[2] - seed[2]
    affect!(integrator) = OrdinaryDiffEq.terminate!(integrator)
    cb = OrdinaryDiffEq.ContinuousCallback(condition, nothing, affect!)
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(), maxiters=2e3,
            dense=false, reltol=1e-8, abstol=1e-8, callback=cb, verbose=false).u
end

function Poincaré_return_distance(calT::Float64,
                                    seed::AbstractVector{T},
                                    λ₁::AbstractMatrix{T},
                                    λ₂::AbstractMatrix{T},
                                    ξ₁::AbstractMatrix{Tensors.Tensor{1,2,T,2}},
                                    ξ₂::AbstractMatrix{Tensors.Tensor{1,2,T,2}},
                                    signum::Int,
                                    xspan::AbstractVector{T},
                                    yspan::AbstractVector{T}) where T <: Real

    sol = compute_returning_orbit(calT,seed,λ₁,λ₂,ξ₁,ξ₂,signum,xspan,yspan)
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
function compute_outermost_closed_orbit(pSection::Vector{Vector{S}},
                                        T::AbstractMatrix{Tensors.SymmetricTensor{2,2,S,3}},
                                        xspan::AbstractVector{S},
                                        yspan::AbstractVector{S};
                                        pmin::Float64 = .7,
                                        pmax::Float64 = 1.3) where S <: Real

    λ₁, λ₂, ξ₁, ξ₂, _, _ = tensor_invariants(T)
    l1itp = ITP.scale(ITP.interpolate(λ₁, ITP.BSpline(ITP.Linear()), ITP.OnGrid()),
                        yspan,xspan)
    l2itp = ITP.scale(ITP.interpolate(λ₂, ITP.BSpline(ITP.Linear()), ITP.OnGrid()),
                        yspan,xspan)

    # for computational tractability, pre-orient the eigenvector fields
    Ω = Tensors.Tensor{2,2}([0., -1., 1., 0.])
    relP = [Tensors.Tensor{1,2}([x, y] .- pSection[1]) for y in yspan, x in xspan]
    n = [Ω⋅dx for dx in relP]
    ξ₁ .= sign.(n .⋅ ξ₁) .* ξ₁
    ξ₂ .= sign.(relP .⋅ ξ₂) .* ξ₂

    # go along the Poincaré section and solve for T
    # first, define a nonlinear root finding problem
    Tval = zeros(length(pSection)-1)
    s = zeros(Int,length(pSection)-1)
    orbits = Vector{Vector{Vector{Float64}}}(undef, length(pSection)-1)
    for i in eachindex(pSection[2:end])
        # println(i)
        Tsol = zero(Float64)
        prd(calT,signum) = Poincaré_return_distance(calT,pSection[i+1],λ₁,λ₂,ξ₁,ξ₂,signum,xspan,yspan)
        prd_plus(calT) = prd(calT,1)
        prd_minus(calT) = prd(calT,-1)
        try
            Tsol = bisection(prd_plus, pmin, pmax)
            # @show Tsol
            # Tsol = fzero(prd_plus,pmin,pmax,abstol=5e-3,reltol=1e-4)
            # Tsol = fzero(prd_plus,pmin,pmax,order=2,abstol=5e-3,reltol=1e-4)
            # Tsol = fzero(prd_plus,1.,[pmin,pmax])
            # Tsol = find_zero(prd_plus,1.,Order2(),bracket=[pmin,pmax],abstol=5e-3,reltol=1e-4)
            orbit = compute_returning_orbit(Tsol, pSection[i+1], λ₁, λ₂, ξ₁, ξ₂, 1, xspan, yspan)
            closed = norm(orbit[1] - orbit[end]) <= 1e-2
            uniform = all([l1itp[p[2], p[1]] .<= Tsol .<= l2itp[p[2], p[1]] for p in orbit])
            # @show (closed, uniform)
            if (closed && uniform)
                Tval[i] = Tsol
                orbits[i] = orbit
                s[i] = 1
            end
        catch
        end
        if iszero(Tsol)
            try
                Tsol = bisection(prd_minus, pmin, pmax)
                # @show Tsol
                # Tsol = fzero(prd_plus,pmin,pmax,abstol=5e-3,reltol=1e-4)
                # Tsol = fzero(prd_plus,pmin,pmax,order=2,abstol=5e-3,reltol=1e-4)
                # Tsol = fzero(prd_plus,1.,[pmin,pmax])
                # Tsol = find_zero(prd_plus,1.,Order2(),bracket=[pmin,pmax],abstol=5e-3,reltol=1e-4)
                orbit = compute_returning_orbit(Tsol, pSection[i+1], λ₁, λ₂, ξ₁, ξ₂, -1, xspan, yspan)
                closed = norm(orbit[1] - orbit[end]) <= 1e-2
                uniform = all([l1itp[p[2], p[1]] .<= Tsol .<= l2itp[p[2], p[1]] for p in orbit])
                # @show (closed, uniform)
                if (closed && uniform)
                    Tval[i] = Tsol
                    orbits[i] = orbit
                    s[i] = -1
                end
            catch
            end
        end
    end
    outerInd = findlast(!iszero, Tval)
    if outerInd != nothing
        return Tval[outerInd], s[outerInd], orbits[outerInd]
    else
        return nothing
    end
end

"""
    ellipticLCS(T,xspan,yspan,p)

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
function ellipticLCS(T::AbstractMatrix{Tensors.SymmetricTensor{2,2,S,3}},
                        xspan::AbstractVector{S},
                        yspan::AbstractVector{S},
                        p) where S <: Real

    # unpack parameters
    radius, MaxWedgeDist, MinWedgeDist, Min2ndDist, p_length, n_seeds = p

    singularities = singularity_location_detection(T, xspan, yspan)
    println("Detected $(length(singularities)) singularity candidates...")

    ξ = [eigvecs(t)[:,1] for t in T]
    ξrad = atan.([v[2]./v[1] for v in ξ])
    ξraditp = ITP.scale(ITP.interpolate(ξrad,
                        ITP.BSpline(ITP.Linear()), ITP.OnGrid()),
                        yspan,xspan)
    singularitytypes = map(singularities) do s
        singularity_type_detection(s, ξraditp, radius)
    end
    println("Determined $(sum(abs.(singularitytypes))) nondegenerate singularities...")

    vortexcenters = detect_elliptic_region(singularities, singularitytypes, MaxWedgeDist, MinWedgeDist, Min2ndDist)
    println("Defined $(length(vortexcenters)) Poincaré sections...")

    p_section = map(vortexcenters) do vc
        set_Poincaré_section(vc, p_length, n_seeds, xspan, yspan)
    end

    Distributed.@everywhere p_section = $p_section
    closedorbits = pmap(p_section) do ps
        compute_outermost_closed_orbit(ps, T, xspan, yspan)
    end

    # closed orbits extraction
    calTs = Vector{Float64}()
    signs = Vector{Int}()
    orbits = Vector{Array{Float64,2}}()
    for co in closedorbits
        try
            push!(calTs,co[1])
            push!(signs,co[2])
            push!(orbits,hcat(co[3]...))
        catch
        end
    end
    println("Found $(length(signs)) vortices.")
    return calTs, signs, orbits
end
