# (c) 2018 Daniel Karrasch

using Contour, Distances#, Roots, Tensors #NLsolve, Interpolations,

function singularity_location_detection(Σ::AbstractMatrix,
                                        xspan::AbstractVector{T},
                                        yspan::AbstractVector{T}) where T

    z1 = [c[1]-c[4] for c in Σ]
    z2 = [c[2] for c in Σ]
    zdiff = z1-z2
    C = Contour.contours(xspan,yspan,zdiff,[0.])
    cl = levels(contours(xspan,yspan,zdiff,[0.]))[1]
    itp = Interpolations.interpolate(z1,Interpolations.BSpline(Interpolations.Linear()),Interpolations.OnGrid())
    sitp = Interpolations.scale(itp, xspan, yspan)
    Xs = Float64[]; Ys = Float64[]
    for line in lines(cl)
        xL, yL = coordinates(line)
        zL = [sitp[xL[i],yL[i]] for i in eachindex(xL,yL)]
        ind = find(zL[1:end-1].*zL[2:end].<=0.)
        Xs = append!(Xs,xL[ind]+( xL[ind+1]-xL[ind] ).*( 0-zL[ind] )./( zL[ind+1]-zL[ind] ));
        Ys = append!(Ys,yL[ind]+( yL[ind+1]-yL[ind] ).*( 0-zL[ind] )./( zL[ind+1]-zL[ind] ));
    end
    return [StaticArrays.SVector{2,T}(Xs[i], Ys[i]) for i in eachindex(Xs,Ys)]
end

function singularity_type_detection(singularity::StaticArrays.SVector{2,T},
                                    ξrad,
                                    radius::Float64) where T

    Ntheta = 360   # number of points used to construct a circle around each singularity
    circle = [StaticArrays.SVector{2,T}(radius*cos(t), radius*sin(t)) for t in linspace(-π,π,Ntheta)]
    pnts = [singularity+c for c in circle]
    radVals = [ξrad[p[1],p[2]] for p in pnts]
    SingularityType = 0
    if (sum(diff(radVals).<0)/Ntheta>0.62)
        SingularityType =  1  # trisector
    elseif (sum(diff(radVals).>0)/Ntheta>0.62)
        SingularityType =  -1  # wedge
    end
    return SingularityType
end

function detect_elliptic_region(singularities::Vector{StaticArrays.SVector{2,T}},
                                singularityTypes::Vector{Int},
                                MaxWedgeDist::Float64,
                                MinWedgeDist::Float64,
                                Min2ndDist::Float64) where T <: Number

    indWedges = find(singularityTypes.==-1.)
    wedgeDist = pairwise(Euclidean(0),hcat(singularities[singularityTypes.==-1.]...))
    idx = zeros(Int64,size(wedgeDist,1),2)
    pairs = Vector{Int64}[]
    for i=1:size(wedgeDist,1)
        idx = selectperm(wedgeDist[i,:],2:3)
        if (wedgeDist[i,idx[1]]<MaxWedgeDist && wedgeDist[i,idx[1]]>MinWedgeDist && wedgeDist[i,idx[2]]>Min2ndDist)
            push!(pairs,[i, idx[1]])
        end
    end
    pairind = indexin(pairs,flipdim.(pairs,1))
    vortexCenters = StaticArrays.SVector{2,T}[]
    for p in pairind
        if p!=0
            push!(vortexCenters,StaticArrays.SVector{2,T}(mean(singularities[indWedges[pairs[p]]])...))
        end
    end
    return unique(vortexCenters)
end

function set_Poincaré_section(vc::StaticArrays.SVector{2,T},
                                pLength::Float64,
                                numSeeds::Int) where T <: Real

    pSection = [vc]::Vector{StaticArrays.SVector{2,T}}
    ex = StaticArrays.SVector{2,T}(1., 0.)
    pspan = linspace(vc+.2*pLength*ex,vc+pLength*ex,numSeeds)
    append!(pSection,pspan)
    return pSection[all.([p.<=[xmax, ymax] for p in pSection]).*all.([p.>=[xmin, ymin] for p in pSection])]
end

function compute_returning_orbit(calT::Float64,
                                 seed::AbstractVector{T},
                                 λ₁::Matrix{T},
                                 λ₂::Matrix{T},
                                 ξ₁::Matrix{Tensors.Tensor{1,2,T,2}},
                                 ξ₂::Matrix{Tensors.Tensor{1,2,T,2}},
                                 s::Int,
                                 xspan::AbstractVector{T},
                                 yspan::AbstractVector{T}) where T <: Real

    # println(calT)
    # println(seed)
    # println(s)
    α = real.(sqrt.(Complex.((λ₂-calT)./(λ₂-λ₁))))
    β = real.(sqrt.(Complex.((calT-λ₁)./(λ₂-λ₁))))
    isposdef(s) ?
        η = α.*ξ₁ + β.*ξ₂ :
        η = α.*ξ₁ - β.*ξ₂
    η = [StaticArrays.SVector{2,T}(n[1],n[2]) for n in η]
    ηitp = Interpolations.interpolate(η,Interpolations.BSpline(Interpolations.Linear()),Interpolations.OnGrid())
    ηsitp = Interpolations.scale(ηitp,xspan,yspan)
    function ηfield(u,p,t)
        field = ηsitp[u[1],u[2]]
        du1 = field[1]
        du2 = field[2]
        return StaticArrays.SVector{2,T}(du1, du2)
    end
    prob = OrdinaryDiffEq.ODEProblem(ηfield,StaticArrays.SVector{2,T}(seed[1],seed[2]),(0.,20.))
    condition(u,t,integrator) = u[2]-seed[2]#+10*eps(seed[2])
    affect!(integrator) = OrdinaryDiffEq.terminate!(integrator)
    cb = OrdinaryDiffEq.ContinuousCallback(condition,nothing,affect!)
    sol = OrdinaryDiffEq.solve(prob,OrdinaryDiffEq.Tsit5(),maxiters=2e3,dense=false,reltol=1e-8,abstol=1e-8,callback=cb).u #,dtmin=1e-3
    # S = hcat(sol...)
    # plot!(S[1,:],S[2,:],color=[:blue],leg=false,xlims=(-4., 6.), ylims=(-34., -28.))
    # if sol[end][2]-seed[2]≈zero(eltype(seed))
    #     return sol
    # else
    #     return seed+ones(seed)
    # end
end

# function Poincaré_return_distance!(calT::Float64,seed::Vector{Float64},
#     ξ₁::Matrix{Tensors.Tensor{1,2,Float64,2}},ξ₂::Matrix{Tensors.Tensor{1,2,Float64,2}},
#     signum::Int,out)
#
#     sol = compute_returning_orbit(calT,seed,ξ₁,ξ₂,signum)
#     out[1] = sol[end][1]-seed[1]
# end

function Poincaré_return_distance(calT::Float64,
                                    seed::AbstractVector{T},
                                    λ₁::Matrix{T},
                                    λ₂::Matrix{T},
                                    ξ₁::Matrix{Tensors.Tensor{1,2,T,2}},
                                    ξ₂::Matrix{Tensors.Tensor{1,2,T,2}},
                                    signum::Int,
                                    xspan::AbstractVector{T},
                                    yspan::AbstractVector{T}) where T <: Real

    # @show calT
    sol = compute_returning_orbit(calT,seed,λ₁,λ₂,ξ₁,ξ₂,signum,xspan,yspan)
    # @show sol
    if abs(sol[end][2]-seed[2])<=1e-2
        return sol[end][1]-seed[1]
    else
        return NaN # one(seed[1])
    end
end

function bisection(f, a::T, b::T, tol::Float64=1.e-4, maxiter::Integer=15) where T <: Real
    fa = f(a)
    fb = f(b)
    # @show a, fa, b, fb
    fa*fb <= 0 || error("No real root in [a,b]")
    i = 0
    local c
    while b-a > tol
        i += 1
        i != maxiter || error("Max iteration exceeded")
        c = (a+b)/2 # bisection
        # c = (a*fb-b*fa)/(fb-fa) # regula falsi
        fc = f(c)
        if abs(fc) < tol
            break
        elseif fa*fc > 0
            a = c  # Root is in the right half of [a,b].
            fa = fc
        else
            b = c  # Root is in the left half of [a,b].
        end
    end
    return c
end

function compute_outermost_closed_orbit(pSection::Vector{StaticArrays.SVector{2,T}},
                                        λ₁::Matrix{T},
                                        λ₂::Matrix{T},
                                        ξ₁::Matrix{Tensors.Tensor{1,2,T,2}},
                                        ξ₂::Matrix{Tensors.Tensor{1,2,T,2}},
                                        l1itp::Interpolations.ScaledInterpolation,
                                        l2itp::Interpolations.ScaledInterpolation,
                                        xspan::AbstractVector{T},
                                        yspan::AbstractVector{T};
                                        calTmin::Float64 = .7,
                                        calTmax::Float64 = 1.3) where T <: Real

    # for computational tractability, pre-orient the eigenvector fields
    const Ω = Tensors.Tensor{2,2}([0.,-1.,1.,0.])
    relP = [Tensors.Tensor{1,2}(p .- pSection[1]) for p in P]
    n = [Ω⋅dx for dx in relP]
    ξ₁ = sign.(n.⋅ξ₁).*ξ₁
    ξ₂ = sign.(relP.⋅ξ₂).*ξ₂

    # go along the Poincaré section and solve for T
    # first, define a nonlinear root finding problem
    Tval = zeros(length(pSection)-1)
    s = zeros(Int,length(pSection)-1)
    orbits = Vector{Vector{Vector{Float64}}}(length(pSection)-1)
    for i in eachindex(pSection[2:end])
        # println(i)
        Tsol = zero(Float64)
        prd(calT,signum) = Poincaré_return_distance(calT,pSection[i+1],λ₁,λ₂,ξ₁,ξ₂,signum,xspan,yspan)
        prd_plus(calT) = prd(calT,1)
        prd_minus(calT) = prd(calT,-1)
        try
            Tsol = bisection(prd_plus,calTmin,calTmax)
            # @show Tsol
            # Tsol = fzero(prd_plus,calTmin,calTmax,abstol=5e-3,reltol=1e-4)
            # Tsol = fzero(prd_plus,calTmin,calTmax,order=2,abstol=5e-3,reltol=1e-4)
            # Tsol = fzero(prd_plus,1.,[calTmin,calTmax])
            # Tsol = find_zero(prd_plus,1.,Order2(),bracket=[calTmin,calTmax],abstol=5e-3,reltol=1e-4)
            orbit = compute_returning_orbit(Tsol,pSection[i+1],λ₁,λ₂,ξ₁,ξ₂,1,xspan,yspan)
            closed = norm(orbit[1]-orbit[end])<=1e-2
            uniform = all([l1itp[p[1],p[2]]-sqrt(eps(1.)) .<= Tsol .<= l2itp[p[1],p[2]]+sqrt(eps(1.)) for p in orbit])
            @show (closed, uniform)
            if (closed && uniform)
                Tval[i] = Tsol
                orbits[i] = orbit
                s[i] = 1
            end
        catch
        end
        if iszero(Tsol)
            try
                Tsol = bisection(prd_minus,calTmin,calTmax,1.e-4,15)
                # @show Tsol
                # Tsol = fzero(prd_minus,calTmin,calTmax,abstol=5e-3,reltol=1e-4)
                # Tsol = fzero(prd_minus,calTmin,calTmax,order=2,abstol=5e-3,reltol=1e-4)
                # Tsol = fzero(prd_minus,1.,[calTmin,calTmax])
                # Tsol = find_zero(prd_minus,1.,Order2(),bracket=[calTmin,calTmax],abstol=5e-3,reltol=1e-4)
                # Tsol = fzeros(prd_minus,calTmin,calTmax,no_pts=30,abstol=5e-3,reltol=1e-4)[1]
                orbit = compute_returning_orbit(Tsol,pSection[i+1],λ₁,λ₂,ξ₁,ξ₂,-1,xspan,yspan)
                closed = norm(orbit[1]-orbit[end])<=1e-2
                uniform = all([l1itp[p[1],p[2]]-sqrt(eps(1.)) .<= Tsol .<= l2itp[p[1],p[2]]+sqrt(eps(1.)) for p in orbit])
                @show (closed, uniform)
                if (closed && uniform)
                    Tval[i] = Tsol
                    orbits[i] = orbit
                    s[i] = -1
                    # [plot(o[1],o[2],".b") for o in orbit]
                # else
                #     display("no good orbit")
                #     O = hcat(orbit...)
                #     plot!(O[1,:],O[2,:],color=[:blue],leg=false)
                end
            catch
            end
        end
    end
    outerInd = findlast(Tval)
    if outerInd>0
        return Tval[outerInd], s[outerInd], orbits[outerInd]
    else
        return nothing
    end
end

function ellipticLCS(T::Matrix{Tensors.SymmetricTensor{2,2,S,3}},
                        λ₁::Matrix{S},
                        λ₂::Matrix{S},
                        ξ₁::Matrix{Tensors.Tensor{1,2,S,2}},
                        ξ₂::Matrix{Tensors.Tensor{1,2,S,2}},
                        l1itp::Interpolations.ScaledInterpolation,
                        l2itp::Interpolations.ScaledInterpolation,
                        ξraditp::Interpolations.ScaledInterpolation,
                        xspan::AbstractVector{S},
                        yspan::AbstractVector{S},
                        p) where S <: Real

    # unpack parameters
    radius, MaxWedgeDist, MinWedgeDist, Min2ndDist, pLength, numSeeds = p

    singularities = singularity_location_detection(T,xspan,yspan)
    singularityTypes = map(s->singularity_type_detection(s,ξraditp,radius),singularities)
    vortexCenters = detect_elliptic_region(singularities,singularityTypes,MaxWedgeDist,MinWedgeDist,Min2ndDist)
    Psection = map(vc -> set_Poincaré_section(vc,pLength,numSeeds),vortexCenters)
    @everywhere coco(ps) = compute_outermost_closed_orbit(ps,λ₁,λ₂,ξ₁,ξ₂,l1itp,l2itp,xspan,yspan)
    @time OCC = pmap(coco,Psection)
    return OCC
end
