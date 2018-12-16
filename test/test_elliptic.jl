using Test, StaticArrays, OrdinaryDiffEq, LinearAlgebra, CoherentStructures, Interpolations
import AxisArrays
const AA = AxisArrays
const CS = CoherentStructures

@testset "compute critical points" begin
    for x in (range(-1, stop=1, length=50), range(-1, stop=1, length=50)),
        y in (range(-1, stop=1, length=50), range(-1, stop=1, length=50))
        for v in (AA.AxisArray(SVector{2}.(x, y'), x, y),
                AA.AxisArray(SVector{2}.(-x, -y'), x, y),
                AA.AxisArray(SVector{2}.(-y', x), x, y))
            α = map(v -> atan(v[2], v[1]), v)
            S = @inferred compute_singularities(α, 2π)
            @test length(S) == 1
            @test iszero(S[1].coords)
            @test S[1].index == 1
            S = @inferred critical_point_detection(v, 0.1, 2π; combine_pairs=false)
            @test length(S) == 1
            @test iszero(S[1].coords)
            @test S[1].index == 1
        end
        v = AA.AxisArray(SVector{2}.(x, -y'), x, y)
        α = map(v -> atan(v[2], v[1]), v)
        S = @inferred compute_singularities(α, 2π)
        @test length(S) == 1
        @test iszero(S[1].coords)
        @test S[1].index == -1
        S = critical_point_detection(v, 0.1, 2π; combine_pairs=false)
        @test length(S) == 1
        @test iszero(S[1].coords)
        @test S[1].index == -1
    end
end

q = 3
tspan = range(0., stop=1., length=q)
ny = 52
nx = 51
xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
xspan = range(xmin, stop=xmax, length=nx)
yspan = range(ymin, stop=ymax, length=ny)
P = AA.AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
mCG_tensor = u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, 1.e-6)
T = @inferred map(mCG_tensor, P)

@testset "combine singularities" begin
    ξ = map(t -> eigvecs(t)[:,1], T)
    α = map(v -> atan(v[2], v[1]), ξ)
    singularities = @inferred compute_singularities(α, π)
    new_singularities = @inferred combine_singularities(singularities, 3*step(xspan))
    @inferred CoherentStructures.combine_isolated_pairs(new_singularities)
    r₁ , r₂ = 2rand(2)
    @test sum(get_indices(combine_singularities(singularities, r₁))) ==
        sum(get_indices(combine_singularities(singularities, r₂))) ==
        sum(get_indices(combine_singularities(singularities, 2)))
end

@testset "closed orbit detection" begin
    Ω = SMatrix{2,2}(0, -1, 1, 0)
    vf(λ) = OrdinaryDiffEq.ODEFunction((u, p, t) -> (Ω - (1 - λ) * I) * u)
    seed = SVector{2}(rand(), 0)
    d = @inferred CS.Poincaré_return_distance(vf(1), seed)
    @test d ≈ 0 atol = 1e-5
    λ⁰ = @inferred CS.bisection(λ -> CS.Poincaré_return_distance(vf(λ), seed), 0.7, 1.4, 1e-4)
    @test λ⁰ ≈ 1 rtol=1e-3
end

@testset "ellipticLCS" begin
    q = @inferred LCSParameters()
    @test q isa LCSParameters
    p = @inferred LCSParameters(3*max(step(xspan), step(yspan)), 0.5, true, 60, 0.7, 1.5, 1e-4)
    @test p isa LCSParameters
    cache = @inferred CS.orient(T, SVector{2}(0.25, 0.5))
    @test cache isa CS.LCScache
    vortices, singularities = ellipticLCS(T, p; outermost=true, verbose=false)
    @test length(vortices) == 2
    @test singularities isa Vector{Singularity{Float64}}
    @test length(singularities) > 5
    vortices, _ = ellipticLCS(T, p; outermost=false, verbose=false)
    @test length(vortices) > 20
end

@testset "constrainedLCS" begin
    Ω = SMatrix{2,2}(0, -1, 1, 0)
    Z = zeros(SVector{2})
    for (nx, ny) in ((50, 50), (51, 51), (50, 51), (51, 50)), combine in (true, false)
        xspan = range(-1, stop=1, length=nx)
        yspan = range(-1, stop=1, length=ny)
        P = AA.AxisArray(SVector{2}.(xspan, yspan'), xspan, yspan)
        q = map(p -> iszero(p) ? ones(typeof(p)) : (Ω + I) * normalize(p), P)
        p = @inferred LCSParameters(3*max(step(xspan), step(yspan)), 1.0, combine, 60, 0.5, 1.5, 1e-4)

        vortices, singularities = constrainedLCS(q, p; outermost=true, verbose=false)
        @test length(vortices) == 1
        @test singularities isa Vector{Singularity{Float64}}
        @test vortices[1].core ≈ Z atol=max(step(xspan), step(yspan))
        @test length(singularities) == 1
        @test singularities[1].coords ≈ Z atol=max(step(xspan), step(yspan))

        vortices, singularities = constrainedLCS(q, p; outermost=false, verbose=false)
        @test length(vortices) > 1
        @test singularities isa Vector{Singularity{Float64}}
        @test length(singularities) == 1
        @test singularities[1].coords ≈ Z atol=max(step(xspan), step(yspan))
    end
end
