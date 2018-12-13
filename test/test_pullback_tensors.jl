using Test, StaticArrays, OrdinaryDiffEq, Tensors, LinearAlgebra, Distributed
using CoherentStructures

@testset "flow" begin
    for dim in (2, 3), q in (2, 5, 11)
        tspan = range(0., stop=1., length=q)

        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xv = Vec{dim}(x0)

        voop = ODEFunction((u, p, t) -> zero(SVector{dim}))
        @test fill(xs, q) == @inferred flow(voop, xs, tspan)
        @test fill(xs, q) == @inferred flow(voop, xv, tspan)
        @test fill(xs, q) == flow(voop, x0, tspan)

        viip = ODEFunction((du, u, p, t) -> du .= zeros(dim))
        @test fill(x0, q) == @inferred flow(viip, x0, tspan)
        @test fill(x0, q) == @inferred flow(viip, xs, tspan)
        @test fill(x0, q) == @inferred flow(viip, xv, tspan)

        One = SVector{dim}(ones(dim))
        voop = ODEFunction((u, p, t) -> One)
        @test xs == flow(voop, xs, tspan)[1]
        @test xs + (tspan[end] - tspan[1]) * One ≈ flow(voop, x0, tspan)[end]
        @test xs + (tspan[end] - tspan[1]) * One ≈ flow(voop, xs, tspan)[end]
        @test xs + (tspan[end] - tspan[1]) * One ≈ flow(voop, xv, tspan)[end]

        f(u) = flow(voop, u, tspan)
        P = [SVector{dim}(rand(dim)) for i=1:10]
        sol(p) = vcat([Vector(p)] .+ (tspan .* [Vector(One)])...)
        F = hcat((sol.(P))...)
        @test F ≈ @inferred parallel_flow(f, P)
    end
end

@testset "linearized flow" begin
    for dim in (2, 3), q in (2, 5, 11), rhs in (zero(SVector{dim}), SVector{dim}(ones(dim)))
        tspan = range(0., stop=1., length=q)
        Id = one(Tensor{2,dim})
        Idspan = fill(Id, length(tspan))

        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xv = Vec{dim}(x0)

        voop = ODEFunction((u, p, t) -> rhs)
        @test Idspan ≈ linearized_flow(voop, x0, tspan, 0.1)[2]
        @test Idspan ≈ @inferred(linearized_flow(voop, xs, tspan, 0.1))[2]
        @test Idspan ≈ @inferred(linearized_flow(voop, xv, tspan, 0.1))[2]

        viip = ODEFunction((du, u, p, t) -> du .= rhs)
        @test Idspan ≈ linearized_flow(viip, x0, tspan, 0.1)[2]
        @test Idspan ≈ @inferred(linearized_flow(viip, xs, tspan, 0.1))[2]
        @test Idspan ≈ @inferred(linearized_flow(viip, xv, tspan, 0.1))[2]
    end
end

@testset "pullback tensors" begin
    # check constant vector fields
    for dim in (2, 3), rhs in (zero(SVector{dim}), SVector{dim}(ones(dim))), q in (2, 5, 11)
        tspan = range(0., stop=1., length=q)
        for pb in (mean_diff_tensor, CG_tensor)
            x0 = rand(dim)
            xs = SVector{dim}(x0)
            xv = Vec{dim}(x0)
            Id = one(SymmetricTensor{2,dim})
            voop = ODEFunction((u, p, t) -> rhs)
            viip = ODEFunction((du, u, p, t) -> du .= rhs)

            @test Id ≈ pb(voop, x0, tspan, 0.1)
            @test Id ≈ pb(viip, x0, tspan, 0.1)
            # next paragraph can be deleted once inference is fixed
            @test Id ≈ pb(voop, x0, tspan, 0.1)
            @test Id ≈ pb(viip, x0, tspan, 0.1)

            @test Id ≈ @inferred pb(voop, xs, tspan, 0.1)
            @test Id ≈ @inferred pb(viip, xs, tspan, 0.1)
            @test Id ≈ @inferred pb(voop, xv, tspan, 0.1)
            @test Id ≈ @inferred pb(viip, xv, tspan, 0.1)
        end

        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xv = Vec{dim}(x0)
        voop = ODEFunction((u, p, t) -> rhs)
        viip = ODEFunction((du, u, p, t) -> du .= rhs)
        S = rand(SymmetricTensor{2,dim})
        B = rand(Tensor{2,dim})
        Sspan = fill(S, length(tspan))

        @test det(S)*inv(S) ≈ av_weighted_CG_tensor(voop, x0, tspan, 0.1; D=S)
        @test det(S)*inv(S) ≈ av_weighted_CG_tensor(viip, x0, tspan, 0.1; D=S)
        @test Sspan ≈ pullback_diffusion_tensor(voop, x0, tspan, 0.1; D=S)
        @test Sspan ≈ pullback_diffusion_tensor(viip, x0, tspan, 0.1; D=S)
        @test Sspan ≈ pullback_metric_tensor(voop, x0, tspan, 0.1; G=S)
        @test Sspan ≈ pullback_metric_tensor(viip, x0, tspan, 0.1; G=S)
        @test fill(B, length(tspan)) ≈ pullback_SDE_diffusion_tensor(voop, x0, tspan, 0.1; B=B)
        @test fill(B, length(tspan)) ≈ pullback_SDE_diffusion_tensor(viip, x0, tspan, 0.1; B=B)
        # next paragraph can be deleted once inference is fixed
        @test det(S)*inv(S) ≈ av_weighted_CG_tensor(voop, x0, tspan, 0.1; D=S)
        @test det(S)*inv(S) ≈ av_weighted_CG_tensor(viip, x0, tspan, 0.1; D=S)
        @test Sspan ≈ pullback_diffusion_tensor(voop, x0, tspan, 0.1; D=S)
        @test Sspan ≈ pullback_diffusion_tensor(viip, x0, tspan, 0.1; D=S)
        @test Sspan ≈ pullback_metric_tensor(voop, x0, tspan, 0.1; G=S)
        @test Sspan ≈ pullback_metric_tensor(viip, x0, tspan, 0.1; G=S)
        @test fill(B, length(tspan)) ≈ pullback_SDE_diffusion_tensor(voop, x0, tspan, 0.1; B=B)
        @test fill(B, length(tspan)) ≈ pullback_SDE_diffusion_tensor(viip, x0, tspan, 0.1; B=B)

        @test det(S)*inv(S) ≈ @inferred(av_weighted_CG_tensor(voop, xs, tspan, 0.1; D=S))
        @test det(S)*inv(S) ≈ @inferred(av_weighted_CG_tensor(viip, xs, tspan, 0.1; D=S))
        @test Sspan ≈ @inferred(pullback_diffusion_tensor(voop, xs, tspan, 0.1; D=S))
        @test Sspan ≈ @inferred(pullback_diffusion_tensor(viip, xs, tspan, 0.1; D=S))
        @test Sspan ≈ @inferred(pullback_metric_tensor(voop, xs, tspan, 0.1; G=S))
        @test Sspan ≈ @inferred(pullback_metric_tensor(viip, xs, tspan, 0.1; G=S))
        @test fill(B, length(tspan)) ≈ @inferred(pullback_SDE_diffusion_tensor(voop, xs, tspan, 0.1; B=B))
        @test fill(B, length(tspan)) ≈ @inferred(pullback_SDE_diffusion_tensor(viip, xs, tspan, 0.1; B=B))

        @test det(S)*inv(S) ≈ @inferred(av_weighted_CG_tensor(voop, xv, tspan, 0.1; D=S))
        @test det(S)*inv(S) ≈ @inferred(av_weighted_CG_tensor(viip, xv, tspan, 0.1; D=S))
        @test Sspan ≈ @inferred(pullback_diffusion_tensor(voop, xv, tspan, 0.1; D=S))
        @test Sspan ≈ @inferred(pullback_diffusion_tensor(viip, xv, tspan, 0.1; D=S))
        @test Sspan ≈ @inferred(pullback_metric_tensor(voop, xv, tspan, 0.1; G=S))
        @test Sspan ≈ @inferred(pullback_metric_tensor(viip, xv, tspan, 0.1; G=S))
        @test fill(B, length(tspan)) ≈ @inferred(pullback_SDE_diffusion_tensor(voop, xv, tspan, 0.1; B=B))
        @test fill(B, length(tspan)) ≈ @inferred(pullback_SDE_diffusion_tensor(viip, xv, tspan, 0.1; B=B))
    end
    # check linear vector field
    for dim in (2, 3)
        tspan = range(0., stop=1., length=2)
        sI = one(SMatrix{dim,dim})
        mI = Matrix(I, dim, dim)
        voop = ODEFunction((u, p, t) -> u)
        viip = ODEFunction((du, u, p, t) -> du .= u)
        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xv = Vec{dim}(x0)
        Id = one(SymmetricTensor{2,dim})

        # test CG_tensor
        CG = exp(sI)'exp(sI)
        @test CG ≈ CG_tensor(voop, x0, tspan, 1e-1) rtol=1e-5
        @test CG ≈ CG_tensor(viip, x0, tspan, 1e-1) rtol=1e-5
        # next paragraph can be deleted once inference is fixed
        @test CG ≈ CG_tensor(voop, x0, tspan, 1e-1) rtol=1e-5
        @test CG ≈ CG_tensor(viip, x0, tspan, 1e-1) rtol=1e-5

        @test CG ≈ @inferred(CG_tensor(voop, xs, tspan, 1e-1)) rtol=1e-5
        @test CG ≈ @inferred(CG_tensor(viip, xs, tspan, 1e-1)) rtol=1e-5
        @test CG ≈ @inferred(CG_tensor(voop, xv, tspan, 1e-1)) rtol=1e-5
        @test CG ≈ @inferred(CG_tensor(viip, xv, tspan, 1e-1)) rtol=1e-5

        # test mean_diff_tensor
        D̅ = 1//2 * (sI + inv(CG))
        @test D̅ ≈ mean_diff_tensor(voop, x0, tspan, 1e-1) rtol=1e-5
        @test D̅ ≈ mean_diff_tensor(viip, x0, tspan, 1e-1) rtol=1e-5
        # next paragraph can be deleted once inference is fixed
        @test D̅ ≈ mean_diff_tensor(voop, x0, tspan, 1e-1) rtol=1e-5
        @test D̅ ≈ mean_diff_tensor(viip, x0, tspan, 1e-1) rtol=1e-5

        @test D̅ ≈ @inferred(mean_diff_tensor(voop, xs, tspan, 1e-1)) rtol=1e-5
        @test D̅ ≈ @inferred(mean_diff_tensor(viip, xs, tspan, 1e-1)) rtol=1e-5
        @test D̅ ≈ @inferred(mean_diff_tensor(voop, xv, tspan, 1e-1)) rtol=1e-5
        @test D̅ ≈ @inferred(mean_diff_tensor(viip, xv, tspan, 1e-1)) rtol=1e-5
    end
end
