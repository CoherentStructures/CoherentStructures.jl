using Test, StaticArrays, OrdinaryDiffEq, Tensors, LinearAlgebra
using CoherentStructures

@testset "flow" begin
    for dim in (2, 3), q in (2, 5, 11)
        tspan = range(0., stop=1., length=q)

        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xt = (x0...,)
        xv = Vec{dim}(x0)

        voop = ODEFunction{false}((u, p, t) -> zero(SVector{dim}))
        viip = ODEFunction{true}((du, u, p, t) -> du .= 0)

        for x in (xs, xt, xv, x0)
            @test fill(xs, q) == (x !== x0 ? @inferred(flow(voop, x, tspan)) : flow(voop, x, tspan))
            if x !== xt
                @test fill(x0, q) == @inferred(flow(viip, x, tspan))
            end
        end

        One = SVector{dim}(ones(dim))
        voop = ODEFunction((u, p, t) -> One)
        @test xs == flow(voop, xs, tspan)[1]
        @test xs + (tspan[end] - tspan[1]) * One ≈ flow(voop, x0, tspan)[end]
        @test xs + (tspan[end] - tspan[1]) * One ≈ flow(voop, xs, tspan)[end]
        @test xs + (tspan[end] - tspan[1]) * One ≈ flow(voop, xv, tspan)[end]
        @test length(flow(voop, xs, tspan; saveat = [0.5])) == 1
    end
end

@testset "linearized flow" begin
    for dim in (2, 3), q in (2, 5, 11), rhs in (zero(SVector{dim}), SVector{dim}(ones(dim)))
        tspan = range(0., stop=1., length=q)
        Id = fill(one(Tensor{2,dim}), length(tspan))

        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xt = (x0...,)
        xv = Vec{dim}(x0)

        voop = ODEFunction((u, p, t) -> rhs)
        viip = ODEFunction((du, u, p, t) -> du .= rhs)
        for v in (voop, viip)
            @test Id ≈ linearized_flow(v, x0, tspan, 0.1)[2]
            @test Id ≈ @inferred(linearized_flow(v, xs, tspan, 0.1))[2]
            @test Id ≈ @inferred(linearized_flow(v, xt, tspan, 0.1))[2]
            @test Id ≈ @inferred(linearized_flow(v, xv, tspan, 0.1))[2]
        end
    end
end

@testset "pullback tensors" begin
    # check constant vector fields
    for dim in (2, 3), rhs in (zero(SVector{dim}), SVector{dim}(ones(dim))), q in (2, 5, 11)
        tspan = range(0., stop=1., length=q)
        for pb in (mean_diff_tensor, CG_tensor)
            x0 = rand(dim)
            xs = SVector{dim}(x0)
            xt = (x0...,)
            xv = Vec{dim}(x0)
            Id = one(SymmetricTensor{2,dim})
            voop = ODEFunction((u, p, t) -> rhs)
            viip = ODEFunction((du, u, p, t) -> du .= rhs)

            @test Id ≈ pb(voop, x0, tspan, 0.1)
            @test Id ≈ pb(viip, x0, tspan, 0.1)

            for x in (xs, xt, xv), v in (voop, viip)
                @test Id ≈ @inferred pb(v, x, tspan, 0.1)
            end
        end

        x0 = rand(dim)
        xs = SVector{dim}(x0)
        xt = (x0...,)
        xv = Vec{dim}(x0)
        voop = ODEFunction((u, p, t) -> rhs)
        viip = ODEFunction((du, u, p, t) -> du .= rhs)
        S = rand(SymmetricTensor{2,dim})
        B = rand(Tensor{2,dim})
        Sspan = fill(S, length(tspan))
        for v in (voop, viip)
            @test det(S)*inv(S) ≈ av_weighted_CG_tensor(v, x0, tspan, 0.1; D=S)
            @test Sspan ≈ pullback_diffusion_tensor(v, x0, tspan, 0.1; D=S)
            @test Sspan ≈ pullback_metric_tensor(v, x0, tspan, 0.1; G=S)
            @test fill(B, length(tspan)) ≈ pullback_SDE_diffusion_tensor(v, x0, tspan, 0.1; B=B)
        end
        for x in (xs, xt, xv), v in (voop, viip)
            @test det(S)*inv(S) ≈ @inferred(av_weighted_CG_tensor(v, x, tspan, 0.1; D=S))
            @test Sspan ≈ @inferred(pullback_diffusion_tensor(v, x, tspan, 0.1; D=S))
            @test Sspan ≈ @inferred(pullback_metric_tensor(v, x, tspan, 0.1; G=S))
            @test fill(B, length(tspan)) ≈ @inferred(pullback_SDE_diffusion_tensor(v, x, tspan, 0.1; B=B))
        end
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
        xt = (x0...,)
        xv = Vec{dim}(x0)
        Id = one(SymmetricTensor{2,dim})

        # test CG_tensor
        CG = exp(sI)'exp(sI)
        for v in (voop, viip)
            @test CG ≈ CG_tensor(v, x0, tspan, 1e-1) rtol=1e-5
            @test CG ≈ @inferred(CG_tensor(v, xs, tspan, 1e-1)) rtol=1e-5
            @test CG ≈ @inferred(CG_tensor(v, xt, tspan, 1e-1)) rtol=1e-5
            @test CG ≈ @inferred(CG_tensor(v, xv, tspan, 1e-1)) rtol=1e-5
        end
        # test mean_diff_tensor
        D̅ = 1//2 * (sI + inv(CG))
        for v in (voop, viip)
            @test D̅ ≈ mean_diff_tensor(v, x0, tspan, 1e-1) rtol=1e-5
            @test D̅ ≈ @inferred(mean_diff_tensor(v, xs, tspan, 1e-1)) rtol=1e-5
            @test D̅ ≈ @inferred(mean_diff_tensor(v, xt, tspan, 1e-1)) rtol=1e-5
            @test D̅ ≈ @inferred(mean_diff_tensor(v, xv, tspan, 1e-1)) rtol=1e-5
        end
    end
end
 
@testset "variational equation" begin
    include("define_vector_fields.jl")
    @test (@inferred CG_tensor(var_rot_double_gyre, (0.5, 0.5), [0.0, 5.0], 0.0)) isa SymmetricTensor
end
