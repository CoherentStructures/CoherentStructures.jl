using CoherentStructures, Test, Tensors, LinearAlgebra, LinearMaps, Random, Distances
using CoherentStructures: get_smallest_eigenpairs

@testset "1d grid piecewise_linear" begin
    ctx, _ = @inferred regular1dP1Grid(3)
    function u_fun(xin)
        #This function is does linear interpolation between 3 values
        x = xin[1]
        if x <= 0.5
            α = 2x
            return 25α + 0.2 * (1 - α)
        else
            α = 2 * (x - 0.5)
            return -3.2 * α + 25 * (1 - α)
        end
    end
    u = @inferred nodal_interpolation(ctx, u_fun)
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.0,))) == u_fun([0.0])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.5,))) == u_fun([0.5])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((1.0,))) == u_fun([1.0])
    for i in 1:500
        p = rand()
        @test evaluate_function_from_dofvals(ctx, u, Vec{1}((p,))) == u_fun([p])
    end
end

@testset "1d grid piecewise_quadratic on linear" begin
    ctx, _ = @inferred regular1dP2Grid(3)
    function u_fun(xin)
        #This function is does linear interpolation between 3 values
        x = xin[1]
        if x <= 0.5
            α = 2x
            return 25α + 0.2 * (1 - α)
        else
            α = 2 * (x - 0.5)
            return -3.2 * α + 25 * (1 - α)
        end
    end
    u = @inferred nodal_interpolation(ctx, u_fun)
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.0,))) == u_fun([0.0])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.25,))) ==
          u_fun([0.25])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.5,))) == u_fun([0.5])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.75,))) ==
          u_fun([0.75])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((1.0,))) == u_fun([1.0])
    for i in 1:500
        p = rand()
        @test abs(
            evaluate_function_from_dofvals(ctx, u, Vec{1}((p,))) - u_fun([p]),
        ) < 1e-14
    end
end

@testset "1d grid piecewise_quadratic on quadratic" begin
    ctx, _ = @inferred regular1dP2Grid(65)
    function quadfun(xin)
        return xin[1]^2
    end
    u = @inferred nodal_interpolation(ctx, quadfun)
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.0,))) == quadfun([0.0])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.25,))) == quadfun([0.25])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.5,))) == quadfun([0.5])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((0.75,))) == quadfun([0.75])
    @test evaluate_function_from_dofvals(ctx, u, Vec{1}((1.0,))) == quadfun([1.0])
    for i in 1:500
        p = rand()
        #if p < 1./64 or p > 1 - 1./64
        #    continue
        #end
        @test abs(
            evaluate_function_from_dofvals(ctx, u, Vec{1}((p,))) - quadfun([p]),
        ) < 1e-14
    end
end

@testset "1d grids Neumann BC" begin
    λᵢ = (π^2 * (0:5) .^ 2)
    # TODO: include (regular1dPCGrid, 1e-1),
    for grid in (regular1dP1Grid, regular1dP2Grid)
        ctx, _ = @inferred grid(200)
        M = @inferred assemble(Mass(), ctx)
        K = @inferred assemble(Stiffness(), ctx)
        λ, = get_smallest_eigenpairs(-K, M, 6)
        @test λ[1] ≈ 0 atol = √eps()
        @test λ[2:end] ≈ λᵢ[2:end] rtol = 1e-2
    end
end

@testset "1d grids Dirichlet BC" begin
    λᵢ = (π^2 * (1:6) .^ 2)
    # TODO: include (regular1dPCGrid, 1e-1),
    for (grid, tol) in ((regular1dP1Grid, 1e-3), (regular1dP2Grid, 1e-7))
        ctx, _ = @inferred grid(200)
        M = @inferred assemble(Mass(), ctx; bdata = getHomDBCS(ctx))
        K = @inferred assemble(Stiffness(), ctx; bdata = getHomDBCS(ctx))
        λ, = get_smallest_eigenpairs(-K, M, 6)
        @test λ ≈ λᵢ rtol = 1e-3
    end
end

@testset "1d non-adaptive TO" begin
    for grid in (regular1dP1Grid, regular1dP2Grid)
        ctx, _ = grid(200)
        bdata = BoundaryData(ctx, PeriodicEuclidean([1.0]))
        finv = x -> Base.mod.(x .- √2, 1.0)
        ALPHAS, _ = nonAdaptiveTOCollocation(
            ctx,
            finv,
            project_in = false,
            bdata_domain = bdata,
        )
        @test size(ALPHAS, 1) == size(ALPHAS, 2)
        u_initial = rand(size(ALPHAS, 1))
        u_initial_dofs = undoBCS(ctx, u_initial, bdata)
        pushforward_u = ALPHAS * u_initial
        pushforward_u_dofs = undoBCS(ctx, pushforward_u, bdata)
        for i in 1:ctx.n
            curnode = ctx.grid.nodes[i].x
            #@test abs(evaluate_function_from_dofvals(ctx,pushforward_u_dofs, curnode) -
            @test abs(
                pushforward_u_dofs[ctx.node_to_dof[i]] -
                evaluate_function_from_dofvals(
                    ctx,
                    u_initial_dofs,
                    Vec{1}((finv(curnode)[1],)),
                ),
            ) == 0.0
        end
    end
end

#Tests for adaptive TO method on periodic grid
@testset "adaptive TO FEM methods" begin
    npoints = 100 * 100
    LL = (0.0, 0.0)
    UR = (2π, 2π)

    Random.seed!(1234)
    ctx, bdata = randomDelaunayGrid(npoints, LL, UR; on_torus = true)
    #ctx, _ = regularP2TriangularGrid((50,50),LL,UR)
    #bdata = BoundaryData(ctx, PeriodicEuclidean([2π,2π]))

    M = assemble(Mass(), ctx, bdata = bdata)
    S = assemble(Stiffness(), ctx, bdata = bdata)
    T = adaptiveTOCollocationStiffnessMatrix(
        ctx,
        standardMap;
        on_torus = true,
        bdata = bdata,
        volume_preserving = false,
    )
    D = 0.5 * (S + T)

    λ, = get_smallest_eigenpairs(D, M, 3)
    @test all(<(sqrt(eps())), λ)

    LL, UR = (0., 0.), (1., 1.)
    gs = 10
    ctx, _ = regularTriangularGrid((gs, gs), LL, UR);
    predicate = (p1, p2) -> peuclidean(p1, p2, [1.0,Inf]) < 2e-10
    bdata = BoundaryData(ctx, predicate, [])

    @test !isnothing(
        adaptiveTOCollocationStiffnessMatrix(
            ctx, identity; on_cylinder=true, bdata)
       )
end
