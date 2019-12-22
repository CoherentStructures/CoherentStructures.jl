#Tests for FEM-code

using CoherentStructures, Test
using Arpack

@testset "1d grids Neumann BC" begin
    λᵢ = (π^2*(0:5).^2)
    # TODO: include (regular1dPCGrid, 1e-1),
    for grid in (regular1dP1Grid, regular1dP2Grid)
        ctx, _ = @inferred regular1dP1Grid(200)
        M = @inferred assembleMassMatrix(ctx)
        K = @inferred assembleStiffnessMatrix(ctx)
        λ, v = eigs(-K, M, which=:SM, nev=6)
        @test λ[1] ≈ 0 atol=√eps()
        @test λ[2:end] ≈ λᵢ[2:end] rtol=1e-3
    end
end

@testset "1d grids Dirichlet BC" begin
    λᵢ = (π^2*(1:6).^2)
    # TODO: include (regular1dPCGrid, 1e-1),
    for (grid, tol) in ((regular1dP1Grid, 1e-3), (regular1dP2Grid, 1e-7))
        ctx, _ = @inferred grid(200)
        M = @inferred assembleMassMatrix(ctx; bdata=getHomDBCS(ctx))
        K = @inferred assembleStiffnessMatrix(ctx; bdata=getHomDBCS(ctx))
        λ, v = eigs(-K, M, which=:SM, nev=6)
        @test λ ≈ λᵢ rtol=1e-3
    end
end

#Tests for adaptive TO method on periodic grid
@testset "adaptive TO FEM methods" begin
    npoints = 100 * 100
    LL = (0.0, 0.0)
    UR = (2π, 2π)

    ctx, bdata = randomDelaunayGrid(npoints, LL, UR; on_torus=true)
    #ctx, _ = regularP2TriangularGrid((50,50),LL,UR)
    #bdata = BoundaryData(ctx, PeriodicEuclidean([2π,2π]))

    M = assembleMassMatrix(ctx, bdata=bdata)
    S = assembleStiffnessMatrix(ctx, bdata=bdata)
    T = adaptiveTOCollocationStiffnessMatrix(
            ctx, standardMap;
            on_torus=true, bdata=bdata,
            volume_preserving=false)
    D = 0.5 * (S + T)

    λ, v = eigs(D, M, which=:SM, nev=12)

    #Some tests to see nothing changed
    @test abs(λ[2] - (-1.3)) < 1e-1
    @test abs(λ[2] - λ[3]) < 1e-1
end
