#Tests for FEM-code

using CoherentStructures, Test
using Arpack


#Tests for adaptive TO method on periodic grid
@testset "adaptive TO FEM methods" begin
    npoints = 100 * 100
    LL = [0.0, 0.0]
    UR = [2π, 2π]

    ctx, bdata = randomDelaunayGrid(npoints; on_torus=true, LL=LL, UR=UR)
    #ctx, _ = regularP2TriangularGrid((50,50),LL,UR)
    #bdata = boundaryData(ctx,PEuclidean([2π,2π]))

    M = assembleMassMatrix(ctx, bdata=bdata)
    S = assembleStiffnessMatrix(ctx, bdata=bdata)
    T = adaptiveTOCollocationStiffnessMatrix(
            ctx, CoherentStructures.standardMap;
            on_torus=true, bdata=bdata,
            volume_preserving=false)
    D = 0.5 * (S + T)

    λ, v = eigs(D, M, which=:SM, nev=12)

    #Some tests so see nothing changed
    @test abs(λ[2] - (-1.3)) < 1e-1
    @test abs(λ[2] - λ[3]) < 1e-1
end
