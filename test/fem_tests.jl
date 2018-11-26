#Tests for FEM-code


using CoherentStructures
using Arpack,Plots,SparseArrays,LinearAlgebra,Tensors,StaticArrays


#Tests for adaptive TO method on periodic grid
function testAdaptiveTOPeriodicGrid()
    npoints=100*100
    LL = [0.0,0.0]
    UR = [2π,2π]

    ctx,bdata = randomDelaunayGrid(
        npoints;on_torus=true, LL=LL,UR=UR
        )
    #ctx, _ = regularP2TriangularGrid((50,50),LL,UR)
    #bdata = boundaryData(ctx,PEuclidean([2π,2π]))

    M = assembleMassMatrix(ctx,bdata=bdata)

    S = assembleStiffnessMatrix(ctx,bdata=bdata)
    S2= adaptiveTOCollocationStiffnessMatrix(
            ctx, CoherentStructures.standardMap;
            on_torus=true,bdata=bdata,
            volume_preserving=false,
            )

    D = 0.5*(S + S2)
    λ,v = eigs(D,M,which=:SM, nev=12)

    #Some tests so see nothing changed
    @assert abs(λ[2] - (-1.3)) < 1e-1
    @assert abs(λ[2] - λ[3]) < 1e-1

end

testAdaptiveTOPeriodicGrid()
