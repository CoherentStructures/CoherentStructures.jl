using Test, StaticArrays, OrdinaryDiffEq, AxisArrays, LinearAlgebra
using CoherentStructures

@testset "singularity detection" begin
    q = 3
    tspan = range(0., stop=1., length=q)
    ny = 52
    nx = 51
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    xspan = range(xmin, stop=xmax, length=nx)
    yspan = range(ymin, stop=ymax, length=ny)
    P = SVector{2}.(xspan, yspan')
    mCG_tensor = u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, 1.e-6)
    Ts = @inferred map(mCG_tensor, P)
    T = AxisArray(Ts, xspan, yspan)
    ξ = [eigvecs(t)[:,1] for t in T]
    α = AxisArray([atan(v[2], v[1]) for v in ξ], T.axes)
    singularities = @inferred compute_singularities(α, π)
    new_singularities = @inferred combine_singularities(singularities, 3*step(xspan))
    @inferred CoherentStructures.combine_isolated_pairs(new_singularities)
    r₁ , r₂ = rand(2)
    @test sum(get_indices(combine_singularities(singularities, r₁))) ==
        sum(get_indices(combine_singularities(singularities, r₂))) ==
        combine_singularities(singularities, 2)[1].index
end
