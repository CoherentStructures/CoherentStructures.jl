using CoherentStructures, Test

include("define_vector_fields.jl")

@testset "FEMheatflow" begin
    ctx, _ = regularTriangularGrid((100, 100))
    M = assembleMassMatrix(ctx)
    U = FEM_heatflow(rot_double_gyre!, ctx, range(0., stop=1., length=11), 1e-3; factor=true)
    λ, V = diffusion_coordinates(U, 6)
    @test first(λ) ≈ 1 rtol=1e-7
end

@testset "DMheatflow" begin
    n = 500
    tspan = range(0, stop=1.0, length=20)
    xs, ys = rand(n), rand(n)
    particles = zip(xs, ys)
    U = DM_heatflow(u -> flow(rot_double_gyre, u, tspan), particles, Neighborhood(0.1), gaussian(0.1))
    λ, V = diffusion_coordinates(U, 6)
    @test first(λ) ≈ 1
end