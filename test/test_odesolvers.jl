using CoherentStructures, Test
using OrdinaryDiffEq, DiffEqDevTools, SparseArrays, LinearAlgebra

@testset "odesolvers" begin
    dts = 0.5 .^(12:-1:7)
    tspan = (0.0, 1.0)

    @testset "autonomous ODEs, no mass matrix" begin
        N = 50
        u0 = sin.(range(0, pi, length = N))
        A = N^2*SymTridiagonal(fill(-2.0, N), ones(N-1))
        f = DiffEqArrayOperator(A)
        sol_analytic = (u0,p,t) -> exp(t*Matrix(A)) * u0
        prob = ODEProblem(ODEFunction(f; analytic=sol_analytic), u0, tspan)

        sim1 = test_convergence(dts, prob, LinearImplicitEuler(linsolve=LinSolveFactorize(factorize)))
        @test sim1.𝒪est[:final] ≈ 1 rtol=5e-2

        sim2 = test_convergence(dts, prob, LinearMEBDF2(linsolve=LinSolveFactorize(factorize)))
        @test sim2.𝒪est[:final] ≈ 2 rtol=5e-2
    end

    @testset "nonautonomous ODEs with mass matrix" begin
        N = 100
        M = 0.5Diagonal(ones(N))
        A = spdiagm(0 => ones(N))
        u0 = sin.(range(0, pi, length = N))
        update_func = (_A, u, p, t) -> (nzv = nonzeros(_A); nzv .= t; _A)
        f = DiffEqArrayOperator(A; update_func=update_func)

        sol_analytic = (u0, p, t) -> exp(t^2/(2*0.5)) .* u0
        prob = ODEProblem(ODEFunction(f; jac=update_func, jac_prototype=A, analytic=sol_analytic, mass_matrix=M), u0, tspan)

        sim1 = test_convergence(dts, prob, LinearImplicitEuler(linsolve=LinSolveFactorize(lu)))
        @test sim1.𝒪est[:final] ≈ 1 rtol=1e-2

        sim2 = test_convergence(dts, prob, LinearMEBDF2(linsolve=LinSolveFactorize(lu)))
        @test sim2.𝒪est[:final] ≈ 2 rtol=1e-2

        # solve(prob, LinearImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # solve(prob, LinearMEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # solve(prob, ImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # solve(prob, MEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # @time solve(prob, LinearImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.1);
        # @time solve(prob, ImplicitEuler(linsolve=LinSolveFactorize(lu)));
        # @time solve(prob, LinearMEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1);
        # @time solve(prob, MEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1);
    end

    @testset "ADE in rotating double gyre" begin
        # rot_double_gyre
        include("define_vector_fields.jl")
        N = 25
        δ = 1e-8
        ϵ = 1e-2
        ctx, _ = regularTriangularGrid((N, N))
        circleFun = x -> (sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) < 0.1) ? 1.0 : 0.0
        sol = CoherentStructures.advect_serialized_quadpoints(ctx, (0.0, 1.1), rot_double_gyre!, nothing, δ)
        M = assemble(Mass(), ctx)
        A = assemble(Stiffness(), ctx)

        function update_coeffs!(A, u, p, t)
            let ϵ=ϵ, δ=δ, sol=sol, ctx=ctx
                vals = nonzeros(A)
                vals .= nonzeros(CoherentStructures.stiffnessMatrixTimeT(ctx, sol, t, δ))
                rmul!(vals, ϵ)
                return A
            end
        end

        f = DiffEqArrayOperator(A; update_func=update_coeffs!)
        u0 = nodal_interpolation(ctx, circleFun)

        prob = ODEProblem(ODEFunction(f; jac=update_coeffs!, jac_prototype=A, mass_matrix=M), u0, tspan)

        sol1 = solve(prob, LinearImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.01)
        sol2 = solve(prob, LinearMEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.01)
        # solve(prob, ImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # solve(prob, MEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1)

        # @time solve(prob, LinearImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # @time solve(prob, ImplicitEuler(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # @time solve(prob, LinearMEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1)
        # @time solve(prob, MEBDF2(linsolve=LinSolveFactorize(lu)), dt=0.1)

        @test sol1.retcode === :Success
        @test sol2.retcode === :Success
        @test sol1.u[end] ≈ sol2.u[end] rtol=1e-3
    end
end
