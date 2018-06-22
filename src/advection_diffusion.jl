#(c) 2018 Nathanael Schilling, with minor contributions by Daniel Karrasch
# Routines for numerically solving the Advection-Diffusion Equation in Lagrangian
# coordinates

# meta function
"""
    FEM_heatflow(velocity!, ctx, tspan, κ, decomp=false, p=nothing, bdata=boundaryData(); kwargs...)

Compute the heat flow operator for the time-dependent heat equation, which
corresponds to the advection-diffusion equation in Lagrangian coordinates, by
employing a spatial FEM discretization and temporal implicit-Euler time stepping.

## Arguments
   * `odefun!`: velocity field in inplace form, e.g.,
     `odefun!(du,u,p,t) = du .= odefun(u,p,t)`;
   * `ctx`: grid context for the FEM discretization;
   * `tspan`: time span determining the temporal resolution at which the heat
     flow is to be computed internally;
   * `κ`: diffusivity constant;
   * `p`: parameter container for the vector field;
   * `kwargs`: are passed to `advect_serialized_quadpoints`.
"""
# TODO: Restrict tspan to be of Range/linspace type
function FEM_heatflow(odefun!, ctx, tspan, κ, decomp::Bool=false, p = nothing, bdata = boundaryData(); kwargs...)

    sol = advect_serialized_quadpoints(ctx, tspan, odefun!, p; kwargs...)
    return implicitEulerStepFamily(ctx, sol, tspan, κ; bdata=bdata)
end

function implicitEulerStepFamily(ctx, sol, tspan, κ, decomp::Bool=false; bdata=boundaryData())

    M = assembleMassMatrix(ctx, bdata=bdata)
    n = size(M)[1]
    Δτ = step(tspan)
    P = map(tspan[1:end-1]) do t
        K = stiffnessMatrixTimeT(ctx, sol, t, bdata=bdata)
        ΔM = decomp ? factorize(M - Δτ * κ * K) : M - Δτ * κ * K
        # TODO: Think about in-place versions (u,v) -> u .= lin_map(v)
        Pₜ = LinearMaps.LinearMap(
             x -> ΔM \ (M * x),
             x -> M * (ΔM \ x),
             n)
        println("Integration time $t done")
        Pₜ
    end
    return prod(reverse(P))
end

"""
    ADimplicitEulerStep(ctx,u,edt, Afun,q=nothing,M=nothing,K=nothing)

Single step with implicit Euler method.
"""
function ADimplicitEulerStep(ctx, u, edt, Afun, q=nothing, M=nothing, K=nothing)
    if M == nothing
        M = assembleMassMatrix(ctx)
    end
    if K == nothing
        K = assembleStiffnessMatrix(ctx, Afun, q)
    end
    return (M - edt * K) \ (M * u)
end


"""
    advect_serialized_quadpoints(ctx, tspan, odefun!, p=nothing, δ=1e-9;
            solver=default_solver, tolerance=default_tolerance)

Advect all quadrature points + finite difference stencils of size `δ`,
from `tspan[1]` to `tspan[end]` with ODE rhs given by `odefun!`, whose parameters
are contained in `p`. Returns an ODE solution object.
"""
function advect_serialized_quadpoints(ctx, tspan, odefun!, p=nothing, δ=1e-9;
                    solver=default_solver,tolerance=default_tolerance)

    u0 = setup_fd_quadpoints_serialized(ctx, δ)
    p2 = Dict(
        "ctx" => ctx,
        "p"=>p,
        )

    prob = OrdinaryDiffEq.ODEProblem((du, u, p, t) -> large_rhs(odefun!, du, u, p, t),
                                        u0, (tspan[1], tspan[end]), p2)
    return OrdinaryDiffEq.solve(prob, solver, abstol=tolerance, reltol=tolerance)
end


function stiffnessMatrixTimeT(ctx,sol,t,δ=1e-9; bdata=bondaryData())
    if t < 0
        return assembleStiffnessMatrix(ctx,bdata=bdata)
    end
    p = sol(t)
    function Afun(x,index,p)
        Df = Tensors.Tensor{2,2}(
            (p[(8*(index-1) + 1):(8*(index-1) + 4)] -
                    p[ (8*(index-1)+5):(8*(index-1) + 8)])/(2δ) )
        return Tensors.dott(inv(Df))
    end
    return assembleStiffnessMatrix(ctx, Afun, p, bdata=bdata)
end

@inline function large_rhs(odefun!, du, u, p, t)
    n_quadpoints = length(p["ctx"].quadrature_points)
    @views arraymap!(du, u, p["p"], t, odefun!, 4n_quadpoints, 2)
end

"""
    setup_serialized_quadpoints(ctx,δ=1e-9)

For each quadrature point in `ctx.quadrature_points`, setup a finite difference scheme around that point.
Then write the resulting points into a (flat) array onto which arraymap can be applied.
Only works in 2D at the moment
"""
function setup_fd_quadpoints_serialized(ctx, δ=1e-9)
    n_quadpoints = length(ctx.quadrature_points)
    u_full = zeros(8n_quadpoints)
    @inbounds for (i,q) in enumerate(ctx.quadrature_points)
        for j in 1:4
            factor = j>2 ? -1 : 1
            shift = (j % 2 == 0) ? 1 : 0
            u_full[8 * (i - 1) + 2(j - 1) + 1] = q[1]
            u_full[8 * (i - 1) + 2(j - 1) + 2] = q[2]
            u_full[8 * (i - 1) + 2(j - 1) + 1 + shift] += factor * δ
        end
    end
    return u_full
end

#Helper function for the case where we've precomputed the diffusion tensors
function PCDiffTensors(x, index, p)
    return p[index]
 end

function extendedRHSNonStiff(odefun, du, u, p, t)
    print("t = $t")
    n_quadpoints = p["n_quadpoints"]
    @views arraymap(du, u, p["p"], t, odefun, 4*n_quadpoints, 2)
end

function extendedRHSStiff!(A, u, p, t)
    #First 4*n_quadpoints*2 points are ODE for flow-map at quadrature points
    #The next n points are for the solution of the AD-equation

    n_quadpoints = p["n_quadpoints"]
    n = p["n"]
    δ = p["δ"]
    DF = map(1:n_quadpoints) do i
        DF[i] = Tensors.Tensor{2,2}(
            (u[(4*(i-1) +1):(4*i)] - u[(4*n_quadpoints +1 +4*(i-1)):(4*n_quadpoints+4*i)])/2δ
            )
    end
    invDiffTensors = Tensors.dott.(inv.(DF))
    ctx = p["ctx"]
    κ = p["κ"]
    K = assembleStiffnessMatrix(ctx, PCDiffTensors, invDiffTensors)
    Is, Js, Vs = findnz(K)
    M = assembleMassMatrix(ctx, lumped=true)
    for index in eachindex(Is, Js, Vs)
        i = Is[index]
        j = Js[index]
        A[i,j] = κ * Vs[index] / M[i,i]
    end
    print("t = $t")
    return A
end
