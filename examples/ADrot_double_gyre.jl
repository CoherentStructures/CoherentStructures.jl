using CoherentStructures, OrdinaryDiffEq, DiffEqOperators,Sundials


ctx = regularTriangularGrid((25,25))
u0 = CoherentStructures.setup_fd_quadpoints_serialized(ctx,δ=1e-9)

p2 = Dict(
        "ctx" => ctx,
        "p"=>nothing,
        )

function large_rhs(du,u,p,t)
    n_quadpoints = length(p["ctx"].quadrature_points)
    @views CoherentStructures.arraymap!(du,u,p["p"],t,rot_double_gyre!,4*n_quadpoints,2)
end

function rot_double_gyre!(du,u,p,t)
        du .= rot_double_gyre(u,p,t)
end

prob = ODEProblem(large_rhs, u0,(0.0,1.0),p2)
sol = solve(prob,BS5())

function stiffnessMatrixTimeT(ctx,sol,t,δ=1e-9)
        if t < 0
                return assembleStiffnessMatrix(ctx)
        end
        p = sol(t)
        function Afun(x,index,p)
                Df = Tensors.Tensor{2,2}((p[(8*(index-1) + 1):(8*(index-1) + 4)] - p[ (8*(index-1)+5):(8*(index-1) + 8)])/(2δ))
                return Tensors.dott(Tensors.inv(Df))
        end
        return assembleStiffnessMatrix(ctx,Afun,p)
end



K = stiffnessMatrixTimeT(ctx,sol,0) + stiffnessMatrixTimeT(ctx,sol,1)
M = assembleMassMatrix(ctx)
λ,v = eigs(K,M,which=:SM)
plot_u(ctx,v[:,6])


function circle(x)
        return ((x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.1) ? 1.0 : 0.0
end

u0 = nodal_interpolation(ctx,circle)

ϵ=1e-2
plot_u(ctx,u0)
p = (ϵ,M,ctx,sol,1e-9,K)

#See also http://docs.juliadiffeq.org/latest/solvers/ode_solve.html#Sundials.jl-1

function compute_residual!(resid,du,u,p,t)
        print("At time $t")
        ϵ = p[1]
        M = p[2]
        ctx = p[3]
        sol = p[4]
        δ = p[5]
        K = stiffnessMatrixTimeT(ctx,sol,t,δ)
        resid .= M*du - ϵ*K*u
end

function rhs3!(du,u::Array{Float64},p,t::Float64)
        print("t = $t")
        ϵ = p[1]
        M = p[2]
        ctx = p[3]
        sol = p[4]
        δ = p[5]
        K = stiffnessMatrixTimeT(ctx,sol,t,δ)
        du.= ϵ*K*u
end


function rhsno_mass!(du,u::Array{Float64},p,t::Float64)
        print("t = $t")
        ϵ = p[1]
        M = p[2]
        ctx = p[3]
        sol = p[4]
        δ = p[5]
        K = p[6]
        du.= M\(ϵ*K*u)
end

differential_vars = [true for i in 1:size(M)[1]]
K0 = assembleStiffnessMatrix(ctx)
prob = DAEProblem(compute_residual!,M\(ϵ*K0*u0),u0,(0.0,1.0),differential_vars=differential_vars,p)
#prob = DAEProblem(compute_residual!,M\(ϵ*K*u0),u0,(0.0,1.0),differential_vars=differential_vars,p)

u = solve(prob,IDA(linear_solver=:Dense))

function update_coeffs(K,u,p,t)
        print("t = $t\n")
        ϵ = p[1]
        ctx = p[3]
        δ = p[5]
        sol = p[4]
        K .= ϵ*stiffnessMatrixTimeT(ctx,sol,0.0,δ)
        return nothing
end


L = DiffEqArrayOperator(ϵ*K,1.0,update_coeffs)
prob = ODEProblem(L, u0, (0.0,1.0),p,mass_matrix=M)
u = solve(prob ,ImplicitEuler(autodiff=false),
        progress=true,dt=1e-2,adaptive=false)


for i in collect(linspace(0,1,20))
  print(i)
  Plots.display(plot_u(ctx,u(i),clim=(-1,1),title="t=$i"))
end


function stiffrhs(du,u,p,t)
        print("$t\n")
        du .= -20*u
end

prob = ODEProblem(stiffrhs, [1.0], (0.0,10.0))
sol = solve(prob,ImplicitEuler(autodiff=false),dt=1e-1,adaptive=false)
Plots.plot(sol)
