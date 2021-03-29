using CoherentStructures, OrdinaryDiffEq, StreamMacros, Plots#, DiffEqOperators, Sundials

@define_stream Ψ_rot_dgyre begin
    st          = heaviside(t)*heaviside(1-t)*t^2*(3-2*t) + heaviside(t-1)
    heaviside(x)= 0.5*(sign(x) + 1)
    Ψ_P         = sin(2π*x)*sin(π*y)
    Ψ_F         = sin(π*x)*sin(2π*y)
    Ψ_rot_dgyre = (1-st) * Ψ_P + st * Ψ_F
end
rot_double_gyre! = @velo!_from_stream Ψ_rot_dgyre

ctx, _ = regularTriangularGrid((100, 100))
M = assembleMassMatrix(ctx)
@time U = FEM_heatflow(rot_double_gyre!, ctx, range(0., stop=1., length=11), 1e-3; factor=true)
@time λ, V = diffusion_coordinates(U, 6)
plot_u(ctx, V[:,4], 200, 200)

####### Stuff below is just testing, don't delete yet


###BEGIN EXAMPLE

using CoherentStructures, OrdinaryDiffEq#, DiffEqOperators

δ = 1e-8
ctx = regularTriangularGrid((25,25))
sol = CoherentStructures.advect_serialized_quadpoints(ctx, (0.0,1.0), rot_double_gyre!, nothing, δ;
        tolerance=1e-4)
M = assembleMassMatrix(ctx)

function update_coeffs(K,u,p,t)
        print("t = $t\n")
        ϵ = p[1]
        ctx = p[3]
        δ = p[5]
        sol = p[4]
        #Uncomment the line below
        #K .= ϵ*CoherentStructures.stiffnessMatrixTimeT(ctx,sol,t,δ)
        return nothing
end

L = DiffEqArrayOperator(ϵ*assembleStiffnessMatrix(ctx),1.0,update_coeffs)

circleFun = x -> ((x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.1) ? 1.0 : 0.0
ϵ=1e-2
p = (ϵ,M,ctx,sol,δ)
prob = ODEProblem(L, nodal_interpolation(ctx,circleFun), (0.0,1.0),p,mass_matrix=M)


function linsolve!(::Type{Val{:init}},f,u0)
  function _linsolve!(x,A,b,update_matrix=false)
          x .= A \  b
  end
end
u = solve(prob ,ImplicitEuler(autodiff=false,linsolve=linsolve!),
        progress=true,dt=1e-1,adaptive=false)

for t in range(0,stop=1,length=10)
        Plots.display(plot_u(ctx,u(t)))
end


######## END EXAMPLE


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







for i in collect(range(0,stop=1,length=20))
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

fs = []

for i in 1:10
        push!(fs, x->i)
        print(fs[1](1))
end

f(2)
a = 4



a=3
f(2)
