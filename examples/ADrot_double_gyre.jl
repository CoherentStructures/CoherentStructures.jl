using CoherentStructures, OrdinaryDiffEq, DiffEqOperators,Sundials


ctx = regularTriangularGrid((25,25))

function rot_double_gyre!(du,u,p,t)
        du .= rot_double_gyre(u,p,t)
end


using LinearMaps
function implicitEulerStepFamily(ctx,sol,t0,tf,nt,ϵ; bdata=boundaryData())
        M = assembleMassMatrix(ctx,bdata=bdata)
        n = size(M)[1]
        result = LinearMap[]
        for i in 0:(nt-1)
                dt = (tf - t0)/nt
                t = t0 + dt*i
                K = CoherentStructures.stiffnessMatrixTimeT(ctx,sol,t,bdata=bdata)

                currentmap = LinearMap(
                     x -> (M - dt*ϵ*K)\(M*x) ,
                     x -> (M*((M - dt*ϵ*K)\x)),
                     n
                        )
                push!(result,currentmap)
        end
        return result
end

bdata = boundaryData()
sol = CoherentStructures.advect_serialized_quadpoints(ctx,0.,1.,rot_double_gyre!,
        nothing,1e-9,tolerance=1e-4,solver=BS5())
steps = implicitEulerStepFamily(ctx,sol,0,1,10,1e-2,bdata=bdata)
λ, V = diffusion_coordinates(prod(reverse(steps)),6)
plot_u(ctx,V[:,2])


####### Stuff below is just testing, only partially related to stuff above ####



@time K = stiffnessMatrixTimeT(ctx,sol,0.0) + stiffnessMatrixTimeT(ctx,sol,1)
@code_warntype assembleStiffnessMatrix(ctx)
@time assembleMassMatrix(ctx)

M = assembleMassMatrix(ctx)
λ,v = eigs(K,M,which=:SM)
plot_u(ctx,v[:,5])


function circle(x)
        return ((x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.1) ? 1.0 : 0.0
end

circ = nodal_interpolation(ctx,circle)

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
        K .= ϵ*stiffnessMatrixTimeT(ctx,sol,t,δ)
        return nothing
end

function linsolve!(::Type{Val{:init}},f,u0)
  function _linsolve!(x,A,b,update_matrix=false)
          x .= A \  b
  end
end




L = DiffEqArrayOperator(ϵ*K0,1.0,update_coeffs)
prob = ODEProblem(L, u0, (0.0,1.0),p,mass_matrix=M)
u = solve(prob ,ImplicitEuler(autodiff=false,linsolve=linsolve!),
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

fs = []

for i in 1:10
        push!(fs, x->i)
        print(fs[1](1))
end

f(2)
a = 4



a=3
f(2)
