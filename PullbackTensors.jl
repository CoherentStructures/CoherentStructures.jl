@everywhere using Tensors, DiffEqBase, OrdinaryDiffEq#, ForwardDiff
@everywhere include("util.jl")

 @everywhere function flow2D(g::Function,u0::Vec{2},tspan::Vector{Float64},tol=1.e-3)
   prob = OrdinaryDiffEq.ODEProblem(g,[u0[1],u0[2]],(tspan[1],tspan[end]))
   return Vec{2}(OrdinaryDiffEq.solve(prob,OrdinaryDiffEq.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=tol,abstol=tol).u[end])
 end

#Calculate derivative of flow map by finite differences.
 @inline function finDiffDFlowMap(x::Vec{2,Float64},tspan::Array{Float64}, δ::Float64,ode_fun,tolerance=1.e-3)
    dx = [δ,0]; dy = [0,δ];
    #In order to solve only one ODE, write all the initial values
    #one after the other in one big vector
    stencil = zeros(8)
    stencil[1:2] = x+dx
    stencil[3:4] = x+dy
    stencil[5:6] = x-dx
    stencil[7:8] = x-dy
    prob = OrdinaryDiffEq.ODEProblem((t,x,result) -> arraymap(ode_fun, 4,2,t,x,result),stencil,(tspan[1],tspan[end]))
    sol = OrdinaryDiffEq.solve(prob,OrdinaryDiffEq.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
    return Tensor{2,2}(sol[end][1:4] - sol[end][5:8])/(2δ)
end



#Returns the average (inverse) CG-Tensor at a point along a set of times
#Derivatives are computed with finite differences
#@param x::Vec{2,Float64} the initial point
#@param tspan::Array{Float64} is the times
#@param δ is the stencil width for the finite differences
#@param ode_fun is a function that takes arguments (x,t,result)
#   ode_fun needs to store its result in result
#   ode_fun evaluates the rhs of the ODE being integrated at (x,t)
#TODO: Use LinearizedFlowMap() to avoid code duplication
@inline function invCGTensor(x::Vec{2,Float64},tspan::Array{Float64}, δ::Float64,ode_fun,tolerance=1.e-3)
    dx = [δ,0]; dy = [0,δ];
    #In order to solve only one ODE, write all the initial values
    #one after the other in one big vector
    stencil = zeros(8)
    stencil[1:2] = x+dx
    stencil[3:4] = x+dy
    stencil[5:6] = x-dx
    stencil[7:8] = x-dy
    prob = OrdinaryDiffEq.ODEProblem((t,x,result) -> arraymap(ode_fun, 4,2,t,x,result),stencil,(tspan[1],tspan[end]))
    sol = OrdinaryDiffEq.solve(prob,OrdinaryDiffEq.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
    const num_tsteps = length(tspan)
    result = zeros(Tensor{2,2},num_tsteps)
    @inbounds for i in 1:num_tsteps
        #The ordering of the stencil vector was chosen so
        #that  a:= stencil[1:4] - stencil[5:8] is a vector
        #so that Tensor{2,2}(a) approximates the Jacobi-Matrix
        @inbounds result[i] = Tensor{2,2}(sol[i][1:4] - sol[i][5:8])
    end
    return mean(dott.(inv.(result))) * 4δ*δ
end


@everywhere function LinearizedFlowMap(odefun,x₀::Vector{Float64},tspan::Vector{Float64},δ::Float64)

    dx = [δ,0.]; dy = [0.,δ];
    stencil = [x₀+dx; x₀+dy; x₀-dx; x₀-dy]
    prob = ODEProblem((t,x,result) -> arraymap(odefun,4,2,t,x,result),stencil,(tspan[1],tspan[end]))
    # Tsit5 seems to be a bit faster than BS5 in this case
    sol = solve(prob,DP5(),saveat=tspan,save_everystep=false,dense=false,reltol=1.e-3,abstol=1.e-3).u
    return [Tensor{2,2}((s[1:4] - s[5:8])./2δ) for s in sol]
end

@everywhere function LinearizedFlowMap(odefun,x₀::Vector{Float64},tspan::Vector{Float64})

    prob = ODEProblem(odefun,[x₀; 1.; 0.; 0.; 1.],(tspan[1],tspan[end]))
    # BS5 seems to be a bit faster than Tsit5 in this case
    sol = solve(prob,DP5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-3,abstol=1e-3).u
    return [Tensor{2,2}(reshape(s[3:6],2,2)) for s in sol]
end

@everywhere function PullBackTensors(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,2,Float64,3})

    G = inv(D)
    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    DFinv = inv.(DF)
    DF = [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
    DFinv = [symmetric(df⋅(D⋅transpose(df))) for df in DFinv]
    return DF, DFinv # DF is pullback metric tensor, DFinv is pullback diffusion tensor
end

@everywhere function PullBackMetricTensor(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,G::Tensors.SymmetricTensor{2,2,Float64,3})

    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    return [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
end

@everywhere function PullBackDiffTensor(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,2,Float64,3})

    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    DF = inv.(DF)
    return [symmetric(df⋅(D⋅transpose(df))) for df in DF]
end
