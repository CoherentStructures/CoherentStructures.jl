
#include("util.jl")

function flow2D(g::Function,u0::Vec{2},tspan::Vector{Float64},tol=1.e-3)
   prob = OrdinaryDiffEq.ODEProblem(g,[u0[1],u0[2]],(tspan[1],tspan[end]))
   return Vec{2}(OrdinaryDiffEq.solve(prob,OrdinaryDiffEq.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=tol,abstol=tol).u[end])
end

#Calculate derivative of flow map by finite differences.
@inline function LinearizedFlowMap(odefun,x::Vec{2,Float64},tspan::Array{Float64}, δ::Float64,tolerance=1.e-3)
    const dx = [δ,0];
    const dy = [0,δ];
    #In order to solve only one ODE, write all the initial values
    #one after the other in one big vector
    stencil = zeros(8)
    @inbounds stencil[1:2] .= x.+dx
    @inbounds stencil[3:4] .= x.+dy
    @inbounds stencil[5:6] .= x.-dx
    @inbounds stencil[7:8] .= x.-dy

    const num_tsteps = length(tspan)
    #TODO: Make p do something here
    prob = OrdinaryDiffEq.ODEProblem((du,u,p,t) -> arraymap(du,u,p,t,odefun, 4,2),stencil,(tspan[1],tspan[end]))
    sol = OrdinaryDiffEq.solve(prob,OrdinaryDiffEq.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
    result = zeros(Tensor{2,2},num_tsteps)
    @inbounds for i in 1:num_tsteps
        #The ordering of the stencil vector was chosen so
        #that  a:= stencil[1:4] - stencil[5:8] is a vector
        #so that Tensor{2,2}(a) approximates the Jacobi-Matrix
	@inbounds result[i] = Tensor{2,2}( (sol[i][1:4] - sol[i][5:8])/2δ)
    end
    return result
end

#TODO: document this
function LinearizedFlowMap(odefun,x₀::Vector{Float64},tspan::Vector{Float64})

    prob = ODEProblem(odefun,[x₀; 1.; 0.; 0.; 1.],(tspan[1],tspan[end]))
    # BS5 seems to be a bit faster than Tsit5 in this case
    sol = solve(prob,DP5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-3,abstol=1e-3).u
    return [Tensor{2,2}(reshape(s[3:6],2,2)) for s in sol]
end


#Returns the average (inverse) CG-Tensor at a point along a set of times
#Derivatives are computed with finite differences
#@param x::Vec{2,Float64} the initial point
#@param tspan::Array{Float64} is the times
#@param δ is the stencil width for the finite differences
#@param odefun is a function that takes arguments (x,t,result)
#   odefun needs to store its result in result
#   odefun evaluates the rhs of the ODE being integrated at (x,t)
@inline function invCGTensor(odefun,x::Vec{2,Float64},tspan::Array{Float64}, δ::Float64,tolerance=1.e-3)
    return mean(dott.(inv.(LinearizedFlowMap(odefun,x,tspan,δ,tolerance))))
end



#TODO: Document the functions below, then add to exports.jl
function PullBackTensors(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,2,Float64,3})

    G = inv(D)
    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    DFinv = inv.(DF)
    DF = [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
    DFinv = [symmetric(df⋅(D⋅transpose(df))) for df in DFinv]
    return DF, DFinv # DF is pullback metric tensor, DFinv is pullback diffusion tensor
end

function PullBackMetricTensor(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,G::Tensors.SymmetricTensor{2,2,Float64,3})

    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    return [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
end

function PullBackDiffTensor(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,2,Float64,3})

    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    DF = inv.(DF)
    return [symmetric(df⋅(D⋅transpose(df))) for df in DF]
end
