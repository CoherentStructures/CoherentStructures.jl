#Functions for pulling back Tensors




#Solve the ODE with right hand side given by @param rhs and initial value given by @param u0
#dim is the dimension of the ODE
#p is a parameter passed as the third argument to rhs
function flow(rhs::Function,u0::Vec{dim,T},tspan::AbstractVector{Float64};tolerance=1.e-3,p=nothing,solver=OrdinaryDiffEq.BS5()) where {dim,T<:Real}
   prob = OrdinaryDiffEq.ODEProblem(rhs,Array{T}(u0),(tspan[1],tspan[end]),p)
   #TODO: can we use point notation here, or something from the DifferentialEquations package?
   return map( Vec{2},OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u)
end

#Calculate derivative of flow map by finite differences.
#TODO: implement this for dim==3, currently it only works if dim==2
@inline function linearized_flow(odefun::Function,x::Vec{dim,T},tspan::AbstractVector{Float64}, δ::Float64;tolerance::Float64=1.e-3,p=nothing,solver=OrdinaryDiffEq.BS5()) where {T <: Real,dim}
    const dx = [δ,zero(δ)];
    const dy = [zero(δ),δ];
    #In order to solve only one ODE, write all the initial values
    #one after the other in one big vector
    stencil = zeros(8)
    @inbounds stencil[1:2] .= x.+dx
    @inbounds stencil[3:4] .= x.+dy
    @inbounds stencil[5:6] .= x.-dx
    @inbounds stencil[7:8] .= x.-dy

    const num_tsteps = length(tspan)
    prob = OrdinaryDiffEq.ODEProblem((du,u,p,t) -> arraymap(du,u,p,t,odefun, 4,2),stencil,(tspan[1],tspan[end]),p)
    sol = OrdinaryDiffEq.solve(prob,solver,saveat=tspan,save_everystep=false,dense=false,reltol=tolerance,abstol=tolerance).u
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
function linearized_flow(odefun::Function,u::Vec{dim,T},tspan::AbstractVector{Float64}) where {T <: Real,dim}
    Flow(x) = flow(odefun,x,tspan,p=p)
    DF      = ForwardDiff.jacobian(Flow,u)
    df      = [Tensor{2,2}(DF[i:i+(dim-1),:]) for i=1:dim:size(DF,1)]
    return df
end

#Returns the average (inverse) CG-Tensor at a point along a set of times
#Derivatives are computed with finite differences
#@param x::Vec{2,Float64} the initial point
#@param tspan::Array{Float64} is the times
#@param δ is the stencil width for the finite differences
#@param odefun is a function that takes arguments (x,t,result)
#   odefun needs to store its result in result
#   odefun evaluates the rhs of the ODE being integrated at (x,t)
@inline function invCGTensor(odefun,x::Vec{dim,T},tspan::AbstractVector{Float64}, δ::Float64;tolerance=1.e-3,p=nothing) where {T<:Real, dim}
    return mean(dott.(inv.(linearized_flow(odefun,x,tspan,δ,tolerance=tolerance,p=p))))
end

#TODO: Document the functions below, then add to exports.jl
#TODO: Pass through tolerance for ODE solver etc.
function pullback_tensors(odefun::Function,u::Vec{dim,T},tspan::AbstractVector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,dim,T,3};p=nothing,tolerance=1.e-3,solver=OrdinaryDiffEq.BS5()) where {T <: Real,dim}

    G = inv(D)
    iszero(δ) ? DF = linearized_flow(odefun,u,tspan,p=p,tolerance=tolerance,solver=solver) : DF = linearized_flow(odefun,u,tspan,δ,p=p,tolerance=tolerance,solver=solver)
    DFinv = inv.(DF)
    MT = [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
    DF .= inv.(DF)
    DT = [symmetric(df⋅(D⋅transpose(df))) for df in DF]
    return MT, DT # MT is pullback metric tensor, DT is pullback diffusion tensor
end

function pullback_metric_tensor(odefun,u::Vec{dim, T},tspan::AbstractVector{Float64},
    δ::Float64,G::Tensors.SymmetricTensor{2,dim,T,3};p=nothing,tolerance=1.e-3,solver=OrdinaryDiffEq.BS5()) where {T <: Real,dim}

    iszero(δ) ? DF = linearized_flow(odefun,u,tspan,p=p,tolerance=tolerance,solver=solver) : DF = linearized_flow(odefun,u,tspan,δ,p=p,tolerance=tolerance,solver=solver)

    return [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
end

function pullback_diffusion_tensor(odefun,u::Vec{dim,T},tspan::AbstractVector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,dim,T}; p=nothing,tolerance=1.e-3,solver=OrdinaryDiffEq.BS5()) where {T <: Real,dim}

    iszero(δ) ? DF = linearized_flow(odefun,u,tspan,p=p,tolerance=tolerance,solver=solver) : DF = linearized_flow(odefun,u,tspan,δ,p=p,tolerance=tolerance,solver=solver)

    DF = inv.(DF)
    return [symmetric(df⋅(D⋅transpose(df))) for df in DF]
end

#TODO: Think whether to define the functions below for the Vec{} type
function met2deg(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2}, [cos(deg2rad(u[2])), one(T)])
end

function deg2met(u::AbstractVector{T}) where T <: Real
    diagm(Tensor{2,2}, [1/cos(deg2rad(u[2])), one(T)])
end
