#tensorComputations.jl from Daniel Karrasch
using Tensors, ForwardDiff
import DifferentialEquations
const DE = DifferentialEquations

function flow(g,u0,tspan)
  Tspan = (tspan[1], tspan[end])
  prob = DE.ODEProblem(g,u0,eltype(u0).(Tspan))
  sol = DE.solve(prob,saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u[end]
end

function avCGTensor(initval::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,myfun,G::Tensors.SymmetricTensor{2,2,Float64,3})
    Tspan = (tspan[1],tspan[end])
    dx = [δ,0]; dy = [0,δ];
    prob1 = DE.ODEProblem(myfun,initval+dx,Tspan)
    prob2 = DE.ODEProblem(myfun,initval-dx,Tspan)
    prob3 = DE.ODEProblem(myfun,initval+dy,Tspan)
    prob4 = DE.ODEProblem(myfun,initval-dy,Tspan)
    sol1 = DE.solve(prob1,DE.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol2 = DE.solve(prob2,DE.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol3 = DE.solve(prob3,DE.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol4 = DE.solve(prob4,DE.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    # DF = Tensor{2,2}.([ForwardDiff.jacobian(x -> flow(myfun,x,(tspan[1], t)),initval) for t in tspan[2:end]])
    DF = Tensor{2,2}.(hcat.((sol1-sol2)/2δ,(sol3-sol4)/2δ))
    DF = SymmetricTensor{2,2}.([transpose(D)⋅G⋅D for D in DF])
    return mean(DF)
end

#The following function is like `map', but operates on 1d-datastructures.
#@param t::Float64 is just some number
#@param x::Float64 must have howmanytimes*basesize elements
#@param myfun is a function that takes arguments (t, x, result)
#     where t::Float64, x is an Array{Float64} of size basesize,
#       and result::Array{Float64} is of size basesize
#       myfun is assumed to return the result into the result array passed to it
#This function applies myfun consecutively to slices of x, and stores
#the result in the relevant slice of result.
#This is so that a "diagonalized" ODE with several starting values can
#be solved without having to call the ODE multiple times.
@inline function arraymap(myfun,howmanytimes::Int64,basesize::Int64,t::Float64,x::Array{Float64},result::Array{Float64})
    @inbounds for i in 1:howmanytimes
        @views @inbounds  myfun(t,x[ 1 + (i-1)*basesize:  i*basesize],result[1 + (i - 1)*basesize: i*basesize])
    end
end

#Returns the average (inverse) CG-Tensor at a point along a set of times
#Derivatives are computed with finite differences
#@param x::Vec{2,Float64} the initial point
#@param tspan::Array{Float64} is the times
#@param δ is the stencil width for the finite differences
#@param ode_fun is a function that takes arguments (x,t,result)
#   ode_fun needs to store its result in result
#   ode_fun evaluates the rhs of the ODE being integrated at (x,t)
function avDiffTensor(x::Vec{2,Float64},tspan::Array{Float64}, δ::Float64,ode_fun)
    dx = [δ,0]; dy = [0,δ];
    #In order to solve only one ODE, write all the initial values
    #one after the other in one big vector
    stencil = zeros(8)
    stencil[1:2] = x+dx
    stencil[3:4] = x+dy
    stencil[5:6] = x-dx
    stencil[7:8] = x-dy
    prob = DE.ODEProblem((t,x,result) -> arraymap(ode_fun, 4,2,t,x,result),stencil,(tspan[1],tspan[end]))
    sol = DE.solve(prob,DE.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=1.e-3,abstol=1.e-3).u
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

function evolCGTensor(x::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,myfun,G::Tensors.SymmetricTensor{2,2,Float64,3})
    dx = [δ,0]; dy = [0,δ]; Tspan = (tspan[1],tspan[end])
    prob1 = DE.ODEProblem(myfun,x+dx,Tspan)
    prob2 = DE.ODEProblem(myfun,x-dx,Tspan)
    prob3 = DE.ODEProblem(myfun,x+dy,Tspan)
    prob4 = DE.ODEProblem(myfun,x-dy,Tspan)
    sol1 = DE.solve(prob1,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol2 = DE.solve(prob2,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol3 = DE.solve(prob3,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol4 = DE.solve(prob4,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    DF = Tensor{2,2}.(hcat.((sol1-sol2)/2δ,(sol3-sol4)/2δ))
    DF = SymmetricTensor{2,2}.([transpose(D)⋅G⋅D for D in DF])
end

function evolDiffTensor(x::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,myfun,DT::Tensors.SymmetricTensor{2,2,Float64,3})
    dx = [δ,0]; dy = [0,δ]; Tspan = (tspan[1],tspan[end])
    prob1 = DE.ODEProblem(myfun,x+dx,Tspan)
    prob2 = DE.ODEProblem(myfun,x-dx,Tspan)
    prob3 = DE.ODEProblem(myfun,x+dy,Tspan)
    prob4 = DE.ODEProblem(myfun,x-dy,Tspan)
    sol1 = DE.solve(prob1,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol2 = DE.solve(prob2,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol3 = DE.solve(prob3,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol4 = DE.solve(prob4,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    DF = Tensor{2,2}.(hcat.((sol1-sol2)/2δ,(sol3-sol4)/2δ))
    DF = inv.(DF)
    DF = SymmetricTensor{2,2}.([D⋅DT⋅transpose(D) for D in DF])
end
