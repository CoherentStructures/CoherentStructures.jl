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
    sol1 = DE.solve(prob1,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol2 = DE.solve(prob2,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol3 = DE.solve(prob3,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    sol4 = DE.solve(prob4,DE.Vern9(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-12,abstol=1e-12).u
    # DF = Tensor{2,2}.([ForwardDiff.jacobian(x -> flow(myfun,x,(tspan[1], t)),initval) for t in tspan[2:end]])
    DF = Tensor{2,2}.(hcat.((sol1-sol2)/2δ,(sol3-sol4)/2δ))
    DF = SymmetricTensor{2,2}.([transpose(D)⋅G⋅D for D in DF])
    return mean(DF)
end

function avDiffTensor(x::Vector{Float64},tspan::Vector{Float64},
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
    return mean(DF)
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
