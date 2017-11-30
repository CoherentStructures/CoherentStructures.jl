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

function arraymap(t,x,result)
    howmanytimes=4
    basesize=2
    for i in 1:howmanytimes
        result[1+(i-1)*basesize: i*basesize] =  rot_double_gyre2(t,x[ 1 + (i-1)*basesize:  i*basesize],result[1 + (i - 1)*basesize: i*basesize])
    end
    return result
end

function set1(a)
    a[1] = 1.0
end

Id = SymmetricTensor{2,2}(eye(2,2))
function avDiffTensor(x::Vec{2,Float64},tspan,
    δ::Float64,myfun)
    dx = [δ,0]; dy = [0,δ];
    stencil = zeros(1,8)
    stencil[1:2] = x+dx
    stencil[3:4] = x+dy
    stencil[5:6] = x-dx
    stencil[7:8] = x-dy
    prob = DE.ODEProblem(arraymap,stencil,(tspan[1],tspan[end]))
    sol = DE.solve(prob,DE.BS5(),saveat=tspan,save_everystep=false,dense=false,reltol=1.e-3,abstol=1.e-3).u
    DF = zeros(Tensor{2,2},length(tspan))
    for i in 1:length(tspan)
        sol[i][1:2] -= sol[i][5:6]
        sol[i][3:4] -= sol[i][7:8]
        DF[i] = transpose(Tensor{2,2}(sol[i][1:4])/2δ) #TODO: Figure out why the transpose is neccessary...
    end
    return mean([SymmetricTensor{2,2}(A'⋅A) for A in inv.(DF)])
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
