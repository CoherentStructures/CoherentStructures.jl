#(c) 2018 Nathanael Schilling

#This file contains plotting routines for plotting convergence plots
#It is not in plotting.jl so as to avoid having to load Plots.jl in plotting.jl
#This is because loading Plots.jl is slow, and GR is fast.
#Plots.jl is used here because GR is poorly documented and I couldn't figure out how
#to use it

import Plots
include("numericalExperiments.jl")

function plotNumericalExperiment(x,y,bins)

end



results = testDoubleGyre()
buildStatistics!(results,1)
x = [Float64(x.ctx.n) for x in results[2:end]]
y = [Float64(x.statistics["λ-errors"][2]) for x in results[2:end]]
gridtypes=[x.ctx.gridType for x in results[2:end]]
Plots.scatter(x,y,group=gridtypes,xlabel="n",ylabel="error",xscale=:log,yscale=:log)
