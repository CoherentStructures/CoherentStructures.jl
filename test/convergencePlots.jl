#(c) 2018 Nathanael Schilling

t = makeDoubleGyreTestCase()
t = makeOceanFlowTestCase()
l = experimentResult(t,"regular triangular grid",(25,25), :CG)
runExperiment!(l)
l.experiment.ode_fun

plotExperiment(l,right_margin=0Plots.px,left_margin=0Plots.px,border=0Plots.px)

results = testDoubleGyre()
buildStatistics!(results,1)
Plots.gr()
#x = [Float64(x.ctx.m) for x in results[2:end]]
y = [x.runtime for x in results[2:end]]
#y = [Float64(x.statistics["L2-errors"][2]) for x in results[2:end]]
#y = x
x = exp.(linspace(1.,100,length(x)-1))
gridtypes=[x.ctx.gridType for x in results[2:end]]
import Plots
#Plots.scatter(x,y,group=gridtypes,xlabel="Number of Elements",ylabel="error",
#  xscale=:log10,yscale=:log10,legend=:bottom,ylim=(1.e-4,1e2))

Plots.scatter(x,y,group=gridtypes,xlabel="Number of Elements",ylabel="error",
  xscale=:log10,legend=:bottom)

import PyPlot
Plots.pyplot()
for i in results[2:end]
  title(i.ctx.gridType)
  plot_u(i.ctx, i.statistics["errors"][3])
  sleep(1)
end
