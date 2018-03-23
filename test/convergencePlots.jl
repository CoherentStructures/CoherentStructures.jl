#(c) 2018 Nathanael Schilling

using juFEMDL
import Plots

t = makeOceanFlowTestCase()
t = makeDoubleGyreTestCase()
l = experimentResult(t,"regular P2 triangular grid",(100,100), :aTO)

runExperiment!(l,12)

plotExperiment(l,axis=false,colorbar=false,margin=0.0Plots.px)

results = juFEMDL.testDoubleGyre()
juFEMDL.buildStatistics!(results,1)

results[1].λ[1]

x = [x.ctx.n for x in results[2:end]]
y = [abs(x.λ[4] - results[1].λ[4])  for x in results[2:end]]

gridtypes = [x.ctx.gridType for x in results[2:end]]
Plots.scatter(x,y,group=gridtypes,xlabel="Number of Elements",ylabel="error",
  xscale=:log10, yscale=:log10, legend=:bottom)

import PyPlot
Plots.pyplot()
for i in results[2:end]
  title(i.ctx.gridType)
  plot_u(i.ctx, i.statistics["errors"][3])
  sleep(1)
end
