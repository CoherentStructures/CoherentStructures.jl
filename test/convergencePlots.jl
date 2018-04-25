#(c) 2018 Nathanael Schilling

using CoherentStructures
import Plots

#t = makeDoubleGyreTestCase()
t = CoherentStructures.makeStaticLaplaceTestCase()
l = experimentResult(t,"regular P2 triangular grid",(100,100), :aTO)

runExperiment!(l,12)

plotExperiment(l,axis=false,colorbar=false,margin=0.0Plots.px)

results = CoherentStructures.testStaticLaplace()
CoherentStructures.buildStatistics!(results,1)

## Plot eigenvalue errors

indexes_to_plot= Int64[]
for (index,i) in enumerate(results)
  if index == 1
    continue
  end
  if i.ctx.gridType ∈ ["regular Delaunay grid", "regular P2 Delaunay grid"]
    #continue
  end
  push!(indexes_to_plot,index)
end


indexes_to_plot = 2:length(results)
x = [getH(x.ctx) for x in results[indexes_to_plot]]
y = [abs(x.λ[2] - results[1].λ[2])  for x in results[indexes_to_plot]]
gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]

Plots.scatter(x,y,group=gridtypes,xlabel="Number of Basis Functions",ylabel="Absolute error",
  xscale=:log10, yscale=:log10, legend=:bottomright,
  title="Errors in second eigenvalue")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/ev1errs.pdf")

x = [getH(x.ctx) for x in results[indexes_to_plot]]
y = [abs(x.λ[3] - results[1].λ[3])  for x in results[indexes_to_plot]]
gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
Plots.scatter(x,y,group=gridtypes,xlabel="Mesh Width",ylabel="Absolute error",
  xscale=:log10,yscale=:log10, legend=:none,
  title="Errors in third eigenvalue",ylim=(1e-16,200))
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/ev2errs.pdf")

#slope 2 line
xl = collect(linspace(minimum(x), maximum(x),100))
yl = xl.^2
legendlabel = ["Slope 2 line" for x in xl]
Plots.plot!(xl,yl,group=legendlabel)

#Slope 4 line
xl = collect(linspace(minimum(x),maximum(x),100))
yl = xl.^4
legendlabel = ["Slope 4 line" for x in xl]
Plots.plot!(xl,yl,group=legendlabel)




#Plot second eigenvector errors

x = [getH(x.ctx) for x in results[indexes_to_plot]]
y = [1 - abs(x.statistics["E"][2,2])  for x in results[indexes_to_plot]]
[x.time]
gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]

Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Error",
  xscale=:log10, yscale=:log10, legend=:best,
  title="Error in second Eigenfunction")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/evec2errs.pdf")


#Plot third eigenvector errors

y

x = [x.ctx.n for x in results[indexes_to_plot]]
y = [max(1e-19, 1 - abs(x.statistics["E"][3,3]))  for x in results[indexes_to_plot]]
gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
Plots.scatter(x,y,group=gridtypes,xlabel="Number of Basis Functions",ylabel="Error",
  xscale=:log10, yscale=:log10,ylim=(1e-7,1e3),xlim=(1e3,10e5), legend=:best,title="Error in third Eigenfunction")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/evec3errs.pdf")


x = [x.ctx.n for x in results[indexes_to_plot]]
y = [max(1e-19, 1 - abs(x.statistics["E"][4,4]))  for x in results[indexes_to_plot]]
gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
Plots.scatter(x,y,group=gridtypes,xlabel="Number of Basis Functions",ylabel="Error",
  xscale=:log10, yscale=:log10,ylim=(1e-7,1e3),xlim=(1e3,10e5), legend=:best,title="Error in fourth Eigenfunction")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/evec4errs.pdf")


x = [x.ctx.n for x in results[indexes_to_plot]]
y = [max(1e-19, 1 - abs(x.statistics["E"][5,5]))  for x in results[indexes_to_plot]]
gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
Plots.scatter(x,y,group=gridtypes,xlabel="Number of Basis Functions",ylabel="Error",
  xscale=:log10, yscale=:log10,ylim=(1e-7,1e3),xlim=(1e3,10e5), legend=:best,title="Error in fifth Eigenfunction")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/evec5errs.pdf")
