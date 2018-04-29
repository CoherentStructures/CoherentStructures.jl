#(c) 2018 Nathanael Schilling

using CoherentStructures
import Plots

plotExperiment(l,axis=false,colorbar=false,margin=0.0Plots.px)


dgTc = makeDoubleGyreTestCase(0.5)
referenceCtx = regularP2TriangularGrid( (300,300), dgTc.LL,dgTc.UR)
reference = experimentResult(dgTc,referenceCtx,:CG)
runExperiment!(reference)
CoherentStructures.buildStatistics!(results,22)

results = CoherentStructures.testDoubleGyre(0.25)
CoherentStructures.buildStatistics!(results,1)
fd = open("DG025","w")
serialize(fd,results)
close(fd)


reference_index = 1
results=open(deserialize,"DG025")

reference_index = 21
results=open(deserialize,"SL")

reference_index = 22
results = open(deserialize, "DG05")

reference_index= 43
results = open(deserialize, "DG")

results[reference_index].ctx.numberOfPointsInEachDirection[1]
results[reference_index].ctx.gridType

## Plot eigenvalue errors

begin
  indexes_to_plot= Int64[]
  for (index,i) in enumerate(results)
    if i.ctx.gridType ∈ ["regular Delaunay grid", "regular P2 Delaunay grid","regular quadrilateral grid","regular P2 quadrilateral grid"]
      continue
    end
    if index == reference_index
      continue
    end
    push!(indexes_to_plot,index)
  end
end

for i in 1:21
  Plots.display(plot_u(results[i].ctx,results[i].V[:,2],title="$i"))
  sleep(1)
end


whichev=2
y = [(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
y
begin
  whichev = 2
  x = [getH(x.ctx) for x in results[indexes_to_plot]]
  y = [-abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
  gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
  Plots.scatter(x,y,group=gridtypes,xlabel="Number of Basis Functions",ylabel="Absolute error",
    xscale=:log10,yscale=:log10,  legend=:bottomright,
    title="Errors in second eigenvalue")
  #slope 2 line
  xl = collect(linspace(minimum(x), maximum(x),100))
  yl = xl.^2*100
  legendlabel = ["Slope 2 line" for x in xl]
  Plots.plot!(xl,yl,group=legendlabel)

  #Slope 4 line
  xl = collect(linspace(minimum(x),maximum(x),100))
  yl = xl.^4*500
  legendlabel = ["Slope 4 line" for x in xl]
  #Plots.plot!(xl,yl,group=legendlabel)
  Plots.display()
end


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/ev1errs.pdf")



#Plot second eigenvector errors
begin
  whichev = 4
  x = [getH(x.ctx) for x in results[indexes_to_plot]]
  y = [max(1e-10,sqrt(1 - abs(x.statistics["E"][whichev,whichev])))  for x in results[indexes_to_plot]]
  gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
    xscale=:log10, yscale=:log10, legend=:bottomright,
    title="Error in Eigenfunction $whichev")

  xl = collect(linspace(minimum(x), maximum(x),100))
  yl = xl.^2*20
  legendlabel = ["Slope 2 line" for x in xl]
  Plots.plot!(xl,yl,group=legendlabel)

  xl = collect(linspace(minimum(x), maximum(x),100))
  yl = xl.^3*100
  legendlabel = ["Slope 3 line" for x in xl]
  Plots.plot!(xl,yl,group=legendlabel)
end
plot_u(results[1].ctx,results[1].V[:,3])
results[1].ctx.numberOfPointsInEachDirection


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/evec2errs.pdf")
