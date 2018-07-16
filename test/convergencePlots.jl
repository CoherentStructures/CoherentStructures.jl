#(c) 2018 Nathanael Schilling

using CoherentStructures
include("numericalExperiments.jl")


LL = Vec{2}([0.0,0.0])
UR=Vec{2}([2π,2π])
bdata_predicate = (x,y) -> (CoherentStructures.distmod(x[1],y[1],2π) < 1e-9 && CoherentStructures.distmod(x[2],y[2],2π)<1e-9)
tC = makeStandardMapTestCase()


#tC = makeStandardMapTestCase()
tC = makeDoubleGyreTestCase(0.25)

tC = makeCylinderFlowTestCase()
ctx = regularTriangularGrid((10,10),tC.LL,tC.UR,quadrature_order=2)
eR = experimentResult(tC, ctx, :naTO)
runExperiment!(eR)
plotExperiment(eR)




tC = makeOceanFlowTestCase()
tC = makeStandardMapTestCase()

tC = makeDoubleGyreTestCase(0.25)

#ctx = regularP2TriangularGrid((10,10),tC.LL,tC.UR,quadrature_order=4)
ctx = regularTriangularGrid((10,10),tC.LL,tC.UR,quadrature_order=2)
M = assembleMassMatrix(ctx,bdata=eR.bdata)
S = assembleStiffnessMatrix(ctx,bdata=eR.bdata)

backwards_flow = u0->flow(tC.ode_fun, u0,[0.25,0.])[end]
ALPHA = nonAdaptiveTO(ctx,backwards_flow)
R = -0.5*(S + ALPHA'*S*ALPHA)
R = 0.5(R + R')
R = S + ALPHA'*S*ALPHA
R = R + R'
maximum(R - R')
λ
plot_u(ctx,v[:,6])

eR = experimentResult(tC, ctx, :naTO)
runExperiment!(eR)
plotExperiment(eR)

plot_u(ctx,ones(ctx.n),400,400,color=:rainbow,colorbar=true,clim=(1-1e-16,1))
Plots.savefig("/tmp/output.svg")
inv_flow_map = CoherentStructures.standardMapInv
plot_u_eulerian(ctx, -1*eR.V[:,3],inv_flow_map,
   LL,UR,300,300,
   bdata=eR.bdata,color=:rainbow)






tC = makeStandardMapTestCase()
referenceCtx = regularP2TriangularGrid((200,200),tC.LL,tC.UR)
reference = experimentResult(tC,referenceCtx, :CG)
runExperiment!(reference)
inv_flow_map = CoherentStructures.standardMapInv


results[1].ctx.numberOfPointsInEachDirection
plot_u(results[1].ctx, -1.0*results[1].V[:,2], bdata=results[1].bdata,300,300,color=:rainbow)
plot_u_eulerian(results[1].ctx, -1.0*results[1].V[:,2],inv_flow_map,
   results[1].ctx.spatialBounds[1], results[1].ctx.spatialBounds[2],300,300,
   bdata=results[1].bdata,color=:rainbow)
plot_real_spectrum(reference.λ)

##### Test case for Standard map


results2 = testStandardMap([109,108,107,106])
buildStatistics!(results2,1)
deleteat!(results2,1)
append!(results,results2)

results[42].ctx.numberOfPointsInEachDirection[1]
deleteat!(results,42)
length(results)-6

results = testStandardMap(10:10:200)
buildStatistics!(results,1)
reference_index = 1
fd = open("SM","w")
serialize(fd,results)
close(fd)

dgTc = makeDoubleGyreTestCase(0.5)
referenceCtx = regularP2TriangularGrid( (300,300), dgTc.LL,dgTc.UR)
reference = experimentResult(dgTc,referenceCtx,:CG)
runExperiment!(reference)
plot_u(reference.ctx,reference.V[:,2])


CoherentStructures.buildStatistics!(results,22)

results = CoherentStructures.testDoubleGyre(0.25)
CoherentStructures.buildStatistics!(results,1)
fd = open("DG025","w")
serialize(fd,results)
close(fd)


reference_index = 1
results=open(deserialize,"DG025")


reference_index = 1
results=open(deserialize,"SM")

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
  #Plots.display(plot_u(results[i].ctx,results[i].V[:,2],title="$i",bdata=results[i].bdata))
  Plots.display(plot_real_spectrum(results[i].λ))
  sleep(1)
end


whichev=2
y = [(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
begin
  whichev = 2
  x = [getH(x.ctx) for x in results[indexes_to_plot]]
  y = [-abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
  gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative error",
    xscale=:log10,yscale=:log10,  legend=:bottomright,
    title="Errors in eigenvalue $whichev")
  loglogleastsquareslines(x,y,gridtypes)
end
begin



Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/ev1errs.pdf")


t = [1 - norm(x.statistics["E"][2:3,2:3]) for x in results[indexes_to_plot]]

#Plot second eigenvector errors
begin
  #whichev = 2
  x = [getH(x.ctx) for x in results[indexes_to_plot]]
  #y = [max(1e-10,sqrt(1 - abs(x.statistics["E"][whichev,whichev])))  for x in results[indexes_to_plot]]
  y = [max(1e-10,sqrt(abs(1 - norm(x.statistics["E"][2:3,2:3]))))  for x in results[indexes_to_plot]]
  gridtypes = [x.ctx.gridType for x in results[indexes_to_plot]]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
    xscale=:log10, yscale=:log10, legend=:bottomright,
    title="Error in First two eigenfunctions")
  loglogleastsquareslines(x,y,gridtypes)
end
index = 32
plot_u(results[index].ctx,results[index].V[:,3],bdata=results[index].bdata)
plot_u(results[29].ctx,results[29].V[:,3],bdata=results[29].bdata)
results[32].ctx.numberOfPointsInEachDirection
1 - norm(results[30].statistics["E"][2:3,2:3])
1 - norm(results[31].statistics["E"][2:3,2:3])
1 - norm(results[32].statistics["E"][2:3,2:3])
1 - norm(results[33].statistics["E"][2:3,2:3])
results[31].statistics["E"][2:3,2:3]


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/evec2errs.pdf")

plot_u_eulerian(results[index].ctx,results[index].V[:,2],CoherentStructures.standardMapInv,
    [0.,0.0],[2π,2π],200,200, bdata=results[index].bdata)
