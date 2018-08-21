#(c) 2018 Nathanael Schilling
#Plots for paper

using CoherentStructures
include("numericalExperiments.jl")
experimentRange=Int.(round.(10*1.3.^range(0,13)))
experimentRangeSmall=Int.(round.(10*1.3.^range(0,7)))

results0 = testStandardMap([],quadrature_order=5,run_reference=true)
results1 = testStandardMap(experimentRange,quadrature_order=2,run_reference=false)
results3 = testStandardMap(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
results4 = testStandardMap(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)

results = copy(results0)
append!(results,results1)
append!(results,results3)
append!(results,results4)
buildStatistics!(results,1)


#fd = open("SM","w")
#serialize(fd,results)
#close(fd)

reference_index = 1
results=open(deserialize,"SM")
gc()
#indexes_to_plot = [i for i in 2:length(results) if results[i].mode == :CG]
for j in [1,2]
  if j == 1
    indexes_to_plot = [i for i in 2:length(results) if results[i].mode == :CG]
  else
    indexes_to_plot = [i for i in 2:length(results) if results[i].mode != :CG]
  end
  begin
    whichev = 2
    x = [getH(x.ctx) for x in results[indexes_to_plot]]
    y = [abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
      end
    else
      legendlabels = [contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
      end
    end
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
    colors = [x.mode == :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in results[indexes_to_plot]]
    Plots.scatter(x,y,group=gridtypes,label=legendlabels,color=colors,xlabel="Mesh width",ylabel="Relative error",m=ms,
      xscale=:log10,yscale=:log10,  legend=(0.40,0.20),
      ylim=(1e-10,1),xlim=(10^-1.5,1.03));
    Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes))
    #loglogleastsquareslines(x,y,gridtypes)
  end
  if j == 1
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMev$(whichev)errsCG.pdf")
  else
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMev$(whichev)errsTO.pdf")
  end

  begin
    x = [getH(x.ctx) for x in results[indexes_to_plot]]
    y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(results[reference_index].statistics["B"]))[2:3,2:3]))))  for x in results[indexes_to_plot]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
    mycolors = [method_colors[f] for f in gridtypes]
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
      end
    else
      legendlabels = [contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
      end
    end
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenspace error",
      xscale=:log10, yscale=:log10, legend=(0.55,0.25),color=mycolors,lab=legendlabels,m=ms,
      ylim=(1.e-7,1e-1),xlim=(10^(-1.5),1.03))
      #loglogleastsquareslines(x,y,gridtypes)
    Plots.display(loglogslopeline(x,y,gridtypes,evec_slopes))
    sleep(2)
  end

  if j == 1
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMevec23errsCG.pdf")
  else
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMevec23errsTO.pdf")
  end
end





results0dg = testDoubleGyre([],quadrature_order=5,tf=0.25,run_reference=true)
results1dg = testDoubleGyre(experimentRange,run_reference=false,quadrature_order=5,tf=0.25)
results3dg = testDoubleGyre(experimentRange[2:end],mode=:naTO,run_reference=false,quadrature_order=2,tf=0.25)
results4dg = testDoubleGyre(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4,tf=0.25)

resultsdg = copy(results0dg)
append!(resultsdg,results1dg)
append!(resultsdg,results3dg)
append!(resultsdg,results4dg)
buildStatistics!(resultsdg,1)


fd = open("DG25","w")
serialize(fd,resultsdg)
close(fd)
gc()


resultsdg=open(deserialize,"DG25")
reference_indexdg = 1

for j in [1,2]
  if j == 1
    indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode == :CG]
  else
    indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode != :CG]
  end
  begin
    whichev = 2
    x = [getH(x.ctx) for x in resultsdg[indexes_to_plotdg]]
    #y = [abs(x.λ[whichev] - resultsdg[reference_indexdg].λ[whichev])/(resultsdg[reference_index].λ[whichev])  for x in resultsdg[indexes_to_plot]]
    y = [abs(x.λ[whichev] - resultsdg[reference_indexdg].λ[whichev])/(resultsdg[reference_indexdg].λ[whichev])  for x in resultsdg[indexes_to_plotdg]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg[indexes_to_plotdg]]
    mycolors = [method_colors[f] for f in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg[indexes_to_plotdg] ]
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
      end
    else
      legendlabels = [contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements" for x in resultsdg[indexes_to_plotdg]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
      end
    end
    if j == 1
      ylim = (1e-6,10^(-0.4))
    else
      ylim = (1e-3,10^(-0.4))
    end
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative eigenvalue error",m =ms,
      xscale=:log10,yscale=:log10,color=mycolors,label=legendlabels,
      legend=(0.43,0.35),ylim=ylim,xlim=(10^-2.4,10^-0.7))
    #loglogleastsquareslines(x,y,gridtypes)
    Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes))
    sleep(2)
  end
  if j == 1
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25ev$(whichev)errsCG.pdf")
  else
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25ev$(whichev)errsTO.pdf")
  end
  begin
    whichev=2
    x = [getH(x.ctx) for x in resultsdg[indexes_to_plotdg]]
    y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg[reference_indexdg].statistics["B"]))[whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
    #y = [max(1e-10,sqrt(abs(1 - abs(x.statistics["E"][whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg[indexes_to_plotdg]]
    mycolors = [method_colors[f] for f in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg[indexes_to_plotdg] ]
    if j == 1
      slopes = evec_slopes
    else
      slopes = evec_slopes2
    end
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",slopes[gridtypes[i]] )
      end
    else
      legendlabels = [(contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements") for x in resultsdg[indexes_to_plotdg]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",slopes[gridtypes[i]] )
      end
    end
    if j == 1
      ylim = (10^(-5.5),10^-0.8)
    else
      ylim = (10^(-2.5),10^-0.8)
    end
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
      xscale=:log10, yscale=:log10, legend=(0.30,0.20),m=ms,label=legendlabels,color=mycolors,
      ylim=ylim,xlim=(10^-2.3,10^-0.5))
      #loglogleastsquareslines(x,y,gridtypes)
    Plots.display(loglogslopeline(x,y,gridtypes,slopes))
    sleep(2)
  end

  if j == 1
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25evec$(whichev)errsCG.pdf")
  else
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25evec$(whichev)errsTO.pdf")
  end

end





resultssl1 = testStaticLaplace(experimentRange,quadrature_order=2)
resultssl3 = testStaticLaplace(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
resultssl4 = testStaticLaplace(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)


resultssl = copy(resultssl1)
append!(resultssl,resultssl3)
append!(resultssl,resultssl4)
buildStatistics!(resultssl,1)


fd = open("SL","w")
serialize(fd,resultssl)
close(fd)
gc()

reference_indexsl = 1
resultssl=open(deserialize,"SL")
indexes_to_plotsl = 2:length(resultssl)
indexes_to_plotsl = [i for i in 2:length(resultssl) if resultssl[i].mode == :CG]


begin
  whichev = 2
  x = [getH(x.ctx) for x in resultssl[indexes_to_plotsl]]
  #y = [abs(x.λ[whichev] - resultssl[reference_indexsl].λ[whichev]/(resultssl[reference_indexsl].λ[whichev]))  for x in resultssl[indexes_to_plotsl]]
  y = [abs((x.λ[whichev] - resultssl[reference_indexsl].λ[whichev])/resultssl[reference_indexsl].λ[whichev])  for x in resultssl[indexes_to_plotsl]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultssl[indexes_to_plotsl]]
  legendlabels = [x * @sprintf("(%.1f)",ev_slopes[x] ) for x in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultssl[indexes_to_plotsl] ]
  colors = [x.mode == :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in resultssl[indexes_to_plotsl]]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative error",
    xscale=:log10,yscale=:log10,legend=:best,m=ms, color=colors,xlim=(1e-3,10),ylim=(1e-10,1),
    title="Errors in eigenvalue $whichev")
  #loglogleastsquareslines(x,y,gridtypes)
  loglogslopeline(x,y,gridtypes,ev_slopes)
end

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SLev$(whichev)errs.pdf")

begin
  x = [getH(x.ctx) for x in resultssl[indexes_to_plotsl]]
  #y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultssl[reference_index].statistics["B"]))[2:3,2:3]))))  for x in resultssl[indexes_to_plotsl]]
  y = [max(1e-10,sqrt(abs(1 - norm((x.statistics["E"])[2:3,2:3]))))  for x in resultssl[indexes_to_plotsl]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultssl[indexes_to_plotsl]]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
    xscale=:log10, yscale=:log10,legend=:none,# legend=(0.6,0.6),
    title="Error in Eigenspace {2,3}",ylim=(1.e-10,1e-1),xlim=(1e-3,10.))
    #loglogleastsquareslines(x,y,gridtypes)
end


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SLevec$(whichev)errs.pdf")



results0dg1 = testDoubleGyre([],quadrature_order=5)
results1dg1 = testDoubleGyre(experimentRange[2:end-1],quadrature_order=5,run_reference=false)
results3dg1 = testDoubleGyre(experimentRange[2:end-1],mode=:naTO,run_reference=false,quadrature_order=2)
results4dg1 = testDoubleGyre(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)

resultsdg1 = copy(results0dg1)
append!(resultsdg1,results1dg1)
append!(resultsdg1,results3dg1)
append!(resultsdg1,results4dg1)
buildStatistics!(resultsdg1,1)


#fd = open("DG10","w")
#serialize(fd,resultsdg1)
#close(fd)
#gc()


reference_indexdg1 = 1
resultsdg1=open(deserialize,"DG10")
for j in [1,2]
  #indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && getH(resultsdg1[i].ctx) < 10^(-1.)]
  if j == 1
    indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && resultsdg1[i].mode == :CG]
  else
    indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && resultsdg1[i].mode != :CG]
  end


  begin
    whichev = 2
    x = [getH(x.ctx) for x in resultsdg1[indexes_to_plotdg1]]
    #y = [abs(x.λ[whichev] - resultsdg1[reference_indexdg1].λ[whichev])/(resultsdg1[reference_index].λ[whichev])  for x in resultsdg1[indexes_to_plot]]
    y = [abs(x.λ[whichev] - resultsdg1[reference_indexdg1].λ[whichev])/(resultsdg1[reference_indexdg1].λ[whichev])  for x in resultsdg1[indexes_to_plotdg1]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg1[indexes_to_plotdg1]]
    mycolors = [method_colors[f] for f in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg1[indexes_to_plotdg1] ]
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg1[indexes_to_plotdg1]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
      end
    else
      legendlabels = [(contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements") for x in resultsdg1[indexes_to_plotdg1]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
      end
    end
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative eigenvalue error",m =ms,
      xscale=:log10,yscale=:log10,color=mycolors,label=legendlabels,
      legend=(0.33,0.25),ylim=(10^-2.5,1),xlim=(10^-2.2,1e-1))
    Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes))
    sleep(1)
  end

  if j == 1
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG10ev$(whichev)errsCG.pdf")
  else
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG10ev$(whichev)errsTO.pdf")
  end

  begin
    whichev=2
    x = [getH(x.ctx) for x in resultsdg1[indexes_to_plotdg1]]
    y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg1[reference_indexdg1].statistics["B"]))[whichev,whichev]))))  for x in resultsdg1[indexes_to_plotdg1]]
    #y = [max(1e-10,sqrt(abs(1 - abs(x.statistics["E"][whichev,whichev]))))  for x in resultsdg1[indexes_to_plotdg1]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg1[indexes_to_plotdg1]]
    mycolors = [method_colors[f] for f in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg1[indexes_to_plotdg1] ]
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg1[indexes_to_plotdg1]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",evec_slopes2[gridtypes[i]] )
      end
    else
      legendlabels = [(contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements") for x in resultsdg1[indexes_to_plotdg1]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes2[gridtypes[i]] )
      end
    end
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
      xscale=:log10, yscale=:log10, legend=(0.55,0.30),m=ms,label=legendlabels,color=mycolors,
      ylim=(10^(-2.5),1),xlim=(10^-2.2,10^-0.7))
      #loglogleastsquareslines(x,y,gridtypes)
    Plots.display(loglogslopeline(x,y,gridtypes,evec_slopes2))
  end


  if j == 1
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG10evec$(whichev)errsCG.pdf")
  else
    Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG10evec$(whichev)errsTO.pdf")
  end
end



using Clustering
function iterated_kmeans(numiterations,args...)
    best = kmeans(args...)
    for i in 1:(numiterations-1)
        cur = kmeans(args...)
        if cur.totalcost < best.totalcost
            print("Improved")
            best = cur
        end
    end
    return best
end
oceanTC = makeOceanFlowTestCase()

begin
  oceanCtx = regularTriangularGrid((75,45),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  oceanEx1 = experimentResult(oceanTC,oceanCtx,:CG,tolerance=1e-5)
  runExperiment!(oceanEx1)


  ctx2 = regularTriangularGrid((200,200),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  Vp1 = zeros(ctx2.n,6)
  for i in 1:6
    Vp1[:,i] = sampleTo(undoBCS(oceanCtx, oceanEx1.V[:,i],oceanEx1.bdata), oceanCtx,ctx2)
  end

  n_partition = 4
  res = iterated_kmeans(100,Vp1[:,1:(n_partition-1)]',n_partition)
  up1 = kmeansresult2LCS(res)
end
clusterplotp1 = plot_u(ctx2,sum([-i*up1[:,i] for i in 1:n_partition]),200,200,colorbar=:none)


begin
  oceanCtxP2 = regularP2TriangularGrid((25,15),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  oceanEx1P2 = experimentResult(oceanTC,oceanCtxP2,:CG,tolerance=1e-5)
  runExperiment!(oceanEx1P2)

  ctx2 = regularTriangularGrid((200,200),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  V = zeros(ctx2.n,6)
  for i in 1:6
    V[:,i] = sampleTo(undoBCS(oceanCtxP2, oceanEx1P2.V[:,i],oceanEx1P2.bdata), oceanCtxP2,ctx2)
  end


  n_partition = 4
  resP2 = iterated_kmeans(100,V[:,1:(n_partition-1)]',n_partition)
  uP2 = kmeansresult2LCS(resP2)
end

clusterplotp2 = plot_u(ctx2,sum([-i*uP2[:,i] for i in 1:n_partition]),200,200,colorbar=:none)

plot_u(ctx2,uP2[:,4],200,200)

plot_u(oceanCtxP2,oceanEx1P2.V[:,3],100,100)

plot_u(oceanEx1P2.ctx,uP2[:,1],100,100)
plot_u(oceanEx1P2.ctx,uP2[:,2],100,100)
plot_u(oceanEx1P2.ctx,uP2[:,3],100,100)
plot_u(oceanEx1P2.ctx,uP2[:,4],100,100)

plots1 = [plot_u(oceanEx1.ctx,oceanEx1.V[:,i],100,100,colorbar=false) for i in 1:3]
plots2 = [plot_u(oceanEx1P2.ctx,oceanEx1P2.V[:,i],100,100,colorbar=false) for i in 1:3]
plots3 = clusterplotp1
plots4 = clusterplotp2

Plots.plot(plots1[1], plots2[1],plots1[2],plots2[2],plots1[3],plots2[3],plots3,plots4,layout=(4,2),margin=-5Plots.px)
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/oceansresultCG.pdf")




function bickleyInvFlow(u0)
  LL = [0.0,-3.0]; UR=[6.371π,3.0]
  tmp =Tensors.Vec{2}(flow(bickleyJet,u0,[tf,t0]; tolerance=1e-8)[end])
  return Tensors.Vec{2}((mod(tmp[1],UR[1]),tmp[2]))
end

ctx.m
ctx.n
length(ctx.quadrature_points)

begin
  bickleyto_results = []
  for j in  [1,2]
    LL = [0.0,-3.0]; UR=[6.371π,3.0]
    if j == 1
      ctx = regularTriangularGrid((60,20),LL,UR,quadrature_order=2)
    else
      ctx = regularP2TriangularGrid((30,10),LL,UR,quadrature_order=2)
    end

    predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], UR[1]) < 1e-10
    bdata = CoherentStructures.boundaryData(ctx,predicate,[]);

    tf = 40*3600*24.0
    t0 = 0.0
    ALPHAS = nonAdaptiveTO(ctx,bickleyInvFlow)
    ALPHASBC = applyBCS(ctx,ALPHAS,bdata)
    K = assembleStiffnessMatrix(ctx)
    M = assembleMassMatrix(ctx,bdata=bdata)
    DL = 0.5(K + ALPHAS'*K*ALPHAS)
    DL = applyBCS(ctx,DL,bdata)
    DL = 0.5(DL + DL')

    print("Finished setting up matrices")
    λ,v = eigs(DL,M,which=:SM,nev=10)

    print("Solved eigenproblem")
    ctx2 = regularP2TriangularGrid((200,60),LL,UR,quadrature_order=1)
    V = zeros(ctx2.n,10)
    for i in 1:10
      V[:,i] = sampleTo(undoBCS(ctx, v[:,i],bdata), ctx,ctx2)
    end
    print("Clustering...")

    n_partition = 8
    res = iterated_kmeans(20,V[:,2:n_partition]',n_partition)
    u = kmeansresult2LCS(res)
    u_combined = sum([u[:,i]*i for i in 1:n_partition])
    res = plot_u(ctx2, u_combined,200,200,
        color=:rainbow,colorbar=:none)
      push!(bickleyto_results,res)
  end
end
Plots.pdf(bickleyto_results[1],"/home/nathanael/Documents/TUM/topmath/plots/bickleyaTOP1.pdf")
Plots.pdf(bickleyto_results[2],"/home/nathanael/Documents/TUM/topmath/plots/bickleyaTOP2.pdf")


plot_u(ctx,v[:,8],bdata=bdata,100,100)
plot_u(ctx2,u[:,9])
plot_u(ctx,u_combined,200,200)


bickleyPlots = []
for j in [1,2]
  LL = [0.0,-3.0]; UR=[6.371π,3.0]
  if j == 1
    ctx = regularTriangularGrid((50,40),LL,UR,quadrature_order=2)
  else
    ctx = regularP2TriangularGrid((10,8),LL,UR,quadrature_order=2)
  end
  predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 6.371π) < 1e-10
  bdata = CoherentStructures.boundaryData(ctx,predicate,[]);

  cgfun = (x -> mean(pullback_diffusion_tensor(bickleyJet, x,linspace(0.0,40*3600*24,81),
       1.e-8,tolerance=1.e-5)))

  K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
  M = assembleMassMatrix(ctx,bdata=bdata)
  λ, v = eigs(K,M,which=:SM, nev= 10)

  ctx2 = regularP2TriangularGrid((200,60),LL,UR,quadrature_order=1)
  V = zeros(ctx2.n,10)
  for i in 1:10
    V[:,i] = sampleTo(undoBCS(ctx, v[:,i],bdata), ctx,ctx2)
  end

  n_partition = 8
  res = iterated_kmeans(10,V[:,2:n_partition]',n_partition)
  u = kmeansresult2LCS(res)
  u_combined = sum([u[:,i]*i for i in 1:n_partition])
  res = plot_u(ctx2, u_combined,200,200,
      color=:rainbow,colorbar=:none)
  Plots.pdf(res,"/home/nathanael/Documents/TUM/topmath/plots/bickleyCGP$j.pdf")
  Plots.display(res)
  sleep(2)
end
