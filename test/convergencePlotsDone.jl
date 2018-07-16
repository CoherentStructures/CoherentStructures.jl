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
#indexes_to_plot = [i for i in 2:length(results) if results[i].mode == :CG]
indexes_to_plot = [i for i in 2:length(results)]
begin
  whichev = 2
  x = [getH(x.ctx) for x in results[indexes_to_plot]]
  y = [abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
  legendlabels = [x * @sprintf("(%.1f)",ev_slopes[x] ) for x in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
  colors = [x.mode == :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in results[indexes_to_plot]]
  Plots.scatter(x,y,group=gridtypes,label=legendlabels,color=colors,xlabel="Mesh width",ylabel="Relative error",m=ms,
    xscale=:log10,yscale=:log10,  legend=(0.40,0.35),
    ylim=(1e-10,1),xlim=(10^-1.5,10));
  loglogslopeline(x,y,gridtypes,ev_slopes)
  #loglogleastsquareslines(x,y,gridtypes)
end

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMev$(whichev)errs.pdf")

begin
  x = [getH(x.ctx) for x in results[indexes_to_plot]]
  y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(results[reference_index].statistics["B"]))[2:3,2:3]))))  for x in results[indexes_to_plot]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
  mycolors = [method_colors[f] for f in gridtypes]
  legendlabels = [x * @sprintf("(%.1f)",ev_slopes[x] ) for x in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
  legendlabels = [x * @sprintf("(%.1f)",evec_slopes[x] ) for x in gridtypes]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenspace error",
    xscale=:log10, yscale=:log10, legend=(0.45,0.3),color=mycolors,lab=legendlabels,m=ms,
    ylim=(1.e-7,1e-1),xlim=(10^(-1.5),5))
    #loglogleastsquareslines(x,y,gridtypes)
  loglogslopeline(x,y,gridtypes,evec_slopes)
end


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMevec23.pdf")



results0dg = testDoubleGyre([],quadrature_order=5,tf=0.25,run_reference=true)
results1dg = testDoubleGyre(experimentRange,run_reference=false,quadrature_order=5,tf=0.25)
results3dg = testDoubleGyre(experimentRange[2:end],mode=:naTO,run_reference=false,quadrature_order=2,tf=0.25)
results4dg = testDoubleGyre(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4,tf=0.25)

resultsdg = copy(results0dg)
append!(resultsdg,results1dg)
append!(resultsdg,results3dg)
append!(resultsdg,results4dg)
buildStatistics!(resultsdg,1)


#fd = open("DG25","w")
#serialize(fd,resultsdg)
#close(fd)


resultsdg=open(deserialize,"DG25")
#indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode == :CG]
indexes_to_plotdg = 2:length(resultsdg)
reference_indexdg = 1

#indexes_to_plot = find(x->x.mode == :CG && (getH(x.ctx) < 1e-2), resultsdg)
#deleteat!(indexes_to_plot,1)


begin
  whichev = 2
  x = [getH(x.ctx) for x in resultsdg[indexes_to_plotdg]]
  #y = [abs(x.λ[whichev] - resultsdg[reference_indexdg].λ[whichev])/(resultsdg[reference_index].λ[whichev])  for x in resultsdg[indexes_to_plot]]
  y = [abs(x.λ[whichev] - resultsdg[reference_indexdg].λ[whichev])/(resultsdg[reference_indexdg].λ[whichev])  for x in resultsdg[indexes_to_plotdg]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg[indexes_to_plotdg]]
  mycolors = [method_colors[f] for f in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg[indexes_to_plotdg] ]
  legendlabels = [x * @sprintf("(%.1f)",ev_slopes[x] ) for x in gridtypes]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative eigenvalue error",m =ms,
    xscale=:log10,yscale=:log10,color=mycolors,label=legendlabels,
    legend=(0.33,0.35),ylim=(1e-6,1e0),xlim=(10^-2.4,1e-0))
  #loglogleastsquareslines(x,y,gridtypes)
  loglogslopeline(x,y,gridtypes,ev_slopes)
end

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25ev$(whichev)errs.pdf")



begin
  whichev=2
  x = [getH(x.ctx) for x in resultsdg[indexes_to_plotdg]]
  y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg[reference_indexdg].statistics["B"]))[whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
  #y = [max(1e-10,sqrt(abs(1 - abs(x.statistics["E"][whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg[indexes_to_plotdg]]
  mycolors = [method_colors[f] for f in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg[indexes_to_plotdg] ]
  legendlabels = [x * @sprintf("(%.1f)",evec_slopes[x] ) for x in gridtypes]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
    xscale=:log10, yscale=:log10, legend=(0.30,0.30),m=ms,label=legendlabels,color=mycolors,
    ylim=(10^(-5.5),10^-0.8),xlim=(10^-2.3,10^-0.5))
    #loglogleastsquareslines(x,y,gridtypes)
  loglogslopeline(x,y,gridtypes,evec_slopes)
end


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25evec$(whichev)errs.pdf")





resultssl1 = testStaticLaplace(experimentRange,quadrature_order=2)
resultssl3 = testStaticLaplace(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
resultssl4 = testStaticLaplace(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)


resultssl = copy(resultssl1)
append!(resultssl,resultssl3)
append!(resultssl,resultssl4)
buildStatistics!(resultssl,1)


#fd = open("SL","w")
#serialize(fd,resultssl)
#close(fd)

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


reference_indexdg1 = 1
resultsdg1=open(deserialize,"DG10")
indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && getH(resultsdg1[i].ctx) < 10^(-1.)]


begin
  whichev = 2
  x = [getH(x.ctx) for x in resultsdg1[indexes_to_plotdg1]]
  #y = [abs(x.λ[whichev] - resultsdg1[reference_indexdg1].λ[whichev])/(resultsdg1[reference_index].λ[whichev])  for x in resultsdg1[indexes_to_plot]]
  y = [abs(x.λ[whichev] - resultsdg1[reference_indexdg1].λ[whichev])/(resultsdg1[reference_indexdg1].λ[whichev])  for x in resultsdg1[indexes_to_plotdg1]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg1[indexes_to_plotdg1]]
  mycolors = [method_colors[f] for f in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg1[indexes_to_plotdg1] ]
  legendlabels = [x * @sprintf("(%.1f)",ev_slopes[x] ) for x in gridtypes]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative eigenvalue error",m =ms,
    xscale=:log10,yscale=:log10,color=mycolors,label=legendlabels,
    legend=(0.33,0.35),ylim=(1e-3,1),xlim=(10^-2.2,1e-1))
  loglogslopeline(x,y,gridtypes,ev_slopes)
end

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG10ev$(whichev)errs.pdf")



begin
  whichev=2
  x = [getH(x.ctx) for x in resultsdg1[indexes_to_plotdg1]]
  y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg1[reference_indexdg1].statistics["B"]))[whichev,whichev]))))  for x in resultsdg1[indexes_to_plotdg1]]
  #y = [max(1e-10,sqrt(abs(1 - abs(x.statistics["E"][whichev,whichev]))))  for x in resultsdg1[indexes_to_plotdg1]]
  gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg1[indexes_to_plotdg1]]
  mycolors = [method_colors[f] for f in gridtypes]
  ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg1[indexes_to_plotdg1] ]
  legendlabels = [x * @sprintf("(%.1f)",evec_slopes2[x] ) for x in gridtypes]
  Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
    xscale=:log10, yscale=:log10, legend=(0.60,0.30),m=ms,label=legendlabels,color=mycolors,
    ylim=(10^(-2.5),1),xlim=(10^-2.2,10^-1.))
    #loglogleastsquareslines(x,y,gridtypes)
  loglogslopeline(x,y,gridtypes,evec_slopes2)
end


Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG10evec$(whichev)errs.pdf")



oceanTC = makeOceanFlowTestCase()

oceanCtx = regularTriangularGrid((50,25),oceanTC.LL,oceanTC.UR,quadrature_order=2)
oceanEx1 = experimentResult(oceanTC,oceanCtx,:CG,tolerance=1e-5)
runExperiment!(oceanEx1)
Plots.display(
  Plots.plot([
    plot_u(oceanEx1.ctx,oceanEx1.V[:,i],100,100)
    for i in 1:3]...)
      )

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/oceanCGP1.pdf")

oceanCtxP2 = regularP2TriangularGrid((50,25),oceanTC.LL,oceanTC.UR,quadrature_order=2)
oceanEx1P2 = experimentResult(oceanTC,oceanCtxP2,:CG,tolerance=1e-5)
runExperiment!(oceanEx1P2)
Plots.display(
    Plots.plot([
    plot_u(oceanEx1P2.ctx,oceanEx1P2.V[:,i],100,100)
    for i in 1:3]...)
    )

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/oceanCGP2.pdf")
