

begin
    results0dg = testDoubleGyre([],quadrature_order=8,tf=0.25,run_reference=true)
    results1dg = testDoubleGyre(experimentRange,run_reference=false,quadrature_order=8,tf=0.25)

    resultsdg = copy(results0dg)
    append!(resultsdg,results1dg)
    buildStatistics!(resultsdg,1)


    fd = open("DG25q","w")
    serialize(fd,resultsdg)
    close(fd)
    gc()
end

ctx = regularTriangularGrid(quadrature_order=5)
length(ctx.quadrature_points)/ctx.m

resultsdgh = open(deserialize, "DG25q")
length(resultsdgh[2].ctx.quadrature_points)/resultsdgh[2].ctx.m

reference_indexdgh = 1

begin
  j = 1
  if j == 1
    indexes_to_plotdgh = [i for i in 2:length(resultsdgh) if resultsdgh[i].mode == :CG]
  else
    indexes_to_plotdgh = [i for i in 2:length(resultsdgh) if resultsdgh[i].mode != :CG]
  end
  begin
    whichev = 2
    x = [getH(x.ctx) for x in resultsdgh[indexes_to_plotdgh]]
    #y = [abs(x.λ[whichev] - resultsdg[reference_indexdg].λ[whichev])/(resultsdg[reference_index].λ[whichev])  for x in resultsdg[indexes_to_plot]]
    y = [abs(x.λ[whichev] - resultsdgh[reference_indexdgh].λ[whichev])/(resultsdgh[reference_indexdgh].λ[whichev])  for x in resultsdgh[indexes_to_plotdgh]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdgh[indexes_to_plotdgh]]
    mycolors = [method_colors[f] for f in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdgh[indexes_to_plotdgh] ]
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdgh[indexes_to_plotdgh]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
      end
    else
      legendlabels = [contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements" for x in resultsdgh[indexes_to_plotdgh]]
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
    Plots.pdf("/tmp/DG25h.pdf")
    #Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25ev$(whichev)errsCG.pdf")
  else
    #Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25ev$(whichev)errsTO.pdf")
  end
  begin
    whichev=2
    x = [getH(x.ctx) for x in resultsdgh[indexes_to_plotdgh]]
    y = [max(1e-10,sqrt(abs(1 - norm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdgh[reference_indexdgh].statistics["B"]))[whichev,whichev]))))  for x in resultsdgh[indexes_to_plotdgh]]
    #y = [max(1e-10,sqrt(abs(1 - abs(x.statistics["E"][whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdgh[indexes_to_plotdgh]]
    mycolors = [method_colors[f] for f in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdgh[indexes_to_plotdgh] ]
    if j == 1
      slopes = evec_slopes
    else
      slopes = evec_slopes2
    end
    if j == 2
      legendlabels = [@sprintf("%s, %s ", contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdgh[indexes_to_plotdgh]]
      for i in 1:length(legendlabels)
        legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",slopes[gridtypes[i]] )
      end
    else
      legendlabels = [(contains(x.ctx.gridType,"P2") ? "P2 elements" : "P1 elements") for x in resultsdgh[indexes_to_plotdgh]]
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
  Plots.pdf("/tmp/DG25hevec.pdf")
end
