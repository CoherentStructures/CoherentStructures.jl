#(c) 2018 Nathanael Schilling
#Plots for paper

using CoherentStructures
using Plots, Serialization, LinearAlgebra, Printf

include("numericalExperiments.jl")
include("define_bickley_jet.jl")

experimentRange= 2 .^ (4:8) .+ 1
experimentRangeSmall= 2 .^ (4:6) .+ 1

function runGenericTest(tC, name; nol2g=false)
  results0 = testGeneric(tC,[],quadrature_order=5,run_reference=true)
  results1 = testGeneric(tC,experimentRange,quadrature_order=2,run_reference=false)
  results3 = testGeneric(tC,experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
  if ! nol2g
    results4 = testGeneric(tC,experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)
  end

  results = copy(results0)
  append!(results,results1)
  append!(results,results3)
  if ! nol2g
    append!(results,results4)
  end
  buildStatistics!(results,1)


  myfd = open(name,"w")
  serialize(myfd,results)
  close(myfd)
end

rtC= makeRotationTestCase()
runGenericTest(rtC,"SM";nol2g=true)
res = testGeneric(rtC,[50], quadrature_order=2,run_reference=false,mode=:CG)
#plot_u(res[1].ctx,res[1].V[:,2],bdata=res[1].bdata)

K = assembleStiffnessMatrix(res[1].ctx,bdata=res[1].bdata)
M = assembleMassMatrix(res[1].ctx,bdata=res[1].bdata)

pdf = res[1].bdata.periodic_dofs_from
plot_real_spectrum(λ)
λ, V = eigs(K,M, which=:SM)
myplot = plot_u(res[1].ctx,V[:,6],bdata=res[1].bdata)
for i in pdf
  global res
  ctx = res[1].ctx
  from_node = ctx.grid.nodes[ctx.dof_to_node[i]]
  myplot = Plots.scatter!([from_node.x[1]],[from_node.x[2]],legend=:none)
end
myplot

function readGenericTestCase(name)
  results=open(Serialization.deserialize,name)
  GC.gc()
  return results
end

readGenericTestCase("SM")[1]

function genericTestCasePlots(name;eigenspace=1:1)
  reference_index = 1
  results = readGenericTestCase(name)

  for j in (1, 2)
    if j == 1
      indexes_to_plot = [i for i in 2:length(results) if results[i].mode === :CG]
    else
      indexes_to_plot = [i for i in 2:length(results) if results[i].mode !== :CG]
    end
    begin
      whichev = 2
      x = [getH(x.ctx) for x in results[indexes_to_plot]]
      y = [abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      colors = [x.mode === :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in results[indexes_to_plot]]
      Plots.scatter(x,y,group=gridtypes,label=legendlabels,color=colors,xlabel="Mesh width",ylabel="Relative error",m=ms,
        xscale=:log10,yscale=:log10,  legend=(0.40,0.20),
        )
        #ylim=(1e-10,1),xlim=(10^-1.8,1.03));
      Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes))
      sleep(2)
      #loglogleastsquareslines(x,y,gridtypes)
    end
    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/$(name)$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/$(name)$(whichev)errsTO.pdf")
    end

    begin
      x = [getH(x.ctx) for x in results[indexes_to_plot]]
      matrices = [(inv(x.statistics["B"])*x.statistics["E"]*inv(results[reference_index].statistics["B"]))[eigenspace,eigenspace] for x in results[indexes_to_plot]]
      #y = [sqrt(abs(1 - opnorm(x))) for x in matrices]
      y = [sqrt(abs(1 - sqrt(abs(maximum(eigvals(x'*x)))))) for x in matrices]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      mycolors = [method_colors[f] for f in gridtypes]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenspace error",
        xscale=:log10, yscale=:log10, legend=(0.55,0.25),color=mycolors,lab=legendlabels,m=ms,
      #  ylim=(1.e-7,1e-1),xlim=(10^(-1.7),1.03))
      )
      Plots.display(loglogslopeline(x,y,gridtypes,evec_slopes))
      sleep(2)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/$(name)evec23errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/$(name)evec23errsTO.pdf")
    end
  end
end

genericTestCasePlots("ROT3",eigenspace=2:5)
results = readGenericTestCase("ROT3")
plot_u(results[1].ctx, results[1].V[:,2], bdata=results[1].bdata)
plot_u(results[2].ctx, results[2].V[:,2], bdata=results[1].bdata)


function runStandardMapTests()
  results0 = testStandardMap([],quadrature_order=5,run_reference=true)
  results1 = testStandardMap(experimentRange,quadrature_order=2,run_reference=false)
  results3 = testStandardMap(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
  results4 = testStandardMap(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)

  results = copy(results0)
  append!(results,results1)
  append!(results,results3)
  append!(results,results4)
  buildStatistics!(results,1)


  myfd = open("SM","w")
  serialize(myfd,results)
  close(myfd)
end

runStandardMapTests()


function runStandardMap8Tests()
  results0 = testStandardMap8([],quadrature_order=5,run_reference=true)
  results1 = testStandardMap8(experimentRange,quadrature_order=2,run_reference=false)
  results3 = testStandardMap8(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
  results4 = testStandardMap8(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)

  results = copy(results0)
  append!(results,results1)
  append!(results,results3)
  append!(results,results4)
  buildStatistics!(results,1)


  myfd = open("SM8","w")
  serialize(myfd,results)
  close(myfd)
end

runStandardMap8Tests()



function runStandardMapQuadratureTests()
  results0 = testStandardMap([63],quadrature_order=8,run_reference=false)
  resultsP1 = [results0[1]]
  resultsP2 = [results0[2]]
  for i in 1:7
    results1 = testStandardMap([63],quadrature_order=i,run_reference=false)
    if i != 1
      push!(resultsP2, results1[2])
    end
      push!(resultsP1, results1[1])
  end

  buildStatistics!(resultsP1,1)
  buildStatistics!(resultsP2,1)

  myfd = open("SMQ","w")
  serialize(myfd,(resultsP1,resultsP2))
  close(myfd)
end

runStandardMapQuadratureTests()

function standardMapPlots()
  reference_index = 1
  results=open(Serialization.deserialize,"SM")
  GC.gc()

  for j in (1, 2)
    if j == 1
      indexes_to_plot = [i for i in 2:length(results) if results[i].mode === :CG]
    else
      indexes_to_plot = [i for i in 2:length(results) if results[i].mode !== :CG]
    end
    begin
      whichev = 2
      x = [getH(x.ctx) for x in results[indexes_to_plot]]
      y = [abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      colors = [x.mode === :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in results[indexes_to_plot]]
      Plots.scatter(x,y,group=gridtypes,label=legendlabels,color=colors,xlabel="Mesh width",ylabel="Relative error",m=ms,
        xscale=:log10,yscale=:log10,  legend=(0.40,0.20),
        ylim=(1e-10,1),xlim=(10^-1.8,1.03));
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
      matrices = [(inv(x.statistics["B"])*x.statistics["E"]*inv(results[reference_index].statistics["B"]))[2:3,2:3] for x in results[indexes_to_plot]]
      y = [sqrt(abs(1 - maximum(eigvals(x'*x)))) for x in matrices]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      mycolors = [method_colors[f] for f in gridtypes]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenspace error",
        xscale=:log10, yscale=:log10, legend=(0.55,0.25),color=mycolors,lab=legendlabels,m=ms,
        ylim=(1.e-7,1e-1),xlim=(10^(-1.7),1.03))
      Plots.display(loglogslopeline(x,y,gridtypes,evec_slopes))
      sleep(2)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMevec23errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMevec23errsTO.pdf")
    end
  end
end

standardMapPlots()


function standardMap8Plots()
  reference_index = 1
  results=open(Serialization.deserialize,"SM8")
  GC.gc()

  for j in (1, 2)
    if j == 1
      indexes_to_plot = [i for i in 2:length(results) if results[i].mode === :CG]
    else
      indexes_to_plot = [i for i in 2:length(results) if results[i].mode !== :CG]
    end
    begin
      whichev = 2
      x = [getH(x.ctx) for x in results[indexes_to_plot]]
      y = [abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      colors = [x.mode === :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in results[indexes_to_plot]]
      Plots.scatter(x,y,group=gridtypes,label=legendlabels,color=colors,xlabel="Mesh width",ylabel="Relative error",m=ms,
        xscale=:log10,yscale=:log10,  legend=(0.40,0.20))
        #ylim=(1e-10,1),xlim=(10^-1.8,1.03));
      Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes))
      sleep(2)
      #loglogleastsquareslines(x,y,gridtypes)
    end
    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SM8ev$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SM8ev$(whichev)errsTO.pdf")
    end

    begin
      x = [getH(x.ctx) for x in results[indexes_to_plot]]
      matrices = [(inv(x.statistics["B"])*x.statistics["E"]*inv(results[reference_index].statistics["B"]))[2:3,2:3] for x in results[indexes_to_plot]]
      y = [sqrt(abs(1 - maximum(eigvals(x'*x)))) for x in matrices]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      mycolors = [method_colors[f] for f in gridtypes]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenspace error",
        xscale=:log10, yscale=:log10, legend=(0.55,0.25),color=mycolors,lab=legendlabels,m=ms)
        ##ylim=(1.e-7,1e-1),xlim=(10^(-1.7),1.03))
      Plots.display(loglogslopeline(x,y,gridtypes,evec_slopes))
      sleep(2)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SM8evec23errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SM8evec23errsTO.pdf")
    end
  end
end

standardMap8Plots()


function standardMapQuadraturePlots()
  reference_index = 1
  resultsboth =open(Serialization.deserialize,"SMQ")
  GC.gc()

  for j in 1:2
    results = resultsboth[j]
    indexes_to_plot = 2:length(results)

    begin
      whichev = 2
      x = indexes_to_plot .- 1
      y = [abs(x.λ[whichev] - results[reference_index].λ[whichev])/(results[reference_index].λ[whichev])  for x in results[indexes_to_plot]]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",ev_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      colors = [x.mode === :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in results[indexes_to_plot]]
      res = Plots.scatter(x,y,group=gridtypes,label=legendlabels,color=colors,xlabel="Quadrature Order",ylabel="Relative error",m=ms,
        yscale=:log10,  legend=(0.40,0.20))
        #ylim=(1e-10,1),xlim=(10^-1.8,1.03));
      #Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes))
      Plots.display(res)
      #loglogleastsquareslines(x,y,gridtypes)
    end
    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMQev$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMQev$(whichev)errsTO.pdf")
    end

    begin
      x = indexes_to_plot .- 2 .+ j
      matrices = [(inv(x.statistics["B"])*x.statistics["E"]*inv(results[reference_index].statistics["B"]))[2:3,2:3] for x in results[indexes_to_plot]]
      y = [sqrt(abs(1 - maximum(eigvals(x'*x)))) for x in matrices]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in results[indexes_to_plot]]
      mycolors = [method_colors[f] for f in gridtypes]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in results[indexes_to_plot]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf(" (%.1f)",evec_slopes[gridtypes[i]] )
        end
      end
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in results[indexes_to_plot] ]
      res = Plots.scatter(x,y,group=gridtypes,xlabel="Quadrature Order",ylabel="Eigenspace error",
        yscale=:log10, legend=(0.55,0.25),color=mycolors,lab=legendlabels,m=ms)
        #ylim=(1.e-7,1e-1),xlim=(10^(-1.7),1.03))
      #Plots.display(loglogslopeline(x,y,gridtypes,evec_slopes))
      Plots.display(res)
      sleep(10)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMQevec23errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMQevec23errsTO.pdf")
    end
  end
end

standardMapQuadraturePlots()



function runDoubleGyreTests()
  results0dg = testDoubleGyre([],quadrature_order=5,tf=0.25,run_reference=true)
  results1dg = testDoubleGyre(experimentRange,run_reference=false,quadrature_order=5,tf=0.25)
  results3dg = testDoubleGyre(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2,tf=0.25)
  results4dg = testDoubleGyre(experimentRange[1:end],mode=:aTO,run_reference=false,quadrature_order=2,tf=0.25)
  #results4dg = testDoubleGyre(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4,tf=0.25)

  resultsdg = copy(results0dg)
  append!(resultsdg,results1dg)
  append!(resultsdg,results3dg)
  append!(resultsdg,results4dg)
  buildStatistics!(resultsdg,1)


  myfd = open("DG25","w")
  serialize(myfd,resultsdg)
  close(myfd)
  GC.gc()
end

runDoubleGyreTests()

function runDoubleGyreEqVariTests()
  results0dg = testDoubleGyreEqVari([],quadrature_order=5,tf=0.25,run_reference=true)
  results1dg = testDoubleGyreEqVari(experimentRange,run_reference=false,quadrature_order=5,tf=0.25)
  #results3dg = testDoubleGyreEqVari(experimentRange[2:end],mode=:naTO,run_reference=false,quadrature_order=2,tf=0.25)
  #results4dg = testDoubleGyreEqVari(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4,tf=0.25)

  resultsdg = copy(results0dg)
  append!(resultsdg,results1dg)
  buildStatistics!(resultsdg,1)


  myfd = open("DGEqVari25","w")
  serialize(myfd,resultsdg)
  close(myfd)
  GC.gc()
end

runDoubleGyreEqVariTests()


function runDoubleGyreEqVari1Tests()
  results0dg = testDoubleGyreEqVari([],quadrature_order=5,tf=1.0,run_reference=true)
  results1dg = testDoubleGyreEqVari(experimentRange,run_reference=false,quadrature_order=5,tf=1.0)
  #results3dg = testDoubleGyreEqVari(experimentRange[2:end],mode=:naTO,run_reference=false,quadrature_order=2,tf=0.25)
  #results4dg = testDoubleGyreEqVari(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4,tf=0.25)

  resultsdg = copy(results0dg)
  append!(resultsdg,results1dg)
  buildStatistics!(resultsdg,1)


  myfd = open("DG1EqVari25","w")
  serialize(myfd,resultsdg)
  close(myfd)
  GC.gc()
end


function doubleGyrePlots()
  resultsdg=open(deserialize,"DG25")
  reference_indexdg = 1

  for j in (1, 2)
    if j == 1
      indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode === :CG]
    else
      indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode !== :CG]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in resultsdg[indexes_to_plotdg]]
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
      Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes,lsq_c=false))
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
      y = [max(1e-10,sqrt(abs(1 - opnorm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg[reference_indexdg].statistics["B"]))[whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",slopes[gridtypes[i]] )
        end
      else
        legendlabels = [(occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements") for x in resultsdg[indexes_to_plotdg]]
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
        ylim=ylim,xlim=(10^-2.5,10^-0.5))
      Plots.display(loglogslopeline(x,y,gridtypes,slopes,lsq_c=false))
      sleep(2)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25evec$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25evec$(whichev)errsTO.pdf")
    end
  end
end

doubleGyrePlots()


function doubleGyreEqVariPlots()
  resultsdg=open(deserialize,"DGEqVari25")
  reference_indexdg = 1

  for j in (1,)
    if j == 1
      indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode === :CG]
    else
      indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode !== :CG]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in resultsdg[indexes_to_plotdg]]
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
      Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes,lsq_c=false))
      sleep(2)
    end
    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25EqVariev$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25EqVariev$(whichev)errsTO.pdf")
    end
    begin
      whichev=2
      x = [getH(x.ctx) for x in resultsdg[indexes_to_plotdg]]
      y = [max(1e-10,sqrt(abs(1 - opnorm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg[reference_indexdg].statistics["B"]))[whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",slopes[gridtypes[i]] )
        end
      else
        legendlabels = [(occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements") for x in resultsdg[indexes_to_plotdg]]
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
        ylim=ylim,xlim=(10^-2.5,10^-0.5))
      Plots.display(loglogslopeline(x,y,gridtypes,slopes,lsq_c=false))
      sleep(2)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25EqVarievec$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG25EqVarievec$(whichev)errsTO.pdf")
    end
  end
end

function doubleGyre1EqVariPlots()
  resultsdg=open(deserialize,"DG1EqVari25")
  reference_indexdg = 1

  for j in (1,)
    if j == 1
      indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode === :CG]
    else
      indexes_to_plotdg = [i for i in 2:length(resultsdg) if resultsdg[i].mode !== :CG]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements" for x in resultsdg[indexes_to_plotdg]]
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
      Plots.display(loglogslopeline(x,y,gridtypes,ev_slopes,lsq_c=false))
      sleep(2)
    end
    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG1EqVariev$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG1EqVariev$(whichev)errsTO.pdf")
    end
    begin
      whichev=2
      x = [getH(x.ctx) for x in resultsdg[indexes_to_plotdg]]
      y = [max(1e-10,sqrt(abs(1 - opnorm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg[reference_indexdg].statistics["B"]))[whichev,whichev]))))  for x in resultsdg[indexes_to_plotdg]]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg[indexes_to_plotdg]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",slopes[gridtypes[i]] )
        end
      else
        legendlabels = [(occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements") for x in resultsdg[indexes_to_plotdg]]
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
        ylim=ylim,xlim=(10^-2.5,10^-0.5))
      Plots.display(loglogslopeline(x,y,gridtypes,slopes,lsq_c=false))
      sleep(2)
    end

    if j == 1
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG1EqVarievec$(whichev)errsCG.pdf")
    else
      Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/DG1EqVarievec$(whichev)errsTO.pdf")
    end
  end
end



function makeStaticLaplaceTestCase()
  resultssl1 = testStaticLaplace(experimentRange,quadrature_order=2)
  resultssl3 = testStaticLaplace(experimentRange,mode=:naTO,run_reference=false,quadrature_order=2)
  resultssl4 = testStaticLaplace(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)


  resultssl = copy(resultssl1)
  append!(resultssl,resultssl3)
  append!(resultssl,resultssl4)
  buildStatistics!(resultssl,1)


  myfd = open("SL","w")
  serialize(myfd,resultssl)
  close(myfd)
  GC.gc()
end

function plotStaticLaplaceResults()

  reference_indexsl = 1
  resultssl=open(deserialize,"SL")
  indexes_to_plotsl = 2:length(resultssl)
  indexes_to_plotsl = [i for i in 2:length(resultssl) if resultssl[i].mode === :CG]


  begin
    whichev = 2
    x = [getH(x.ctx) for x in resultssl[indexes_to_plotsl]]
    #y = [abs(x.λ[whichev] - resultssl[reference_indexsl].λ[whichev]/(resultssl[reference_indexsl].λ[whichev]))  for x in resultssl[indexes_to_plotsl]]
    y = [abs((x.λ[whichev] - resultssl[reference_indexsl].λ[whichev])/resultssl[reference_indexsl].λ[whichev])  for x in resultssl[indexes_to_plotsl]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultssl[indexes_to_plotsl]]
    legendlabels = [x * @sprintf("(%.1f)",ev_slopes[x] ) for x in gridtypes]
    ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultssl[indexes_to_plotsl] ]
    colors = [x.mode === :CG ? :green : ((x.mode==:naTO) ?  :blue : :orange ) for x in resultssl[indexes_to_plotsl]]
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Relative error",
      xscale=:log10,yscale=:log10,legend=:best,m=ms, color=colors,xlim=(1e-3,10),ylim=(1e-10,1),
      title="Errors in eigenvalue $whichev")
    #loglogleastsquareslines(x,y,gridtypes)
    loglogslopeline(x,y,gridtypes,ev_slopes)
  end

  Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SLev$(whichev)errs.pdf")

  begin
    x = [getH(x.ctx) for x in resultssl[indexes_to_plotsl]]
    #y = [max(1e-10,sqrt(abs(1 - opnorm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultssl[reference_index].statistics["B"]))[2:3,2:3]))))  for x in resultssl[indexes_to_plotsl]]
    y = [max(1e-10,sqrt(abs(1 - opnorm((x.statistics["E"])[2:3,2:3]))))  for x in resultssl[indexes_to_plotsl]]
    gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultssl[indexes_to_plotsl]]
    Plots.scatter(x,y,group=gridtypes,xlabel="Mesh width",ylabel="Eigenvector error",
      xscale=:log10, yscale=:log10,legend=:none,# legend=(0.6,0.6),
      title="Error in Eigenspace {2,3}",ylim=(1.e-10,1e-1),xlim=(1e-3,10.))
      #loglogleastsquareslines(x,y,gridtypes)
  end


  Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SLevec$(whichev)errs.pdf")
end



function makeDoubleGyre1Results()
  results0dg1 = testDoubleGyre([],quadrature_order=5)
  results1dg1 = testDoubleGyre(experimentRange[2:end],quadrature_order=5,run_reference=false)
  results3dg1 = testDoubleGyre(experimentRange[2:end],mode=:naTO,run_reference=false,quadrature_order=2)
  results4dg1 = testDoubleGyre(experimentRangeSmall,mode=:L2GTOb,run_reference=false,quadrature_order=4)

  resultsdg1 = copy(results0dg1)
  append!(resultsdg1,results1dg1)
  append!(resultsdg1,results3dg1)
  append!(resultsdg1,results4dg1)
  buildStatistics!(resultsdg1,1)


  myfd = open("DG10","w")
  serialize(myfd,resultsdg1)
  close(myfd)
  gc()
end

makeDoubleGyre1Results()


function plotDoubleGyre1Results()
  reference_indexdg1 = 1
  resultsdg1=open(deserialize,"DG10")
  for j in (1, 2)
    #indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && getH(resultsdg1[i].ctx) < 10^(-1.)]
    if j == 1
      indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && resultsdg1[i].mode === :CG]
    else
      indexes_to_plotdg1 = [i for i in 1:length(resultsdg1) if i != reference_indexdg1 && resultsdg1[i].mode !== :CG]
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
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg1[indexes_to_plotdg1]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",ev_slopes[gridtypes[i]] )
        end
      else
        legendlabels = [(occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements") for x in resultsdg1[indexes_to_plotdg1]]
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
      y = [max(1e-10,sqrt(abs(1 - opnorm((inv(x.statistics["B"])*x.statistics["E"]*inv(resultsdg1[reference_indexdg1].statistics["B"]))[whichev,whichev]))))  for x in resultsdg1[indexes_to_plotdg1]]
      #y = [max(1e-10,sqrt(abs(1 - abs(x.statistics["E"][whichev,whichev]))))  for x in resultsdg1[indexes_to_plotdg1]]
      gridtypes = [x.ctx.gridType * " $(x.mode)" for x in resultsdg1[indexes_to_plotdg1]]
      mycolors = [method_colors[f] for f in gridtypes]
      ms = [x.ctx.gridType == "regular triangular grid" ? :utri : :s for x in resultsdg1[indexes_to_plotdg1] ]
      if j == 2
        legendlabels = [@sprintf("%s, %s ", occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements","$(x.mode)") for x in resultsdg1[indexes_to_plotdg1]]
        for i in 1:length(legendlabels)
          legendlabels[i] = legendlabels[i] * @sprintf("(%.1f)",evec_slopes2[gridtypes[i]] )
        end
      else
        legendlabels = [(occursin("P2",x.ctx.gridType) ? "P2 elements" : "P1 elements") for x in resultsdg1[indexes_to_plotdg1]]
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
end

plotDoubleGyre1Results()



function makeOceanFlowResults()
  oceanTC = makeOceanFlowTestCase()

  oceanCtx = regularTriangularGrid((80,48),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  oceanEx1 = experimentResult(oceanTC,oceanCtx,:CG,tolerance=1e-5)
  runExperiment!(oceanEx1)


  ctx2 = regularTriangularGrid((200,120),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  Vp1 = zeros(ctx2.n,6)
  for i in 1:6
    Vp1[:,i] = sample_to(undoBCS(oceanCtx, oceanEx1.V[:,i],oceanEx1.bdata), oceanCtx,ctx2)
  end

  n_partition = 4
  res = iterated_kmeans(100,permutedims(Vp1[:,1:(n_partition-1)]),n_partition)
  up1 = kmeansresult2LCS(res)
  clusterplotp1 = plot_u(ctx2,sum([-i*up1[:,i] for i in 1:n_partition]),800,480,colorbar=:none)




  oceanCtxP2 = regularP2TriangularGrid((25,15),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  oceanEx1P2 = experimentResult(oceanTC,oceanCtxP2,:CG,tolerance=1e-5)
  runExperiment!(oceanEx1P2)

  ctx2 = regularTriangularGrid((200,120),oceanTC.LL,oceanTC.UR,quadrature_order=2)
  V = zeros(ctx2.n,6)
  for i in 1:6
    V[:,i] = sample_to(undoBCS(oceanCtxP2, oceanEx1P2.V[:,i],oceanEx1P2.bdata), oceanCtxP2,ctx2)
  end


  n_partition = 4
  resP2 = iterated_kmeans(100,permutedims(V[:,1:(n_partition-1)]),n_partition)
  uP2 = kmeansresult2LCS(resP2)

  clusterplotp2 = plot_u(ctx2,sum([-i*uP2[:,i] for i in 1:n_partition]),400,480,colorbar=:none)

  Plots.display(Plots.plot(clusterplotp1,clusterplotp2,layout=(1,2),margin=-5Plots.px))
  Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/oceansresultCG.pdf")

end

makeOceanFlowResults()



function bickleyInvFlow(u0)
  LL = [0.0,-3.0]; UR=[6.371π,3.0]
  tmp =Vec{2}(flow(bickleyJet,u0,[tf,t0]; tolerance=1e-8)[end])
  return Vec{2}((mod(tmp[1],UR[1]),tmp[2]))
end

ctx.m
ctx.n
length(ctx.quadrature_points)

begin
  bickleyto_results = []
  for j in  [1,2]
    LL = [0.0,-3.0]; UR=[6.371π,3.0]
    if j == 1
      ctx = regularTriangularGrid((60,20), LL, UR, quadrature_order=2)
    else
      ctx = regularP2TriangularGrid((30,10), LL, UR, quadrature_order=2)
    end

    predicate = (p1, p2) -> peuclidean(p1, p2, [6.371π, Inf]) < 1e-10
    bdata = BoundaryData(ctx, predicate, []);

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
      V[:,i] = sample_to(undoBCS(ctx, v[:,i],bdata), ctx,ctx2)
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
for j in (1, 2)
  LL = [0.0,-3.0]; UR=[6.371π,3.0]
  if j == 1
    ctx = regularTriangularGrid((50,40),LL,UR,quadrature_order=2)
  else
    ctx = regularP2TriangularGrid((10,8),LL,UR,quadrature_order=2)
  end
  predicate = (p1, p2) -> peuclidean(p1, p2, [6.371π, Inf]) < 1e-10
  bdata = BoundaryData(ctx, predicate, []);

  cgfun = (x -> mean(pullback_diffusion_tensor(bickleyJet, x,linspace(0.0,40*3600*24,81),
       1.e-8,tolerance=1.e-5)))

  K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
  M = assembleMassMatrix(ctx,bdata=bdata)
  λ, v = eigs(K,M,which=:SM, nev= 10)

  ctx2 = regularP2TriangularGrid((200,60),LL,UR,quadrature_order=1)
  V = zeros(ctx2.n,10)
  for i in 1:10
    V[:,i] = sample_to(undoBCS(ctx, v[:,i],bdata), ctx,ctx2)
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
