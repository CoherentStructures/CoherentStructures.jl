#(c) 2018 Nathanael Schilling
#Plots for (more general) L^2-Galerkin approximations of the transfer-operator

using CoherentStructures
include("numericalExperiments.jl")

mutable struct FEMTransferExperimentResult
    ctx::CoherentStructures.gridContext
    bdata::CoherentStructures.boundaryData
    ϵ::Float64
    n_stencil_points::Int
    λ::Vector{Complex128}
    V::Array{Complex128,2}
end

function sm(u)
    ϵ = 1
    ρ = 0.3
    u2 = u[2]+0.5
    return StaticArrays.SVector{2,Float64}((
        mod(u[1] + ϵ*u2 + ϵ^2*ρ*sin(2π*u[1]), 1),
        mod(u[2] + ϵ*ρ*sin(2π*u[1]) , 1)
        ))
end

2π/500

function sminv(x::Vec{2,Float64})::Vec{2,Float64}
    ϵ::Float64 = 1.0
    ρ::Float64 = 0.3
    return Vec{2,Float64}((
        mod(x[1] - x[2] - 0.5, 1.0),
        mod(x[2] - ρ*sin(2π*(x[1] - x[2] - 0.5)) , 1.0)
        ))
end

Plots.clibrary(:misc)

FEMTransferResults = FEMTransferExperimentResult[]
gridResolutions = [(2^i,2^i) for i in 3:1:8]
gridResolutions
ϵ=0.02
n_stencil_points = 10
for resolution in gridResolutions
    for j in (1, 2)
        print("P$j elements on $resolution\n")
        if j == 1
            ctx = regularTriangularGrid(resolution, [0.0,0.0],[1.,1.],quadrature_order=5)
        else
            ctx = regularP2TriangularGrid(resolution, [0.0,0.0],[1.,1.],quadrature_order=5)
        end
        pred  = (x,y) -> peuclidean(x, y, [1, 1]) < 1e-9
        bdata = boundaryData(ctx,pred) #Periodic boundary
        ALPHApreBC = L2GalerkinTOFromInverse(ctx,sminv,ϵ,periodic_directions=(true,true) ,n_stencil_points=n_stencil_points )
        gc()
        ALPHA = applyBCS(ctx,ALPHApreBC,bdata)
        M = assembleMassMatrix(ctx,bdata=bdata)
        print("Constructed Matrices")
        λ, V = eigs(ALPHA,M, which=:LR, nev=20)
        print("Solved eigenproblem")

        push!(FEMTransferResults, FEMTransferExperimentResult(ctx,bdata,ϵ,n_stencil_points,λ,V))
        gc()
    end
end
#fd = open("FMSM","w")
#serialize(fd,FEMTransferResults)
#close(fd)



hs = []
ev_errs = []
evec_errs = []
gridTypes = []
begin
    whichev = 2
    reference_index = length(FEMTransferResults)
    referenceEv = FEMTransferResults[reference_index].λ[whichev]
    referenceCtx = FEMTransferResults[reference_index].ctx
    referenceEvec = undoBCS(referenceCtx,FEMTransferResults[reference_index].V[:,whichev],FEMTransferResults[reference_index].bdata)
    M = assembleMassMatrix(referenceCtx)
    referenceEvec /= getnorm(referenceEvec,referenceCtx,"L2",M)
    for (index,res) in enumerate(FEMTransferResults)
        if index == reference_index
            continue
        end
        push!(hs, getH(res.ctx))
        thisevec = sampleTo(
                undoBCS(res.ctx,res.V[:,whichev],res.bdata),res.ctx,
                    FEMTransferResults[end].ctx)
        thisev = res.λ[whichev]
        thisevec /= getnorm(thisevec,referenceCtx,"L2",M)
        err = √(1 - abs(getInnerProduct(referenceCtx,thisevec,referenceEvec,M)))
        push!(evec_errs,err)
        push!(ev_errs, abs((referenceEv - thisev)/referenceEv))
        push!(gridTypes, res.ctx.gridType)
    end
end

legendlabels = [contains(x,"P2")? "P2 Elements" : "P1 Elements" for x in gridTypes]
ms = [contains(x,"P2")? :s : :utri for x in gridTypes]
Plots.scatter(hs, ev_errs,xscale=:log10,yscale=:log10,group=gridTypes,label=legendlabels,legend=(0.8,0.2),m=ms,xlabel="Mesh width",ylabel="Relative error")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/FEMTOL2SM$(whichev)everrs.pdf")
Plots.scatter(hs,evec_errs,xscale=:log10,yscale=:log10,group=gridTypes,label=legendlabels,legend=(0.8,0.2),m=ms,xlabel="Mesh width", ylabel="Error")
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/FEMTOL2SM$(whichev)evecerrs.pdf")

using LaTeXStrings
plotsres = []
for i in 2:7
    push!(plotsres,
        plot_u(FEMTransferResults[end].ctx,real.(FEMTransferResults[end].V[:,i])/maximum(abs.(real.(FEMTransferResults[end].V[:,i]))),200,200,bdata=FEMTransferResults[end].bdata,color=:rainbow,colorbar=:none,
        #title=L"Re(\lambda) = " * @sprintf("%.2f", 1.2))
        title=latexstring("Re(\\lambda) = "*@sprintf("%.2f", real(FEMTransferResults[end].λ[i]))))
        )
end
Plots.plot(plotsres...)
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/FEMTOL2SMevecs.pdf")
Plots.pdf("/tmp/output.pdf")
plot_spectrum(FEMTransferResults[3].λ)
length(FEMTransferResults[end].V[:,1])
FEMTransferResults[end].bdata

for i in 1:length(FEMTransferResults)
    Plots.display(
        plot_u(FEMTransferResults[i].ctx, real.(FEMTransferResults[i].V[:,2]),200,200,bdata=FEMTransferResults[i].bdata,color=:rainbow))
        sleep(2)
    end
end
