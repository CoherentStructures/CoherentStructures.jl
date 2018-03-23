#(c) 2018 Nathanael Schilling
#This file contains code for running numerical experiments with juFEMDL


#TODO: Replace Float64 with a more general type
#TODO: Generalize to dim != 2

#This struct contains information about the experiment run
struct testCase
    name::String #Something like "Ocean Flow" or other (human-readable) description
    LL::Vec{2}
    UR::Vec{2}
    t_initial::Float64
    t_final::Float64
    #Things that may be needed for this experiment
    ode_fun::Function
    p #Parameters to pass to ode_fun
end

mutable struct experimentResult
    experiment::testCase
    ctx::gridContext
    mode::Symbol #One of :naTO, :aTO, :CG, :L2GTO, etc..
    done::Bool
    runtime::Float64
    λ::Vector{Float64}
    V::Array{Float64,2}
    statistics::Dict{String, Any} #Things we can calculate about this solution

    #Constructor from general gridContext object
    function experimentResult(experiment,ctx,mode)
        result = new(experiment,ctx,mode,false,-1.0,Vector{Float64}(0),Array{Float64}(0,2),Dict{String,Any}())
        return result
    end
    #For regular Grids:
    function experimentResult(experiment, gridType::String, howmany , mode)
        ctx = regularGrid(gridType,howmany, experiment.LL, experiment.UR)
        return experimentResult(experiment,ctx,mode)
    end

end

function runExperiment!(eR::experimentResult,nev=6)
    if eR.done
        print("Experiment was already run, not running again...")
        return
    end
    eR.runtime = 0.0
    if eR.mode == :CG
        times = [eR.experiment.t_initial,eR.experiment.t_final]
        ode_fun = eR.experiment.ode_fun
        #TODO: Think about varying the parameters below.
        cgfun = (x -> invCGTensor(ode_fun,x,times, 1.e-8,tolerance=1.e-3,p=eR.experiment.p))
        assembleStiffnessMatrix(eR.ctx)
        eR.runtime += (@elapsed S = assembleStiffnessMatrix(eR.ctx))
        eR.runtime += (@elapsed K = assembleStiffnessMatrix(eR.ctx,cgfun))
        #TODO:Vary whether or not we lump the mass matrices or not
        eR.runtime += (@elapsed M = assembleMassMatrix(eR.ctx,lumped=false))
        eR.runtime +=  (@elapsed λ, v = eigs(K,M,which=:SM,nev=nev))
        eR.λ = λ
        eR.V = v
    elseif eR.mode == :aTO
        times = [eR.experiment.t_initial,eR.experiment.t_final]
        ode_fun = eR.experiment.ode_fun
        forwards_flow = u0->flow(ode_fun, u0,times,p=eR.experiment.p)[end]
        eR.runtime += (@elapsed S = assembleStiffnessMatrix(eR.ctx))
        eR.runtime += (@elapsed M = assembleMassMatrix(eR.ctx))
        eR.runtime += (@elapsed S2= adaptiveTO(eR.ctx,forwards_flow))
        eR.runtime += (@elapsed λ, v = eigs(-1*(S + S2),M,which=:SM,nev=nev))
        eR.λ = λ
        eR.V = v
    else
        error("Not yet implemented")
    end
    eR.done = true
    return eR
end

function plotExperiment(eR::experimentResult,nev=-1; kwargs...)
    if !eR.done
        print("Experiment not yet run")
        return
    end
    #TODO: Can we get rid of the error in any other way
    Plots.clibrary(:misc)
    allplots = []
    for (i,lam) in enumerate(eR.λ)
        if nev != -1 && i > nev
            break
        end
        push!(allplots,plot_u(eR.ctx,real.(eR.V[:,i]),title=(@sprintf("%.2f",lam)),plotit=false,color=:rainbow;kwargs...))
    end
    Plots.plot(allplots...)
end



#TODO: Think of moving helper functions like these to GridFunctions.jl
function sampleTo(u::Vector{Float64}, ctx_old::gridContext, ctx_new::gridContext)
    u_new::Vector{Float64} = zeros(ctx_new.n)
    for i in 1:ctx_new.n
        u_new[ctx_new.node_to_dof[i]] = evaluate_function(ctx_old,ctx_new.grid.nodes[i].x,u)
    end
    return u_new
end

function getnorm(u::Vector{Float64},ctx::gridContext,which="L∞")
    if which == "L∞"
        return maximum(abs.(u))
    elseif which == "L2"
        M = assembleMassMatrix(ctx)
        Mu = M*u
        return Mu ⋅ u
    else
        error("Not yet implemented")
    end
end

function makeOceanFlowTestCase(location::AbstractString="examples/Ocean_geostrophic_velocity.jld2")

    JLD2.@load location Lon Lat Time UT VT
    # JLD version, requires more dependencies
    # vars = JLD.@load(location)
    # Lat = vars["Lat"]
    # Lon = vars["Lon"]
    # Time = vars["Time"]
    # UT = vars["UT"]
    # VT = vars["VT"]

    UI, VI = interpolateVF(Lon,Lat,Time,UT,VT)
    p = (UI,VI)

    #The computational domain
    LL = Vec{2}([-4.0,-34.0])
    UR = Vec{2}([6.0,-28.0])

    t_initial = minimum(Time)
    t_final = t_initial + 90
    result = testCase("Ocean Flow", LL,UR,t_initial,t_final, interp_rhs,p)
    return result
end

function makeDoubleGyreTestCase()
    LL=Vec{2}([0.0,0.0])
    UR=Vec{2}([1.0,1.0])
    result = testCase("Rotating Double Gyre",LL,UR,0.0,1.0, rot_double_gyre2!,nothing)
    return result
end


function accuracyTest(tC::testCase,reference::experimentResult)
    print("Running reference experiment")
    experimentResults = Vector{experimentResult}(0)
    runExperiment!(reference)
    push!(experimentResults,reference)
    print("Finished running reference experiment")
    gridConstructors = [regularTriangularGrid, regularDelaunayGrid, regularP2TriangularGrid, regularP2DelaunayGrid , regularQuadrilateralGrid,regularP2QuadrilateralGrid]
    gridConstructorNames = ["regular triangular grid", "regular Delaunay grid","regular P2 triangular grid", "regular P2 Delaunay Grid", "regular quadrilateral grid", "regular P2 quadrilateral grid"]
    for (gCindex,gC) in enumerate(gridConstructors)
        #TODO: replace this with something more sensible...
        for width in collect(20:20:200)
            ctx = gC((width,width),tC.LL,tC.UR)
            testCaseName = tC.name
            gCName = gridConstructorNames[gCindex]
            print("Running $testCaseName test case on $width×$width $gCName")
            eR = experimentResult(tC, ctx,:CG)
            runExperiment!(eR)
            push!(experimentResults,eR)
        end
    end
    return experimentResults
end

function buildStatistics!(experimentResults::Vector{experimentResult}, referenceIndex::Int64)
    reference = experimentResults[referenceIndex]
    for (eRindex, eR) in enumerate(experimentResults)
        if eRindex == referenceIndex
            continue
        end
        linftyerrors = Vector{Float64}(0)
        l2errors = Vector{Float64}(0)
        λerrors = Vector{Float64}(0)
        errors = Array{Array{Float64}}(0)
        for i in 1:6
            index = sortperm(real.(eR.λ))[end- (i - 1)]
            error = sampleTo(eR.V[:,index],eR.ctx, reference.ctx ) - reference.V[:,index]
            push!(errors,error)
            push!(linftyerrors, getnorm(error, reference.ctx,"L∞"))
            push!(l2errors, getnorm(error, reference.ctx,"L2"))
            push!(λerrors, abs(eR.λ[index] - reference.λ[index]))
        end
        experimentResults[eRindex].statistics["λ-errors"] = λerrors
        experimentResults[eRindex].statistics["L∞-errors"]  = linftyerrors
        experimentResults[eRindex].statistics["L2-errors"]  = l2errors
        experimentResults[eRindex].statistics["errors"]  = errors
    end
end

function testDoubleGyre()
    doubleGyreTestCase = makeDoubleGyreTestCase()
    referenceCtx = regularP2QuadrilateralGrid( (200,200), doubleGyreTestCase.LL,doubleGyreTestCase.UR)
    reference = experimentResult(doubleGyreTestCase,referenceCtx,:CG)
    result =  accuracyTest(doubleGyreTestCase, reference)
    buildStatistics!(result,1)
    return result
end
