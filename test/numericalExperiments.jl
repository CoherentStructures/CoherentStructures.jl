#(c) 2018 Nathanael Schilling
#This file contains code for running numerical experiments with CoherentStructures.jl

using JLD2,StaticArrays,Tensors

#TODO: Replace Float64 with a more general type
#TODO: Generalize to dim != 2

#This struct contains information about the experiment run
struct testCase
    name::String #Something like "Ocean Flow" or other (human-readable) description
    LL::Vec{2}
    UR::Vec{2}
    bdata_predicate::Function
    dbc_facesets

    is_ode::Bool

    #We need the stuff below if is_ode == true
    t_initial::Float64
    t_final::Float64
    #Things that may be needed for this experiment
    ode_fun
    p #Parameters to pass to ode_fun,

    #If is_ode == false
    f
    Df

end

mutable struct experimentResult
    experiment::testCase
    ctx::CoherentStructures.gridContext
    bdata::CoherentStructures.boundaryData
    mode::Symbol #One of :naTO, :aTO, :CG, :L2GTO, etc..
    done::Bool
    runtime::Float64
    λ::Vector{Float64}
    V::Array{Float64,2}
    statistics::Dict{String, Any} #Things we can calculate about this solution

    #Like below, but make boundary data first
    function experimentResult(experiment::testCase,ctx::CoherentStructures.gridContext,mode)
        bdata = boundaryData(ctx,experiment.bdata_predicate, experiment.dbc_facesets)
        result = new(experiment,ctx,bdata,mode,false,-1.0,Vector{Float64}(0),Array{Float64}(0,2),Dict{String,Any}())
        return result
    end

    #Constructor from general CoherentStructures.gridContext object
    function experimentResult(experiment::testCase,ctx::CoherentStructures.gridContext,bdata::CoherentStructures.boundaryData,mode)
        result = new(experiment,ctx,bdata,mode,false,-1.0,Vector{Float64}(0),Array{Float64}(0,2),Dict{String,Any}())
        return result
    end
    #For regular Grids:
    function experimentResult(experiment::testCase, gridType::String, howmany , mode)
        ctx = regularGrid(gridType,howmany, experiment.LL, experiment.UR)
        bdata = boundaryData(ctx,experiment.bdata_predicate,experiment.dbc_facesets)
        return experimentResult(experiment,ctx,bdata,mode)
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
        if eR.experiment.is_ode
            ode_fun = eR.experiment.ode_fun
            #TODO: Think about varying the parameters below.
            cgfun = (x -> mean_diff_tensor(ode_fun,x,times, 1.e-8,tolerance=1.e-3,p=eR.experiment.p))
        else
            cgfun = x->0.5*(one(SymmetricTensor{2,2,Float64,4}) + dott(inv(eR.experiment.Df(x))))
        end
        eR.runtime += (@elapsed K = assembleStiffnessMatrix(eR.ctx,cgfun,bdata=eR.bdata))
        #TODO:Vary whether or not we lump the mass matrices or not
        eR.runtime += (@elapsed M = assembleMassMatrix(eR.ctx,bdata=eR.bdata,lumped=false))
        eR.runtime +=  (@elapsed λ, v = eigs(K,M,which=:SM,nev=nev))
        eR.λ = λ
        eR.V = v
    elseif eR.mode == :aTO
        if ! isempty(eR.bdata.periodic_dofs_from)
            error("Periodic boundary conditions not yet implemented for adaptive TO")
        end
        if !eR.experiment.is_ode
            error("TODO: Implement this!")
        end
        times = [eR.experiment.t_initial,eR.experiment.t_final]
        ode_fun = eR.experiment.ode_fun
        forwards_flow = u0->flow(ode_fun, u0,times,p=eR.experiment.p)[end]
        eR.runtime += (@elapsed S = assembleStiffnessMatrix(eR.ctx))
        eR.runtime += (@elapsed M = assembleMassMatrix(eR.ctx,bdata=eR.bdata))
        eR.runtime += (@elapsed S2= adaptiveTO(eR.ctx,forwards_flow))
        eR.runtime += (@elapsed λ, v = eigs(-1*(S + S2),M,which=:SM,nev=nev))
        meanS = CoherentStructures.applyBCS(eR.ctx, -0.5*(S+S2),eR.bdata)

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
    #TODO: Can we get rid of the error message from using :rainbow in any other way?
    Plots.clibrary(:misc)
    allplots = []
    for (i,lam) in enumerate(eR.λ)
        if nev != -1 && i > nev
            break
        end
        push!(allplots,plot_u(eR.ctx,real.(eR.V[:,i]),bdata=eR.bdata,title=(@sprintf("%.2f",lam)),plotit=false,color=:rainbow;kwargs...))
    end
    Plots.plot(allplots...)
end



#TODO: Think of moving helper functions like these to GridFunctions.jl
function sampleTo(u::Vector{Float64}, ctx_old::CoherentStructures.gridContext, ctx_new::CoherentStructures.gridContext)
    u_new::Vector{Float64} = zeros(ctx_new.n)*NaN
    for i in 1:ctx_new.n
        u_new[ctx_new.node_to_dof[i]] = evaluate_function_from_dofvals(ctx_old,u,ctx_new.grid.nodes[i].x,NaN,true)
    end
    return u_new
end

function getnorm(u::Vector{Float64},ctx::CoherentStructures.gridContext,which="L∞", M=nothing)
    if which == "L∞"
        return maximum(abs.(u))
    elseif which == "L2"
        return sqrt(getInnerProduct(ctx,u,u,M))
    else
        error("Not yet implemented")
    end
end

function getInnerProduct(ctx::CoherentStructures.gridContext, u1::Vector{Float64},u2::Vector{Float64},Min=nothing)
        if Min == nothing
            M = assembleMassMatrix(ctx)
        else
            M = Min
        end
        Mu1 = M*u1
        return  u2 ⋅ Mu1
end

function getDiscreteInnerProduct(ctx1::CoherentStructures.gridContext, u1::Vector{Float64}, ctx2::CoherentStructures.gridContext, u2::Vector{Float64},nx=400,ny=400)
    res = 0.0
    for x in linspace(ctx1.spatialBounds[1][1],ctx1.spatialBounds[2][1],nx)
        for y in linspace(ctx1.spatialBounds[1][2],ctx1.spatialBounds[2][2],ny)
            res += evaluate_function_from_dofvals(ctx1,u1,[x,y],NaN) * evaluate_function_from_dofvals(ctx2,u2,[x,y],NaN)
        end
    end
    return  res/(nx*ny)
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
    bdata_predicate = (x,y) -> false
    result = testCase("Ocean Flow", LL,UR,bdata_predicate,"all",true,t_initial,t_final, interp_rhs,p,nothing,nothing)
    return result
end

function makeDoubleGyreTestCase(tf=1.0)
    LL=Vec{2}([0.0,0.0])
    UR=Vec{2}([1.0,1.0])
    bdata_predicate = (x,y) -> false
    result = testCase("Rotating Double Gyre",LL,UR,bdata_predicate,[], true,0.0,tf, rot_double_gyre,nothing,nothing,nothing)
    return result
end

function makeStandardMapTestCase()
    LL = Vec{2}([0.0,0.0])
    UR=Vec{2}([2π,2π])
    bdata_predicate = (x,y) -> (mod(x[1] - y[1],2π) < 1e-9 && mod(x[2] - y[2],2π)<1e-9)
    result = testCase("Standard Map",LL,UR,bdata_predicate,[], false,NaN,NaN, nothing,nothing,CoherentStructures.standardMap,CoherentStructures.DstandardMap)
    return result
end

function zeroRHS(u,p,t)
    return @SVector [0.0, 0.0]
end

function makeStaticLaplaceTestCase()
    LL=Vec{2}([0.0,0.0])
    UR=Vec{2}([1.5,1.0])
    bdata_predicate = (x,y) -> false
    result = testCase("Static Laplace",LL,UR,bdata_predicate,[], true,0.0,0.01, zeroRHS,nothing,nothing,nothing)
    return result
end


function accuracyTest(tC::testCase,reference::experimentResult,quadrature_order=CoherentStructures.default_quadrature_order)
    print("Running reference experiment")
    experimentResults = Vector{experimentResult}(0)
    runExperiment!(reference)
    push!(experimentResults,reference)
    print("Finished running reference experiment")
    #gridConstructors = [regularTriangularGrid, regularDelaunayGrid, regularP2TriangularGrid, regularP2DelaunayGrid , regularQuadrilateralGrid,regularP2QuadrilateralGrid]
    #gridConstructorNames = ["regular triangular grid", "regular Delaunay grid","regular P2 triangular grid", "regular P2 Delaunay grid", "regular quadrilateral grid", "regular P2 quadrilateral grid"]

    gridConstructors = [regularTriangularGrid, regularP2TriangularGrid ]
    gridConstructorNames = ["regular triangular grid", "regular P2 triangular grid" ]
    for (gCindex,gC) in enumerate(gridConstructors)
        #TODO: replace this with something more sensible...
        for width in collect(20:20:200)
            ctx = gC((width,width),tC.LL,tC.UR,quadrature_order=quadrature_order)
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
    M_ref = assembleMassMatrix(reference.ctx)
    n_experiments = size(experimentResults)[1]
    for (eRindex, eR) in enumerate(experimentResults)
        print("Building Statistics for run $eRindex/$n_experiments \n")
        if eRindex == referenceIndex
            continue
        end
        linftyerrors = Vector{Float64}(0)
        l2errors = Vector{Float64}(0)
        λerrors = Vector{Float64}(0)
        errors = Array{Array{Float64}}(0)

        E = zeros(6,6)
        E_discrete = zeros(6,6)
        for i in 1:6
            index = sortperm(real.(eR.λ))[end- (i - 1)]
            upsampled = sampleTo(undoBCS(eR.ctx,eR.V[:,index],eR.bdata),eR.ctx,reference.ctx)
            upsampled /= getnorm(upsampled,reference.ctx,"L2",M_ref)
            for j in 1:6
                index_ref = sortperm(real.(reference.λ))[end - (j - 1)]
                ref_ev = undoBCS(reference.ctx, reference.V[:,j],reference.bdata)
                ref_ev /= getnorm(ref_ev,reference.ctx,"L2",M_ref)
                E[i,j] = getInnerProduct(reference.ctx, reference.V[:,index_ref], upsampled,M_ref)
            end

            index_ref = sortperm(real.(reference.λ))[end - (i - 1)]
            push!(λerrors, abs(eR.λ[index] - reference.λ[index_ref]))
        end
        #for j in 1:6
        #    index_ref = sortperm(real.(reference.λ))[end - (j - 1)]
        #    normIndex_ref = sqrt(getDiscreteInnerProduct(reference.ctx, reference.V[:,index_ref],
        #                                                 reference.ctx, reference.V[:,index_ref],800,800))
        #    for i in 1:6
        #        index = sortperm(real.(eR.λ))[end- (i - 1)]
        #        normIndex = sqrt(getDiscreteInnerProduct(eR.ctx, eR.V[:,index], eR.ctx, eR.V[:,index],800,800))
        #        E_discrete[i,j] = getDiscreteInnerProduct(reference.ctx, reference.V[:,index_ref],eR.ctx,eR.V[:,index])/(normIndex_ref*normIndex)
        #    end
        #end
        experimentResults[eRindex].statistics["λ-errors"] = λerrors
        experimentResults[eRindex].statistics["E"]  = E
        #experimentResults[eRindex].statistics["E_discrete"]  = E_discrete
    end
end

function testDoubleGyre(tf=1.0)
    doubleGyreTestCase = makeDoubleGyreTestCase(tf)
    referenceCtx = regularP2TriangularGrid( (300,300), doubleGyreTestCase.LL,doubleGyreTestCase.UR)
    reference = experimentResult(doubleGyreTestCase,referenceCtx,:CG)
    result =  accuracyTest(doubleGyreTestCase, reference)
    return result
end


function testStaticLaplace()
    staticLaplaceTestCase = makeStaticLaplaceTestCase()
    referenceCtx = regularP2QuadrilateralGrid( (200,200), staticLaplaceTestCase.LL,staticLaplaceTestCase.UR)
    reference = experimentResult(staticLaplaceTestCase,referenceCtx,:CG)
    result =  accuracyTest(staticLaplaceTestCase, reference)
    return result
end

function testStandardMap()
    tC = makeStandardMapTestCase()
    referenceCtx = regularP2TriangularGrid( (300,300), tC.LL,tC.UR)
    reference = experimentResult(tC,referenceCtx,:CG)
    result =  accuracyTest(tC, reference)
    return result
end
