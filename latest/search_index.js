var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#CoherentStructures.jl-1",
    "page": "Home",
    "title": "CoherentStructures.jl",
    "category": "section",
    "text": "Tools for computing Lagrangian Coherent Structures in Julia"
},

{
    "location": "index.html#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "CoherentStructures.jl is a toolbox for computing Lagrangian Coherent Structures in aperiodic flows in Julia. It has been developed in Oliver Junge\'s research group at TUM, Germany, by (in alphabetical order)Alvaro de Diego (@adediego)\nDaniel Karrasch (@dkarrasch)\nNathanael Schilling (@natschil)Contributions from colleagues in the field are most welcome via raising issues or, even better, via pull requests."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "First install the JuAFEM.jl package.Then, run the following in the Julia REPL:]add https://github.com/CoherentStructures/CoherentStructures.jl.git"
},

{
    "location": "rot_double_gyre.html#",
    "page": "Rotating Double Gyre",
    "title": "Rotating Double Gyre",
    "category": "page",
    "text": ""
},

{
    "location": "rot_double_gyre.html#Rotating-Double-Gyre-1",
    "page": "Rotating Double Gyre",
    "title": "Rotating Double Gyre",
    "category": "section",
    "text": ""
},

{
    "location": "rot_double_gyre.html#Description-1",
    "page": "Rotating Double Gyre",
    "title": "Description",
    "category": "section",
    "text": "The rotating double gyre model was introduced by Mosovsky & Meiss. It can be derived from the stream functionpsi(xyt)=(1s(t))psi_P +s(t)psi_F  psi_P (x y) = sin(2pi x) sin(pi y)  psi_F (x y) = sin(pi x) sin(2pi y)where s is (usually taken to be) a cubic interpolating function satisfying s(0) = 0 and s(1) = 1. It therefore interpolates two double gyre flow fields, from horizontally to vertically arranged counter-rotating gyres. The corresponding velocity field is provided by the package and callable as rot_double_gyre.(Image: )"
},

{
    "location": "rot_double_gyre.html#FEM-Based-Methods-1",
    "page": "Rotating Double Gyre",
    "title": "FEM-Based Methods",
    "category": "section",
    "text": "The following code-snippet shows how these methods can be used.using CoherentStructures,Arpack\nLL = [0.0,0.0]; UR = [1.0,1.0];\nctx = regularTriangularGrid((50,50),LL,UR)\n\nA = x-> mean_diff_tensor(rot_double_gyre,x,[0.0,1.0], 1.e-10,tolerance= 1.e-4)\nK = assembleStiffnessMatrix(ctx,A)\nM = assembleMassMatrix(ctx)\nλ, v = eigs(-K,M,which=:SM);This velocity field is given by the rot_double_gyre function. The second argument to mean_diff_tensor are the times at which we average the pullback diffusion tensors. The third parameter is the step size δ used for the finite-difference scheme, tolerance is passed to the ODE solver from DifferentialEquations.jl. In the above, A(x) approximates the mean diffusion tensor given byA(x) = sum_t in mathcal T(DPhi^t(x))^-1 (DPhi^t x)^-TThe eigenfunctions saved in v approximate those of Delta^dynimport Plots\nres = [plot_u(ctx, v[:,i],100,100,colorbar=:none,clim=(-3,3)) for i in 1:6];\nPlots.plot(res...,margin=-10Plots.px)Looking at the spectrum, there appears a gap after the third eigenvalue:Plots.scatter(1:6, real.(λ))We can use the Clustering.jl package to compute coherent structures from the first two nontrivial eigenfunctions:using Clustering\n\nctx2 = regularTriangularGrid((200,200))\nv_upsampled = sample_to(v,ctx,ctx2)\n\n\nnumclusters=2\nres = kmeans(permutedims(v_upsampled[:,2:numclusters+1]),numclusters+1)\nu = kmeansresult2LCS(res)\nPlots.plot([plot_u(ctx2,u[:,i],200,200,color=:viridis,colorbar=:none) for i in [1,2,3]]...)\n"
},

{
    "location": "rot_double_gyre.html#Geodesic-vortices-1",
    "page": "Rotating Double Gyre",
    "title": "Geodesic vortices",
    "category": "section",
    "text": "Here, we demonstrate how to calculate black-hole vortices, see Geodesic elliptic material vortices for references and details.using CoherentStructures\nimport Tensors, OrdinaryDiffEq, Plots\n\nconst q = 51\nconst tspan = collect(range(0.,stop=1.,length=q))\nny = 101\nnx = 101\nN = nx*ny\nxmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0\nxspan, yspan = range(xmin,stop=xmax,length=nx), range(ymin,stop=ymax,length=ny)\nP = vcat.(xspan\',yspan)\nconst δ = 1.e-6\nmCG_tensor = u -> CG_tensor(rot_double_gyre,u,tspan,δ,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())\n\nC = map(mCG_tensor,P)\n\nLCSparams = (.1, 0.5, 0.01, 0.2, 0.3, 60)\nvals, signs, orbits = ellipticLCS(C,xspan,yspan,LCSparams);The results are then visualized as follows.using Statistics\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C)\n# damp \"outliers\"\nl₁ = min.(λ₁,quantile(λ₁[:],0.999))\nl₁ = max.(λ₁,1e-2)\nl₂ = min.(λ₂,quantile(λ₂[:],0.995))\n\nfig = Plots.heatmap(xspan,yspan,log10.(l₂),aspect_ratio=1,color=:viridis,\n            title=\"FTLE-field and transport barriers\",xlims=(xmin, xmax),ylims=(ymin, ymax),leg=true)\nfor i in eachindex(orbits)\n    Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label=\"T = $(round(vals[i],digits=2))\")\nend\nPlots.plot(fig)"
},

{
    "location": "ocean_flow.html#",
    "page": "Geostrophic Ocean Flow",
    "title": "Geostrophic Ocean Flow",
    "category": "page",
    "text": ""
},

{
    "location": "ocean_flow.html#Geostrophic-Ocean-Flow-1",
    "page": "Geostrophic Ocean Flow",
    "title": "Geostrophic Ocean Flow",
    "category": "section",
    "text": "For a more realistic application, we consider an unsteady ocean surface velocity data set obtained from satellite altimetry measurements produced by SSALTO/DUACS and distributed by AVISO. The particular space-time window has been used several times in the literature."
},

{
    "location": "ocean_flow.html#Geodesic-vortices-1",
    "page": "Geostrophic Ocean Flow",
    "title": "Geodesic vortices",
    "category": "section",
    "text": "Here, we demonstrate how to detect material barriers to diffusive transport.using CoherentStructures\nimport JLD2, OrdinaryDiffEq, Plots\n###################### load and interpolate velocity data sets #############\nJLD2.@load(\"../../examples/Ocean_geostrophic_velocity.jld2\")\n\nUI, VI = interpolateVF(Lon,Lat,Time,UT,VT)\np = (UI,VI)\n\n############################ integration set up ################################\nq = 91\nt_initial = minimum(Time)\nt_final = t_initial + 90\nconst tspan = range(t_initial,stop=t_final,length=q)\nnx = 500\nny = Int(floor(0.6*nx))\nN = nx*ny\nxmin, xmax, ymin, ymax = -4.0, 6.0, -34.0, -28.0\nxspan, yspan = range(xmin,stop=xmax,length=nx), range(ymin,stop=ymax,length=ny)\nP = vcat.(xspan\',yspan)\nconst δ = 1.e-5\nmCG_tensor = u -> av_weighted_CG_tensor(interp_rhs,u,tspan,δ,p = p,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())\n\nC̅ = map(mCG_tensor,P)\nLCSparams = (.09, 0.5, 0.05, 0.5, 1.0, 60)\nvals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);The result is visualized as follows:using Statistics\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)\nl₁ = min.(λ₁,quantile(λ₁[:],0.999))\nl₁ = max.(λ₁,1e-2)\nl₂ = min.(λ₂,quantile(λ₂[:],0.995))\nfig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂),aspect_ratio=1,color=:viridis,\n            title=\"DBS-field and transport barriers\",xlims=(xmin, xmax),ylims=(ymin, ymax),leg=true)\nfor i in eachindex(orbits)\n    Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label=\"T = $(round(vals[i],digits=2))\")\nend\nPlots.plot(fig)"
},

{
    "location": "ocean_flow.html#FEM-based-methods-1",
    "page": "Geostrophic Ocean Flow",
    "title": "FEM-based methods",
    "category": "section",
    "text": "Here we showcase how the adaptive TO method can be used to calculate coherent sets.First we setup the problem:using CoherentStructures\nimport JLD2, OrdinaryDiffEq, Plots\n\n#Import and interpolate ocean dataset\n#The @load macro initializes Lon,Lat,Time,UT,VT\nJLD2.@load(\"../../examples/Ocean_geostrophic_velocity.jld2\")\nUI, VI = interpolateVF(Lon,Lat,Time,UT,VT)\np = (UI,VI)\n\n#Define a flow function from it\nt_initial = minimum(Time)\nt_final = t_initial + 90\ntimes = [t_initial,t_final]\nflow_map = u0 -> flow(interp_rhs, u0,times,p=p,tolerance=1e-5,solver=OrdinaryDiffEq.BS5())[end]\n\n#Setup the domain. We want to use zero Dirichlet boundary conditions here\nLL = [-4.0,-34.0]\nUR = [6.0,-28.0]\nctx = regularTriangularGrid((150,90), LL,UR)\nbdata = getHomDBCS(ctx,\"all\");For the TO method, we seek generalized eigenpairs involving a bilinear form of the form a_h(uv) = frac12 left(a_0(uv) + a_1(I_h u I_h v) right)Here a_0 is the weak form of the Laplacian on the initial domain, and a_1 is the weak form of the Laplacian on the final domain. The operator I_h is an interpolation operator onto the space of test functions on the final domain. For the adaptive TO method, we use pointwise nodal interpolation (i.e. collocation) and the mesh on the final domain is obtained by doing  a Delaunay triangulation on the images of the nodal points of the initial domain. This results in the representation matrix of I_h being the identity, so in matrix form we get:S = 05(S_0 + S_1)where S_0 is the stiffness matrix for the triangulation at initial time, and S_1 is the stiffness matrix for the triangulation at final time.M = assembleMassMatrix(ctx,bdata=bdata)\nS0 = assembleStiffnessMatrix(ctx)\nS1 = adaptiveTO(ctx,flow_map)\n\n#Average matrices and apply boundary conditions\nS = applyBCS(ctx,0.5(S0 + S1),bdata);We can now solve the eigenproblemusing Arpack\n\nλ,v = eigs(S, M,which=:SM, nev=6);We upsample the eigenfunctions and then clusterusing Clustering\n\nctx2 = regularTriangularGrid((200,120),LL,UR)\nv_upsampled = sample_to(v,ctx,ctx2,bdata=bdata)\n\n#Run k-means several times, keep the best result\nfunction iterated_kmeans(numiterations,args...)\n    best = kmeans(args...)\n    for i in 1:(numiterations-1)\n        cur = kmeans(args...)\n        if cur.totalcost < best.totalcost\n            best = cur\n        end\n    end\n    return best\nend\n\n\nn_partition = 4\nres = iterated_kmeans(20,permutedims(v_upsampled[:,1:(n_partition-1)]),n_partition)\nu = kmeansresult2LCS(res)\nu_combined = sum([u[:,i]*i for i in 1:n_partition])\nplot_u(ctx2, u_combined,200,200,\n    color=:viridis,colorbar=:none,title=\"$n_partition-partition of Ocean Flow\")"
},

{
    "location": "bickley.html#",
    "page": "Bickley Jet",
    "title": "Bickley Jet",
    "category": "page",
    "text": ""
},

{
    "location": "bickley.html#Bickley-Jet-1",
    "page": "Bickley Jet",
    "title": "Bickley Jet",
    "category": "section",
    "text": "The Bickley jet flow is a kinematic idealized model of a meandering zonal jet flanked above and below by counterrotating vortices. It was introduced by Rypina et al.; cf. also del‐Castillo‐Negrete and Morrison.The Bickley jet is described by a time-dependent velocity field arising from a stream-function. The corresponding velocity field is provided by the package and callable as bickleyJet.Instead of using the bickleyJet function to get this velocity field, we could also use the @velo_from_stream macro:using CoherentStructures\n# after this, \'bickley\' will reference a Dictionary of functions\n# access it via the desired signature. e.g. F = bickley[:(dU, U, p, t)]\n# for the right side of the equation of variation.\nbickley = @velo_from_stream stream begin\n    stream = psi₀ + psi₁\n    psi₀   = - U₀ * L₀ * tanh(y / L₀)\n    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term\n\n    re_sum_term =  Σ₁ + Σ₂ + Σ₃\n\n    Σ₁  =  ε₁ * cos(k₁*(x - c₁*t))\n    Σ₂  =  ε₂ * cos(k₂*(x - c₂*t))\n    Σ₃  =  ε₃ * cos(k₃*(x - c₃*t))\n\n    k₁ = 2/r₀      ; k₂ = 4/r₀    ; k₃ = 6/r₀\n\n    ε₁ = 0.0075    ; ε₂ = 0.15    ; ε₃ = 0.3\n    c₂ = 0.205U₀   ; c₃ = 0.461U₀ ; c₁ = c₃ + (√5-1)*(c₂-c₃)\n\n    U₀ = 62.66e-6  ; L₀ = 1770e-3 ; r₀ = 6371e-3\nend;"
},

{
    "location": "bickley.html#Geodesic-vortices-1",
    "page": "Bickley Jet",
    "title": "Geodesic vortices",
    "category": "section",
    "text": "Here we briefly demonstrate how to find material barriers to diffusive transport; see Geodesic elliptic material vortices for references and details.using CoherentStructures\nimport Tensors, OrdinaryDiffEq\n\nconst q = 81\nconst tspan = collect(range(0.,stop=3456000.,length=q))\nny = 120\nnx = div(ny*24,6)\nN = nx*ny\nxmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0\nxspan, yspan = range(xmin,stop=xmax,length=nx), range(ymin,stop=ymax,length=ny)\nP = vcat.(xspan\',yspan)\nconst δ = 1.e-6\nconst DiffTensor = Tensors.SymmetricTensor{2,2}([2., 0., 1/2])\nmCG_tensor = u -> av_weighted_CG_tensor(bickleyJet,u,tspan,δ,\n          D=DiffTensor,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())\n\nC̅ = map(mCG_tensor,P)\n\nLCSparams = (.1, 0.5, 0.04, 0.5, 1.8, 60)\nvals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);The result is visualized as follows:import Plots\nPlots.clibrary(:misc) #hide\nusing Statistics\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)\nl₁ = min.(λ₁,quantile(λ₁[:],0.999))\nl₁ = max.(λ₁,1e-2)\nl₂ = min.(λ₂,quantile(λ₂[:],0.995))\nbegin\n    fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂),aspect_ratio=1,color=:viridis,\n            title=\"DBS-field and transport barriers\",xlims=(0., 6.371π),ylims=(-3., 3.),leg=true)\n    for i in eachindex(orbits)\n        Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label=\"T = $(round(vals[i],digits=2))\")\n    end\nend\nPlots.plot(fig)"
},

{
    "location": "bickley.html#FEM-based-Methods-1",
    "page": "Bickley Jet",
    "title": "FEM-based Methods",
    "category": "section",
    "text": "Assume we have setup the bickley function using the @velo_from_stream macro like described above. As we are using a periodic domain in one direction:LL = [0.0,-3.0]; UR=[6.371π,3.0]\nctx = regularP2TriangularGrid((50,15),LL,UR,quadrature_order=2)\npredicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 6.371π) < 1e-10\nbdata = CoherentStructures.boundaryData(ctx,predicate,[]);Using a FEM-based method to compute coherent structures:using Arpack,Statistics\ncgfun = (x -> mean(pullback_diffusion_tensor(bickley, x,range(0.0,stop=40*3600*24,length=81),\n     1.e-8,tolerance=1.e-5)))\n\nK = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)\nM = assembleMassMatrix(ctx,bdata=bdata)\nλ, v = eigs(K,M,which=:SM, nev= 10)\n\nimport Plots\nplot_real_spectrum(λ)K-means clustering gives something we can plot:using Clustering\nn_partition = 8\nres = kmeans(permutedims(v[:,2:n_partition]),n_partition)\nu = kmeansresult2LCS(res)\nu_combined = sum([u[:,i]*i for i in 1:n_partition])\nplot_u(ctx, u_combined,200,200,bdata=bdata,\n    color=:rainbow,colorbar=:none,title=\"$n_partition-partition of Bickley Jet\")We ran kmeans only once. To get better results, kmeans should be run several times and only the run with the lowest objective function be kept. We also can upsample the eigenvectors to a finer grid, to obtain a better clustering:ctx2 = regularTriangularGrid((200,60),LL,UR)\nv_upsampled = sample_to(v, ctx,ctx2,bdata=bdata)\n\nfunction iterated_kmeans(numiterations,args...)\n    best = kmeans(args...)\n    for i in 1:(numiterations-1)\n        cur = kmeans(args...)\n        if cur.totalcost < best.totalcost\n            best = cur\n        end\n    end\n    return best\nend\n\nres = iterated_kmeans(20, permutedims(v_upsampled[:,2:n_partition]),n_partition) \nu = kmeansresult2LCS(res)\nu_combined = sum([u[:,i]*i for i in 1:n_partition])\nplot_u(ctx2, u_combined,400,400,\n    color=:rainbow,colorbar=:none,title=\"$n_partition-partition of Bickley Jet\")"
},

{
    "location": "standard_map.html#",
    "page": "Standard Map",
    "title": "Standard Map",
    "category": "page",
    "text": ""
},

{
    "location": "standard_map.html#Standard-Map-1",
    "page": "Standard Map",
    "title": "Standard Map",
    "category": "section",
    "text": "The \"standard map\" with parameter a is defined on a 2-dimensional doubly 2π-periodic domain by f(xy) = (x+ y+ a sin(x)y + acos(x)).For a = 0971635, the standard map is implemented in CoherentStructures.standardMap, its Jacobi-matrix in CoherentStructures.DstandardMap.See also Froyland & Junge (2015), who calculate Coherent Structures for this map.Below are some orbits of the standard mapusing CoherentStructures\nusing Random,Plots\nto_plot = []\nfor i in 1:50\n    Random.seed!(i)\n    x = rand(2)*2π\n    for i in 1:500\n        x = CoherentStructures.standardMap(x)\n        push!(to_plot,x)\n    end\nend\nPlots.scatter([x[1] for x in to_plot],[x[2] for x in to_plot],\n    m=:pixel,ms=1,aspect_ratio=1,legend=:none)Approximating the Dynamical Laplacian by FEM methods is straightforward:using Tensors, Plots, Arpack, Printf\nctx = regularTriangularGrid((100,100), [0.0,0.0],[2π,2π])\npred  = (x,y) -> peuclidean(x[1],y[1],2π) < 1e-9 && peuclidean(x[2],y[2],2π) < 1e-9\nbdata = boundaryData(ctx,pred) #Periodic boundary\n\nid2 = one(Tensors.Tensor{2,2}) # 2D identity tensor\ncgfun = x -> 0.5*(id2 +  Tensors.dott(Tensors.inv(CoherentStructures.DstandardMap(x))))\n\nK = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)\nM = assembleMassMatrix(ctx,lumped=false,bdata=bdata)\n@time λ, v = eigs(-1*K,M,which=:SM)\nPlots.plot(\n    [plot_u(ctx,v[:,i],bdata=bdata,title=@sprintf(\"\\\\lambda = %.3f\",λ[i]),\n        clim=(-0.25,0.25),colorbar=:none)\n         for i in 1:6]...)"
},

{
    "location": "basics.html#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "basics.html#Basics-1",
    "page": "Basics",
    "title": "Basics",
    "category": "section",
    "text": ""
},

{
    "location": "basics.html#Dynamical-Systems-Utilities-1",
    "page": "Basics",
    "title": "Dynamical Systems Utilities",
    "category": "section",
    "text": "CurrentModule = CoherentStructures"
},

{
    "location": "basics.html#CoherentStructures.@define_stream",
    "page": "Basics",
    "title": "CoherentStructures.@define_stream",
    "category": "macro",
    "text": "@define_stream(name::Symbol, code::Expr)\n\nDefine a scalar stream function on R^2. The defining code can be a series of definitions in an enclosing begin ... end-block and is treated as a series of symbolic substitutions. Starting from the symbol name, substitutions are performed until the resulting expression only depends on x, y and t.\n\nThe symbol name is not brought into the namespace. To access the resulting vector field and variational equation  use @velo_from_stream name and @var_velo_from_stream name\n\nThis is a convenience macro for the case where you want to use @velo_from_stream and @var_velo_from_stream without typing the code twice. If you only use one, you might as well use @velo_from_stream name code or @var_velo_from_stream directly.\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.@velo_from_stream",
    "page": "Basics",
    "title": "CoherentStructures.@velo_from_stream",
    "category": "macro",
    "text": "@velofromstream(name::Symbol, [code::Expr])\n\nGet the velocity field corresponding to a stream function on R^2.  The defining code can be a series of definitions (in an enclosing begin ... end-block and is treated as a series of symbolic substitutions. Starting from the symbol name, substitutions are performed until the resulting expression only depends on x, y and t.\n\nThe macro returns an anonymous function f(u,p,t) that returns a SVector{2} holding the vector field at u at time t.\n\nThe macro can be called without the code if the stream name has been define beforehand via @define_stream.\n\nExamples\n\njulia> using CoherentStructures\n\njulia> f = @velo_from_stream Ψ_ellipse begin\n               Ψ_ellipse = a*x^2 + b*y^2\n               a = t\n               b = 3\n           end\n(#3) #1 (generic function with 1 method)\n\njulia> f([1.0,1.0], nothing, 1.0)\n2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:\n -6.0\n  2.0\n\njulia> using CoherentStructures\n\njulia> @define_stream Ψ_circular begin\n           Ψ_circular = f(x) + g(y)\n           # naming of function variables\n           # does not matter:\n           f(a) = a^2\n           g(y) = y^2\n       end\n\njulia> f2 = @velo_from_stream Ψ_circular\n(#5) #1 (generic function with 1 method)\n\njulia> f2([1.0,1.0], nothing, 0.0)\n2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:\n -2.0\n  2.0\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.interpolateVF",
    "page": "Basics",
    "title": "CoherentStructures.interpolateVF",
    "category": "function",
    "text": "interpolateVF(xspan,yspan,tspan,u,v,interpolation_type=ITP.BSpline(ITP.Cubic(ITP.Free())))) -> UI, VI\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.interp_rhs",
    "page": "Basics",
    "title": "CoherentStructures.interp_rhs",
    "category": "function",
    "text": "interp_rhs(u, p, t) -> SA.SVector{2}\n\nDefines a 2D vector field that is readily usable for trajectory integration from vector field interpolants of the x- and y-direction, resp. It assumes that the interpolants are provided as a 2-tuple (UI, VI) via the parameter p. Here, UI and VI are the interpolants for the x- and y-components of the velocity field.\n\n\n\n\n\n"
},

{
    "location": "basics.html#Definition-of-vector-fields-1",
    "page": "Basics",
    "title": "Definition of vector fields",
    "category": "section",
    "text": "CoherentStructures.jl is set up for handling two- and three-dimensional dynamical systems only. For such low-dimensional flows it is advantageous (for top performance) to obey the following syntax:function vectorfield2d(u,p,t)\n    du1 = ... # equation for $\\dot{x}$\n    du2 = ... # equation for $\\dot{y}$\n    return StaticArrays.SVector{2}(du1,du2)\nendand correspondingly for three-dimensional ODEs:function vectorfield3d(u,p,t)\n    du1 = ... # equation for $\\dot{x}$\n    du2 = ... # equation for $\\dot{y}$\n    du3 = ... # equation for $\\dot{z}$\n    return StaticArrays.SVector{3}(du1,du2,du3)\nendFurthermore, there is a convenience macro to define two-dimensional velocity fields from stream functions.@define_stream\n@velo_from_streamIn fact, two of the predefined velocity fields, the rotating double gyre rot_double_gyre, and the Bickley jet flow bickleyJet, are generated from these macros.Another typical use case is when velocities are given as a data set. In this case, one first interpolates the velocity components with interpolateVF to obtain callable interpolation functions, say, UI and VI. The corresponding vector field is then interp_rhs, which the velocity interpolants enter via the parameter argument p; see below for examples.interpolateVF\ninterp_rhs"
},

{
    "location": "basics.html#CoherentStructures.flow",
    "page": "Basics",
    "title": "CoherentStructures.flow",
    "category": "function",
    "text": "flow(odefun,  u0, tspan; tolerance, p, solver) -> Vector{Vector}\n\nSolve the ODE with right hand side given by odefun and initial value u0. p is a parameter passed to odefun. tolerance is passed as both relative and absolute tolerance to the solver, which is determined by solver.\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.parallel_flow",
    "page": "Basics",
    "title": "CoherentStructures.parallel_flow",
    "category": "function",
    "text": "parallel_flow(flow_fun,P) -> Array\n\nApply the flow_fun to each element in P in parallel, if possible. Returns a 2D array with dimensions ((space dim x no. of time instances) x no. of trajectories), in which each column corresponds to a concatenated trajectory, i.e., represented in delay coordinates.\n\n\n\n\n\n"
},

{
    "location": "basics.html#Flow-maps-1",
    "page": "Basics",
    "title": "Flow maps",
    "category": "section",
    "text": "flowparallel_flow"
},

{
    "location": "basics.html#CoherentStructures.linearized_flow",
    "page": "Basics",
    "title": "CoherentStructures.linearized_flow",
    "category": "function",
    "text": "linearized_flow(odefun, x, tspan,δ; ...) -> Vector{Tensor{2,2}}\n\nCalculate derivative of flow map by finite differences if δ != 0. If δ==0, attempts to solve variational equation (odefun is assumed to be the rhs of variational equation in this case). Return time-resolved linearized flow maps.\n\n\n\n\n\n"
},

{
    "location": "basics.html#Linearized-flow-map-1",
    "page": "Basics",
    "title": "Linearized flow map",
    "category": "section",
    "text": "linearized_flow"
},

{
    "location": "basics.html#CoherentStructures.parallel_tensor",
    "page": "Basics",
    "title": "CoherentStructures.parallel_tensor",
    "category": "function",
    "text": "parallel_tensor(tensor_fun,P) -> Array{SymmetricTensor}\n\nComputes a tensor field via tensor_fun for each element of P, which is an array of vectors. tensor_fun is a function that takes initial conditions as input and returns a symmetric tensor. The final tensor field array has the same size as P.\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.CG_tensor",
    "page": "Basics",
    "title": "CoherentStructures.CG_tensor",
    "category": "function",
    "text": "CG_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor\n\nReturns the classic right Cauchy–Green strain tensor. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nkwargs...: are passed to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.mean_diff_tensor",
    "page": "Basics",
    "title": "CoherentStructures.mean_diff_tensor",
    "category": "function",
    "text": "mean_diff_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor\n\nReturns the averaged diffusion tensor at a point along a set of times. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nkwargs...: are passed to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.av_weighted_CG_tensor",
    "page": "Basics",
    "title": "CoherentStructures.av_weighted_CG_tensor",
    "category": "function",
    "text": "av_weighted_CG_tensor(odefun, u, tspan, δ; G, kwargs...) -> SymmetricTensor\n\nReturns the transport tensor of a trajectory, aka  time-averaged, di ffusivity-structure-weighted version of the classic right Cauchy–Green strain tensor. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nG: (constant) metric tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.pullback_tensors",
    "page": "Basics",
    "title": "CoherentStructures.pullback_tensors",
    "category": "function",
    "text": "pullback_tensors(odefun, u, tspan, δ; D, kwargs...) -> Tuple(Vector{SymmetricTensor},Vector{SymmetricTensor})\n\nReturns the time-resolved pullback tensors of both the diffusion and the metric tensor along a trajectory. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nD: (constant) diffusion tensor, metric tensor is computed via inversion; defaults to eye(2)\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.pullback_metric_tensor",
    "page": "Basics",
    "title": "CoherentStructures.pullback_metric_tensor",
    "category": "function",
    "text": "pullback_metric_tensor(odefun, u, tspan, δ; G, kwargs...) -> Vector{SymmetricTensor}\n\nReturns the time-resolved pullback tensors of the metric tensor along a trajectory, aka right Cauchy-Green strain tensor. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nG: (constant) metric tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.pullback_diffusion_tensor",
    "page": "Basics",
    "title": "CoherentStructures.pullback_diffusion_tensor",
    "category": "function",
    "text": "pullback_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...) -> Vector{SymmetricTensor}\n\nReturns the time-resolved pullback tensors of the diffusion tensor along a trajectory. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nD: (constant) diffusion tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.pullback_SDE_diffusion_tensor",
    "page": "Basics",
    "title": "CoherentStructures.pullback_SDE_diffusion_tensor",
    "category": "function",
    "text": "pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...) -> Vector{SymmetricTensor}\n\nReturns the time-resolved pullback tensors of the diffusion tensor in SDEs. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nD: (constant) diffusion tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.tensor_invariants",
    "page": "Basics",
    "title": "CoherentStructures.tensor_invariants",
    "category": "function",
    "text": "tensor_invariants(T) -> λ₁, λ₂, ξ₁, ξ₂, traceT, detT\n\nReturns pointwise invariants of the 2D symmetric tensor field T, i.e., smallest and largest eigenvalues, corresponding eigenvectors, trace and determinant.\n\nExample\n\nT = [Tensors.SymmetricTensor{2,2}(rand(3)) for i in 1:10, j in 1:20]\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(T)\n\nAll output variables have the same array arrangement as T; e.g., λ₁ is a 10x20 array with scalar entries.\n\n\n\n\n\n"
},

{
    "location": "basics.html#Cauchy-Green-and-other-pullback-tensors-1",
    "page": "Basics",
    "title": "Cauchy-Green and other pullback tensors",
    "category": "section",
    "text": "parallel_tensorCG_tensormean_diff_tensorav_weighted_CG_tensorpullback_tensors\npullback_metric_tensor\npullback_diffusion_tensor\npullback_SDE_diffusion_tensorA second-order symmetric two-dimensional tensor field may be diagonalized pointwise (\"eigendecomposition\") by the following function.tensor_invariants"
},

{
    "location": "basics.html#CoherentStructures.PEuclidean",
    "page": "Basics",
    "title": "CoherentStructures.PEuclidean",
    "category": "type",
    "text": "PEuclidean(L)\n\nCreate a Euclidean metric on a rectangular periodic domain. Periods per dimension are contained in the vector L. For dimensions without periodicity put Inf in the respective component.\n\nExample\n\njulia> using Distances\n\njulia> x, y, L = [0.0, 0.0], [0.7, 0.0], [0.5, Inf]\n([0.0, 0.0], [0.7, 0.0], [0.5, Inf])\n\njulia> evaluate(PEuclidean(L),x,y)\n0.19999999999999996\n\n\n\n\n\n"
},

{
    "location": "basics.html#CoherentStructures.STmetric",
    "page": "Basics",
    "title": "CoherentStructures.STmetric",
    "category": "type",
    "text": "STmetric(Smetric, dim, p)\n\nCreates a spatiotemporal, averaged in time metric.\n\nProperties\n\nSmetric is a metric as defined in the Distances package, e.g., Euclidean, PEuclidean, or Haversine;\ndim corresponds to the spatial dimension;\np corresponds to the kind of average applied to the vector of spatial distances:\np = Inf: maximum\np = 2: mean squared average\np = 1: arithmetic mean\np = -1: harmonic mean (does not yield a metric!)\np = -Inf: minimum (does not yield a metric!)\n\nExample\n\njulia> x, y = rand(10), rand(10)\n\njulia> Distances.evaluate(STmetric(Distances.Euclidean(),2,1),x,y)\n\n\n\n\n\n"
},

{
    "location": "basics.html#Distance-computations-1",
    "page": "Basics",
    "title": "Distance computations",
    "category": "section",
    "text": "To compute distances w.r.t. standard metrics, there exists the excellent Distance.jl package. The Euclidean distance between two points is computed by any of the following lines:using Distances\nx, y = rand(3), rand(3)\nevaluate(Euclidean(),x,y)\neuclidean(x,y)Other metrics of potential interest include Haversine(r), the geodesic distance of two points on the sphere with radius r. In CoherentStructures.jl, there are two more types of metrics implemented:PEuclidean\nSTmetricThat is, the distance on a periodic torus/cylinder, and a spatiotemporal metric that interprets vectors as concatenated trajectories, applies the spatial metric to each time instance and reduces the vector of spatial distances by computing its l_p-norm. Notably, p may be any \"real\" number, including Inf and -Inf for the maximum- and \"minimum\"-norm. The spatiotemporal metric is a mathematical metric only for pgeq 1, in which case it smoothly operates with efficient sparsification methods like BallTree and inrange as implemented in the NearestNeighbors.jl  package."
},

{
    "location": "fem.html#",
    "page": "FEM-based methods",
    "title": "FEM-based methods",
    "category": "page",
    "text": ""
},

{
    "location": "fem.html#FEM-based-Methods-1",
    "page": "FEM-based methods",
    "title": "FEM-based Methods",
    "category": "section",
    "text": "These methods rely on the theory outlined by Froyland\'s Dynamical Laplacian and the Geometric Heat Flow of Karrasch & Keller.The Laplace-like operators are best discretized by finite-element-based methods, see this paper by Froyland & Junge.This involves the discretization of the average of a one-parameter family of Laplace operators of the form:Delta^dyn = sum_t in mathcal T P_t^* Delta P_tfor a finite series of times mathcal T, where P_t is the transfer-operator for the flow at time t (in volume-preserving flows).The resulting operator is both symmetric and uniformly elliptic. Eigenfunctions of Delta^dyn can be used to find Lagrangian Coherent Structures.See the Examples section for examples of how these methods can be used."
},

{
    "location": "fem.html#Features-1",
    "page": "FEM-based methods",
    "title": "Features",
    "category": "section",
    "text": ""
},

{
    "location": "fem.html#CG-and-TO-methods-1",
    "page": "FEM-based methods",
    "title": "CG and TO methods",
    "category": "section",
    "text": "The standard Galerkin formulation of the weak dynamical Laplace is refered to as the CG-method here, due to the fact that the inverse Cauchy-Green tensor appears in the weak formulation. This gives a bilinear form overline a(uv) = sum_t in mathcal Ta^t(P_t u P_t v) Here P_t is the Transfer-Operator (or pushforward) to time-t, and a^t is the weak-form of the Laplacian on the range of the time-t map being considered.   There are also a range of Transfer-Operator based approaches implemented here. These approximate the weak form of the Dynamical-Laplace by a bilinear-form:tilde a_h(uv) = sum_t in mathcal T a^t(I_hP_t u I_h P_t v)where I_h is a suitable interpolation operator depending on the mesh-width h. Options for I_h implemented in this package are:Collocation (pointwise interpolation)...\nPoints used are mesh points from domain grid (\"adaptive TO\")\nPoints usedare arbitrary(\"non-adaptive TO\")\nthe L^2-orthogonal projection onto a FEM-space\nUsing the forwards flow map (currently gives poor results)\nUsing the inverse flow mapNote that the L^2-Galerkin methods currently perform very poorly on larger problems.For more details, see this paper."
},

{
    "location": "fem.html#Grids-1",
    "page": "FEM-based methods",
    "title": "Grids",
    "category": "section",
    "text": "Various types of regular and irregular meshes (with Delaunay triangulation using VoronoiDelaunay.jl ) are supported. These are based on the corresponding elements from JuAFEM.jl and include:Triangular P1-Lagrange elements in 2D (all methods)\nQuadrilateral P1-Lagrange elements in 2D (all methods except adaptive TO)\nTriangular and Quadrilateral P2-Lagrange elements in 2D (all methods except adaptive TO)\nTetrahedral P1-Lagrange elements in 3D (only CG method tested, non-adaptive TO might work also)"
},

{
    "location": "fem.html#The-gridContext-Type-1",
    "page": "FEM-based methods",
    "title": "The gridContext Type",
    "category": "section",
    "text": "The FEM-based methods of CoherentStructures.jl rely heavily on the JuAFEM.jl package. This package is very low-level and does not provide point-location/plotting functionality. To be able to more conveniently work with the specific types of grids that we need, all necessary variables for a single grid are combined in a gridContext structure - including the grid points, the quadrature formula used and the type of element used (e.g. Triangular P1, Quadrilateral P2, etc..). This makes it easier to assemble mass/stiffness matrices, and provides an interface for point-location and plotting.In this documentation, the variable name ctx is exclusively used for gridContext objects.See also Constructing Grids in the FEM-API section."
},

{
    "location": "fem.html#Node-ordering-and-dof-ordering-1",
    "page": "FEM-based methods",
    "title": "Node ordering and dof ordering",
    "category": "section",
    "text": "Finite Element methods work with degrees of freedom (dof), which are elements of some dual space. For nodal finite elements, these correspond to evaluation functionals at the nodes of the grid.The nodes of the grid can be obtained in the following way [n.x for n in ctx.grid.nodes]. However, most of the methods of this package do not return results in this order, but instead use JuAFEM.jl\'s dof-ordering.See also the documentation in dof2node and CoherentStructures.gridContextWhen working with (non-natural) Boundary Conditions, the ordering is further changed, due to there being fewer degrees of freedom in total."
},

{
    "location": "fem.html#Assembly-1",
    "page": "FEM-based methods",
    "title": "Assembly",
    "category": "section",
    "text": "See Stiffness and Mass Matrices from the FEM-API section."
},

{
    "location": "fem.html#Evaluating-Functions-in-the-Approximation-Space-1",
    "page": "FEM-based methods",
    "title": "Evaluating Functions in the Approximation Space",
    "category": "section",
    "text": "given a series of coefficients that represent a function in the approximation space, to evaluate a function at a point, use the evaluate_function_from_nodevals or evaluate_function_from_dofvals functions.using CoherentStructures #hide\nusing Plots\nctx = regularP2TriangularGrid((10,10))\nu = zeros(ctx.n)\nu[45] = 1.0\nPlots.heatmap(range(0,stop=1,length=200),range(0,stop=1,length=200), (x,y)->evaluate_function_from_nodevals(ctx,u,[x,y]))For more details, consult the API: evaluate_function_from_dofvals, evaluate_function_from_nodevals"
},

{
    "location": "fem.html#Nodal-Interpolation-1",
    "page": "FEM-based methods",
    "title": "Nodal Interpolation",
    "category": "section",
    "text": "To perform nodal interpolation of a grid, use the nodal_interpolation function."
},

{
    "location": "fem.html#Boundary-Conditions-1",
    "page": "FEM-based methods",
    "title": "Boundary Conditions",
    "category": "section",
    "text": "To use something other than the natural homogeneous von Neumann boundary conditions, the CoherentStructures.boundaryData type can be used. This currently supports combinations of homogeneous Dirichlet and periodic boundary conditions.Homogeneous Dirichlet BCs require rows and columns of the stiffness/mass matrices to be deleted\nPeriodic boundary conditions require rows and columns of the stiffness/mass matrices to be added to each other.This means that the coefficient vectors for elements of the approximation space that satisfy the boundary conditions are potentially smaller and in a different order. Given a bdata argument, functions like plot_u will take this into account."
},

{
    "location": "fem.html#Constructing-Boundary-Conditions-1",
    "page": "FEM-based methods",
    "title": "Constructing Boundary Conditions",
    "category": "section",
    "text": "Natural von-Neumann boundary conditions can be constructed with: boundaryData() and are generally the defaultHomogeneous Dirichlet boundary conditions can be constructed with the getHomDBCS(ctx,[which=\"all\"]) function. The optional which parameter is a vector of strings, corresponding to JuAFEM face-sets, e.g. getHomDBCS(ctx,which=[\"left\",\"right\"])Periodic boundary conditions are constructed by calling boundaryData(ctx,predicate,[which_dbc=[]]). The argument predicate is a function that should return true if and only if two points should be identified. Due to floating-point rounding errors, note that using exact comparisons (==) should be avoided. Only points that are in JuAFEM.jl boundary facesets are considered. If this is too restrictive, use the boundaryData(dbc_dofs, periodic_dofs_from,periodic_dofs_to) constructor.For details, see boundaryData"
},

{
    "location": "fem.html#Example-1",
    "page": "FEM-based methods",
    "title": "Example",
    "category": "section",
    "text": "Here we apply Homogeneous DBC to top and bottom, and identify the left and right side:using CoherentStructures\nctx = regularQuadrilateralGrid((10,10))\npredicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1],p2[1],1.0) < 1e-10\nbdata = boundaryData(ctx,predicate,[\"top\",\"bottom\"])\nu = ones(nDofs(ctx,bdata))\nu[20] = 2.0; u[38] = 3.0; u[56] = 4.0\nplot_u(ctx,u,200,200,bdata=bdata,colorbar=:none)To apply boundary conditions to a stiffness/mass matrix, use the applyBCS function. Note that assembleStiffnessMatrix and assembleMassMatrix take a bdata argument that does this internally."
},

{
    "location": "fem.html#Plotting-and-Videos-1",
    "page": "FEM-based methods",
    "title": "Plotting and Videos",
    "category": "section",
    "text": "There are some helper functions that exist for making plots and videos of functions on grids. These rely on the Plots.jl library. Plotting recipes are unfortunately not implemented.The simplest way to plot is using the plot_u function. Plots and videos of eulerian plots like f circ Phi^0_t can be made with the plot_u_eulerian and  eulerian_videos functions."
},

{
    "location": "fem.html#Parallelisation-1",
    "page": "FEM-based methods",
    "title": "Parallelisation",
    "category": "section",
    "text": "Many of the plotting functions support parallelism internally. Tensor fields can be constructed in parallel, and then passed to assembleStiffnessMatrix. For an example that does this, see TODO: Add this example"
},

{
    "location": "fem.html#FEM-API-1",
    "page": "FEM-based methods",
    "title": "FEM-API",
    "category": "section",
    "text": "CurrentModule = CoherentStructures"
},

{
    "location": "fem.html#CoherentStructures.assembleStiffnessMatrix",
    "page": "FEM-based methods",
    "title": "CoherentStructures.assembleStiffnessMatrix",
    "category": "function",
    "text": "assembleStiffnessMatrix(ctx,A,[p; bdata])\n\nAssemble the stiffness-matrix for a symmetric bilinear form\n\na(uv) = int nabla u(x)cdot A(x)nabla v(x)f(x) dx\n\nThe integral is approximated using quadrature. A is a function that returns a Tensors.SymmetricTensor{2,dim} and has one of the following forms:\n\nA(x::Vector{Float64})\nA(x::Vec{dim})\nA(x::Vec{dim}, index::Int, p). Here x is equal to ctx.quadrature_points[index], and p is that which is passed to assembleStiffnessMatrix\n\nThe ordering of the result is in dof order, except that boundary conditions from bdata are applied. The default is natural boundary conditions.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.assembleMassMatrix",
    "page": "FEM-based methods",
    "title": "CoherentStructures.assembleMassMatrix",
    "category": "function",
    "text": "assembleMassMatrix(ctx;[bdata,lumped=false])\n\nAssemble the mass matrix\n\nM_ij = int varphi_j(x) varphi_i(x) f(x)dlambda^d\n\nThe integral is approximated using numerical quadrature. The values of f(x) are taken from ctx.mass_weights, and should be ordered in the same way as ctx.quadrature_points\n\nThe result is ordered in a way so as to be usable with a stiffness matrix with boundary data bdata.\n\nReturns a lumped mass matrix if lumped==true.\n\nExample\n\nctx.mass_weights = map(f, ctx.quadrature_points)\nM = assembleMassMatrix(ctx)\n\n\n\n\n\n"
},

{
    "location": "fem.html#Stiffness-and-Mass-Matrices-1",
    "page": "FEM-based methods",
    "title": "Stiffness and Mass Matrices",
    "category": "section",
    "text": "assembleStiffnessMatrix\nassembleMassMatrix"
},

{
    "location": "fem.html#CoherentStructures.regular2DGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regular2DGrid",
    "category": "function",
    "text": "regular2DGrid(gridType, numnodes, LL=[0.0,0.0],UR=[1.0,1.0];quadrature_order=default_quadrature_order)\n\nConstructs a regular grid. gridType should be from CoherentStructures.regular2DGridTypes\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.regularTriangularGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularTriangularGrid",
    "category": "function",
    "text": "regularTriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0], quadrature_order=default_quadrature_order)\n\nCreate a regular P1 triangular grid on a rectangle; it does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.regularP2TriangularGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2TriangularGrid",
    "category": "function",
    "text": "regularP2TriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)\n\nCreate a regular P2 triangular grid on a Rectangle. Does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.regularQuadrilateralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularQuadrilateralGrid",
    "category": "function",
    "text": "regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)\n\nCreate a regular P1 quadrilateral grid on a Rectangle.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.regularP2QuadrilateralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2QuadrilateralGrid",
    "category": "function",
    "text": "regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0], quadrature_order=default_quadrature_order)\n\nCreate a regular P2 quadrilateral grid on a rectangle.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.regularTetrahedralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularTetrahedralGrid",
    "category": "function",
    "text": "regularTetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)\n\nCreate a regular P1 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.regularP2TetrahedralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2TetrahedralGrid",
    "category": "function",
    "text": "regularP2TetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)\n\nCreate a regular P2 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.gridContext",
    "page": "FEM-based methods",
    "title": "CoherentStructures.gridContext",
    "category": "type",
    "text": "struct gridContext<dim>\n\nStores everything needed as \"context\" to be able to work on a FEM grid based on the JuAFEM package. Adds a point-locator API which facilitates plotting functions defined on the grid within Julia.\n\nFields\n\ngrid::JuAFEM.Grid, ip::JuAFEM.Interpolation, qr::JuAFEM.QuadratureRule - See the JuAFEM package\nloc::CellLocator object used for point-location on the grid.\nnode_to_dof::Vector{Int}  lookup table for dof index of a node\ndof_to_node::Vector{Int}  inverse of nodetodof\nn::Int number of nodes on the grid\nm::Int number of elements (e.g. triangles,quadrilaterals, ...) on the grid\nquadrature_points::Vector{Vec{dim,Float64}} All quadrature points on the grid, in a fixed order.\nmass_weights::Vector{Float64} Weighting for mass matrix\nspatialBounds If available, the corners of a bounding box of a domain. For regular grids, the bounds are tight.\nnumberOfPointsInEachDirection For regular grids, how many (non-interior) nodes make up the regular grid.\ngridType A string describing what kind of grid this is (e.g. \"regular triangular grid\")\n\n\n\n\n\n"
},

{
    "location": "fem.html#Constructing-Grids-1",
    "page": "FEM-based methods",
    "title": "Constructing Grids",
    "category": "section",
    "text": "There are several helper functions available for constructing grids. The simplest is:regular2DGridSupported values for the gridType argument are:using CoherentStructures #hide\nCoherentStructures.regular2DGridTypesThe following functions are conceptually similar:regularTriangularGrid\n#regularDelaunayGrid #TODO 1.0\nregularP2TriangularGrid\n#regularP2DelaunayGrid #TODO 1.0\nregularQuadrilateralGrid\nregularP2QuadrilateralGridIn 3D we haveregularTetrahedralGrid\nregularP2TetrahedralGridAll of these methods return a gridContext object.CoherentStructures.gridContext"
},

{
    "location": "fem.html#Irregular-grids-1",
    "page": "FEM-based methods",
    "title": "Irregular grids",
    "category": "section",
    "text": "The constructors for CoherentStructures.gridContext, including one for irregular Delaunay grids, are not exported by default, the documentation is available through the REPL:help?> (::Type{CoherentStructures.gridContext{2}})"
},

{
    "location": "fem.html#CoherentStructures.boundaryData",
    "page": "FEM-based methods",
    "title": "CoherentStructures.boundaryData",
    "category": "type",
    "text": "mutable struct boundaryData\n\nRepresent (a combination of) homogeneous Dirichlet and periodic boundary conditions. Fields:\n\ndbc_dofs list of dofs that should have homogeneous Dirichlet boundary conditions. Must be sorted.\nperiodic_dofs_from and periodic_dofs_to are both Vector{Int}. The former must be strictly increasing, both must be the same length. periodic_dofs_from[i] is identified with periodic_dofs_to[i]. periodic_dofs_from[i] must be strictly larger than periodic_dofs_to[i]. Multiple dofs can be identified with the same dof. If some dof is identified with another dof and one of them is in dbc_dofs, both points must be in dbc_dofs\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.getHomDBCS",
    "page": "FEM-based methods",
    "title": "CoherentStructures.getHomDBCS",
    "category": "function",
    "text": "getHomDBCS(ctx,which=\"all\")\n\nReturn boundaryData object corresponding to homogeneous Dirichlet Boundary Conditions for a set of facesets. which=\"all\" is shorthand for [\"left\",\"right\",\"top\",\"bottom\"].\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.undoBCS",
    "page": "FEM-based methods",
    "title": "CoherentStructures.undoBCS",
    "category": "function",
    "text": "undoBCS(ctx,u,bdata)\n\nGiven a vector u in dof order with boundary conditions applied, return the corresponding u in dof order without the boundary conditions.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.applyBCS",
    "page": "FEM-based methods",
    "title": "CoherentStructures.applyBCS",
    "category": "function",
    "text": "applyBCS(ctx,K,bdata)\n\nApply the boundary conditions from bdata to the ctx.n by ctx.n sparse matrix K.\n\n\n\n\n\n"
},

{
    "location": "fem.html#Boundary-Conditions-API-1",
    "page": "FEM-based methods",
    "title": "Boundary Conditions API",
    "category": "section",
    "text": "boundaryData\ngetHomDBCS\nundoBCS\napplyBCS"
},

{
    "location": "fem.html#CoherentStructures.dof2node",
    "page": "FEM-based methods",
    "title": "CoherentStructures.dof2node",
    "category": "function",
    "text": "dof2node(ctx,u)\n\nInterprets u as an array of coefficients ordered in dof order, and reorders them to be in node order.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.getDofCoordinates",
    "page": "FEM-based methods",
    "title": "CoherentStructures.getDofCoordinates",
    "category": "function",
    "text": "getDofCoordinates(ctx,dofindex)\n\nReturn the coordinates of the node corresponding to the dof with index dofindex\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.evaluate_function_from_dofvals",
    "page": "FEM-based methods",
    "title": "CoherentStructures.evaluate_function_from_dofvals",
    "category": "function",
    "text": "evaluate_function_from_dofvals(ctx,dofvals,x_in; [outside_value=0,project_in=false])\n\nEvaluate a function in the approximation space at the point x_in. If x_in is out of points, return outside_value. If project_in is true, points not within ctx.spatialBounds are first projected into the domain.\n\nThe coefficients in nodevals are interpreted to be in dof order.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.evaluate_function_from_nodevals",
    "page": "FEM-based methods",
    "title": "CoherentStructures.evaluate_function_from_nodevals",
    "category": "function",
    "text": "evaluate_function_from_nodevals(ctx,nodevals,x_in; [outside_value=0, project_in=false])\n\nLike evaluate_function_from_dofvals, but the coefficients from nodevals are assumed to be in node order.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.nodal_interpolation",
    "page": "FEM-based methods",
    "title": "CoherentStructures.nodal_interpolation",
    "category": "function",
    "text": "nodal_interpolation(ctx,f)\n\nPerform nodal interpolation of a function. Returns a vector of coefficients in dof order\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.getH",
    "page": "FEM-based methods",
    "title": "CoherentStructures.getH",
    "category": "function",
    "text": "getH(ctx)\n\nReturn the mesh width of a regular grid.\n\n\n\n\n\n"
},

{
    "location": "fem.html#Helper-functions-1",
    "page": "FEM-based methods",
    "title": "Helper functions",
    "category": "section",
    "text": "dof2node\ngetDofCoordinatesevaluate_function_from_dofvals\nevaluate_function_from_nodevalsnodal_interpolationgetH"
},

{
    "location": "fem.html#Plotting-API-1",
    "page": "FEM-based methods",
    "title": "Plotting API",
    "category": "section",
    "text": ""
},

{
    "location": "fem.html#CoherentStructures.plot_u",
    "page": "FEM-based methods",
    "title": "CoherentStructures.plot_u",
    "category": "function",
    "text": "plot_u(ctx, dof_vals, nx, ny; bdata=nothing, kwargs...)\n\nPlot the function with coefficients (in dof order, possible boundary conditions in bdata) given by dof_vals on the grid ctx. The domain to be plotted on is given by ctx.spatialBounds. The function is evaluated on a regular nx by ny grid, the resulting plot is a heatmap. Keyword arguments are passed down to plot_u_eulerian, which this function calls internally.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.plot_u_eulerian",
    "page": "FEM-based methods",
    "title": "CoherentStructures.plot_u_eulerian",
    "category": "function",
    "text": "plot_u_eulerian(ctx,dof_vals,inverse_flow_map,\n    LL,UR,nx,ny,\n    euler_to_lagrange_points=nothing, only_get_lagrange_points=false,\n    z=nothing,\n    postprocessor=nothing,\n    bdata=nothing, ....)\n\nPlot a heatmap of a function in eulerian coordinates, i.e. the pushforward of f. This is given by f circ Phi^-1, f is a function defined on the grid ctx, represented by coefficients given by dof_vals (with possible boundary conditions given in bdata)\n\nThe argument inverse_flow_map is Phi^-1.\n\nThe resulting plot is on a regular nx by ny grid on the grid with lower left corner LL and upper right corner UR.\n\nPoints that fall outside of the domain represented by ctx are plotted as NaN, which results in transparency.\n\nOne can pass values to be plotted directly by providing them in an array in the argument z. postprocessor can modify the values being plotted, return_scalar_field results in these values being returned.  See the source code for further details.  Additional arguments are passed to Plots.heatmap\n\nInverse flow maps are computed in parallel if there are multiple workers.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.eulerian_videos",
    "page": "FEM-based methods",
    "title": "CoherentStructures.eulerian_videos",
    "category": "function",
    "text": " eulerian_videos(ctx, us, inverse_flow_map_t, t0,tf, nx,ny,nt, LL,UR, num_videos=1;\n    extra_kwargs_fun=nothing, ...)\n\nCreate num_videos::Int videos in eulerian coordinates, i.e. where the time t is varied, plot f_i circ Phi_t^0 for f_1 dots.\n\nus(i,t) is a vector of dofs to be plotted at time t for the ith video.\n\ninverse_flow_map_t(t,x) is Phi_t^0(x)\n\nt0, tf  are initial and final time. Spatial bounds are given by LL,UR\n\nnx,ny,nt give the number of points in each direction.\n\nextra_kwargs_fun(i,t) can be used to provide additional keyword arguments to Plots.heatmap()\n\nAdditional kwargs are passed down to plot_eulerian_video\n\nAs much as possible is done in parallel.\n\nReturns a Vector of iterables result. Call Plots.animate(result[i]) to get an animation.\n\n\n\n\n\n"
},

{
    "location": "fem.html#CoherentStructures.eulerian_video",
    "page": "FEM-based methods",
    "title": "CoherentStructures.eulerian_video",
    "category": "function",
    "text": "eulerian_video(ctx, u, inverse_flow_map_t,t0,tf, nx, ny, nt, LL, UR;extra_kwargs_fun=nothing,...)\n\nLike eulerian_videos, but u(t) is a vector of dofs, and extra_kwargs_fun(t) gives extra keyword arguments. Returns only one result, on which Plots.animate() can be applied.\n\n\n\n\n\n"
},

{
    "location": "fem.html#FEM-1",
    "page": "FEM-based methods",
    "title": "FEM",
    "category": "section",
    "text": "plot_u\nplot_u_eulerian\neulerian_videos\neulerian_video"
},

{
    "location": "fem.html#CoherentStructures.plot_ftle",
    "page": "FEM-based methods",
    "title": "CoherentStructures.plot_ftle",
    "category": "function",
    "text": "plot_ftle(odefun,p,tspan,LL,UR,nx,ny;\n    δ=1e-9,tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),\n    existing_plot=nothing,flip_y=false, check_inbounds=always_true, pass_on_errors=false)\n\nMake a heatmap of a FTLE field using finite differences. If existing_plot is given a value, plot using heatmap! on top of it. If flip_y is true, then flip the y-coordinate (needed sometimes due to a bug in Plots). Points where check_inbounds(x[1],x[2],p) == false are set to NaN (i.e. transparent). Unless pass_on_errors is set to true, errors from calculating FTLE values are caught and ignored.\n\n\n\n\n\n"
},

{
    "location": "fem.html#Other-plotting-utilities-1",
    "page": "FEM-based methods",
    "title": "Other plotting utilities",
    "category": "section",
    "text": "plot_ftle"
},

{
    "location": "fem.html#Defaults-1",
    "page": "FEM-based methods",
    "title": "Defaults",
    "category": "section",
    "text": "const default_quadrature_order=5\nconst default_solver = OrdinaryDiffEq.BS5()"
},

{
    "location": "elliptic.html#",
    "page": "Geodesic vortices",
    "title": "Geodesic vortices",
    "category": "page",
    "text": ""
},

{
    "location": "elliptic.html#Geodesic-elliptic-material-vortices-1",
    "page": "Geodesic vortices",
    "title": "Geodesic elliptic material vortices",
    "category": "section",
    "text": "CurrentModule = CoherentStructuresThe following functions implement an LCS methodology developed in the following papers:Haller & Beron-Vera, 2012\nHaller & Beron-Vera, 2013\nKarrasch, Huhn, and Haller, 2015The present code is structurally inspired–albeit partially significantly improved–by Alireza Hadjighasem\'s MATLAB implementation, which was written in the context of the SIAM Review paper. Depending on the indefinite metric tensor field used, the functions below yield the following types of coherent structures:black-hole/Lagrangian coherent vortices (Haller & Beron-Vera, 2012)\nelliptic objective Eulerian coherent structures (OECSs) (Serra & Haller, 2016)\nmaterial diffusive transport barriers (Haller, Karrasch, and Kogelbauer, 2018)The general procedure is the following. Assume T is the symmetric tensor field of interest, say, (i) the Cauchy-Green strain tensor field C, (ii) the rate-of-strain tensor field S, or (iii) the averaged diffusion-weighted Cauchy-Green tensor field barC_D; cf. the references above. Denote by 0lambda_1leqlambda_2 the eigenvalue and by xi_1 and xi_2 the corresponding eigenvector fields of T. Then the direction fields of interest are given byeta_lambda^pm = sqrtfraclambda_2 - lambdalambda_2-lambda_1xi_1 pm sqrtfraclambda - lambda_1lambda_2-lambda_1xi_2Tensor singularities are defined as points at which lambda_2=lambda_1, i.e., at which the two characteristic directions xi_1 and xi_2 are not well-defined. Then, the algorithm put forward in Karrasch et al., 2015 consists of the following steps:locate singularities of the tensor field T;\ndetermine the type of the singularities (only non-degenerate types like wedges and trisectors are detected);\nlook for wedge pairs that are reasonably isolated from other singularities;\nplace an eastwards oriented Poincaré section at the pair center;\nfor each point on the discretized Poincaré section, scan through the parameter space such that the corresponding η-orbit closes at that point;\nfor each Poincaré section, take the outermost closed orbit as the coherent vortex barrier (if there exist any)."
},

{
    "location": "elliptic.html#CoherentStructures.ellipticLCS",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.ellipticLCS",
    "category": "function",
    "text": "ellipticLCS(T,xspan,yspan,p)\n\nComputes elliptic LCSs as null-geodesics of the Lorentzian metric tensor field given by shifted versions of T on the 2D computational grid spanned by xspan and yspan. p is a tuple of the following parameters (in that order):\n\nradius: radius in tensor singularity type detection,\nMaxWdgeDist: maximal distance to nearest wedge-type singularity,\nMinWedgeDist: minimal distance to nearest wedge-type singularity,\nMin2ndDist: minimal distance to second-nearest wedge-type singularity,\np_length: length of Poincaré section,\nn_seeds: number of seeding points along the Poincaré section,\n\nReturns a list of tuples, each tuple containing\n\nthe parameter value λ in the η-formula,\nthe sign used in the η-formula,\nthe outermost closed orbit for the corresponding λ and sign.\n\n\n\n\n\n"
},

{
    "location": "elliptic.html#CoherentStructures.singularity_location_detection",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.singularity_location_detection",
    "category": "function",
    "text": "singularity_location_detection(T,xspan,yspan)\n\nDetects tensor singularities of the tensor field T, given as a matrix of SymmetricTensor{2,2}. xspan and yspan correspond to the uniform grid vectors over which T is given. Returns a list of static 2-vectors.\n\n\n\n\n\n"
},

{
    "location": "elliptic.html#CoherentStructures.singularity_type_detection",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.singularity_type_detection",
    "category": "function",
    "text": "singularity_type_detection(singularity,T,radius,xspan,yspan)\n\nDetermines the singularity type of the singularity candidate singularity by querying the tensor eigenvector field of T in a circle of radius radius around the singularity. xspan and yspan correspond to the computational grid. Returns 1 for a trisector, -1 for a wedge, and 0 otherwise.\n\n\n\n\n\n"
},

{
    "location": "elliptic.html#CoherentStructures.detect_elliptic_region",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.detect_elliptic_region",
    "category": "function",
    "text": "detect_elliptic_region(singularities,singularityTypes,MaxWedgeDist,MinWedgeDist,Min2ndDist)\n\nDetermines candidate regions for closed tensor line orbits.\n\nsingularities: list of all singularities\nsingularityTypes: list of corresponding singularity types\nMaxWedgeDist: maximum distance to closest wedge\nMinWedgeDist: minimal distance to closest wedge\nMin2ndDist: minimal distance to second closest wedge\n\nReturns a list of vortex centers.\n\n\n\n\n\n"
},

{
    "location": "elliptic.html#CoherentStructures.set_Poincaré_section",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.set_Poincaré_section",
    "category": "function",
    "text": "set_Poincaré_section(vc,p_length,n_seeds,xspan,yspan)\n\nGenerates a horizontal Poincaré section, centered at the vortex center vc of length p_length consisting of n_seeds starting at 0.2*p_length eastwards. All points are guaranteed to lie in the computational domain given by xspan and yspan.\n\n\n\n\n\n"
},

{
    "location": "elliptic.html#CoherentStructures.compute_outermost_closed_orbit",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.compute_outermost_closed_orbit",
    "category": "function",
    "text": "compute_outermost_closed_orbit(pSection, T, xspan, yspan; pmin = .7, pmax = 1.3)\n\nCompute the outermost closed orbit for a given Poincaré section pSection, tensor field T, where the total computational domain is spanned by xspan and yspan. Keyword arguments pmin and pmax correspond to the range of shift parameters in which closed orbits are sought.\n\n\n\n\n\n"
},

{
    "location": "elliptic.html#Function-documentation-1",
    "page": "Geodesic vortices",
    "title": "Function documentation",
    "category": "section",
    "text": "The fully automated meta-function is the following:ellipticLCSEssentially, it calls sequentially the following functions.singularity_location_detectionsingularity_type_detectiondetect_elliptic_regionset_Poincaré_sectioncompute_outermost_closed_orbit"
},

{
    "location": "Laplace.html#",
    "page": "Graph Laplacian-based methods",
    "title": "Graph Laplacian-based methods",
    "category": "page",
    "text": ""
},

{
    "location": "Laplace.html#Graph-Laplacian/diffusion-maps-based-LCS-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Graph Laplacian/diffusion maps-based LCS methods",
    "category": "section",
    "text": "CurrentModule = CoherentStructuresCite a couple of important papers:Shi & Malik, Normalized cuts and image segmentation, 2000\nCoifman & Lafon, Diffusion maps, 2006\nMarshall & Hirn, Time coupled diffusion maps, 2018In the LCS context, we havesomewhat related Froyland & Padberg-Gehle, 2015\nHadjighasem et al., 2016\nBanisch & Koltai, 2017\nRypina et al., 2017/Padberg-Gehle & Schneide, 2018\nDe Diego et al., 2018"
},

{
    "location": "Laplace.html#Function-documentation-1",
    "page": "Graph Laplacian-based methods",
    "title": "Function documentation",
    "category": "section",
    "text": ""
},

{
    "location": "Laplace.html#CoherentStructures.KNN",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.KNN",
    "category": "type",
    "text": "KNN(k)\n\nDefines the KNN (k-nearest neighbors) sparsification method. In this approach, first k nearest neighbors are sought. In the final graph Laplacian, only those particle pairs are included which are contained in some k-neighborhood.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.mutualKNN",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.mutualKNN",
    "category": "type",
    "text": "mutualKNN(k)\n\nDefines the mutual KNN (k-nearest neighbors) sparsification method. In this approach, first k nearest neighbors are sought. In the final graph Laplacian, only those particle pairs are included which are mutually contained in each others k-neighborhood.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.neighborhood",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.neighborhood",
    "category": "type",
    "text": "neighborhood(ε)\n\nDefines the ε-neighborhood sparsification method. In the final graph Laplacian, only those particle pairs are included which have distance less than ε.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#Sparsification-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Sparsification methods",
    "category": "section",
    "text": "Two commonly used sparsification methods are implemented for use with various graph Laplacian methods, see below.KNN\nmutualKNN\nneighborhoodOther sparsification methods can be implemented by defining a corresponding sparseaffinitykernel instance."
},

{
    "location": "Laplace.html#CoherentStructures.diff_op",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.diff_op",
    "category": "function",
    "text": "diff_op(data, sp_method, kernel = gaussian_kernel; α=1.0, metric=Euclidean\"()\")\n\nReturn a diffusion/Markov matrix P.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nsp_method: employed sparsification method (neighborhood or mutualKNN);\nkernel: diffusion kernel, e.g., x -> exp(-x*x/4σ);\nα: exponent in diffusion-map normalization;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.sparse_diff_op_family",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_diff_op_family",
    "category": "function",
    "text": "sparse_diff_op_family(data, sp_method, kernel=gaussian_kernel, dim=2; op_reduce, α, metric)\n\nReturn a list of sparse diffusion/Markov matrices P.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nsp_method: sparsification method;\nkernel: diffusion kernel, e.g., x -> exp(-x*x/4σ);\ndim: the columns are interpreted as concatenations of dim- dimensional points, to which metric is applied individually;\nop_reduce: time-reduction of diffusion operators, e.g. mean or P -> prod(LinearMaps.LinearMap,Iterators.reverse(P)) (default)\nα: exponent in diffusion-map normalization;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.sparse_diff_op",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_diff_op",
    "category": "function",
    "text": "sparse_diff_op(data, sp_method, kernel; α=1.0, metric=Euclidean()) -> SparseMatrixCSC\n\nReturn a sparse diffusion/Markov matrix P.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nsp_method: sparsification method;\nkernel: diffusion kernel, e.g., x -> exp(-x*x) (default);\nα: exponent in diffusion-map normalization;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.sparseaffinitykernel",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparseaffinitykernel",
    "category": "function",
    "text": "sparseaffinitykernel(data, sp_method, kernel, metric=Euclidean()) -> SparseMatrixCSC\n\nReturn a sparse matrix W where w_ij = k(x_i x_j). The x_i are taken from the columns of data. Entries are only calculated for pairs determined by the sparsification method sp_method. Default metric is Euclidean().\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.α_normalize!",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.α_normalize!",
    "category": "function",
    "text": "α_normalize!(A, α = 1.0)\n\nNormalize rows and columns of A in-place with the respective row-sum to the α-th power; i.e., return a_ij=a_ijq_i^alphaq_j^alpha, where q_k = sum_ell a_kell. Default for α is 1.0.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.wLap_normalize!",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.wLap_normalize!",
    "category": "function",
    "text": "wLap_normalize!(A)\n\nNormalize rows of A in-place with the respective row-sum; i.e., return a_ij=a_ijq_i.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#Diffusion-maps-type-graph-Laplacian-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Diffusion-maps type graph Laplacian methods",
    "category": "section",
    "text": "diff_op\nsparse_diff_op_family\nsparse_diff_op\nsparseaffinitykernel\nα_normalize!\nwLap_normalize!"
},

{
    "location": "Laplace.html#CoherentStructures.sparse_adjacency",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_adjacency",
    "category": "function",
    "text": "sparse_adjacency(data, ε[, dim]; metric) -> SparseMatrixCSC\n\nReturn a sparse adjacency matrix A with integer entries 0 or 1. If the third argument dim is passed, then data is interpreted as concatenated points of length dim, to which metric is applied individually. Otherwise, metric is applied to the whole columns of data.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nε: distance threshold;\ndim: the columns of data are interpreted as concatenations of dim- dimensional points, to which metric is applied individually;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.sparse_adjacency_list",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_adjacency_list",
    "category": "function",
    "text": "sparse_adjacency_list(data, ε; metric=Euclidean()) -> idxs::Vector{Vector}\n\nReturn two lists of indices of data points that are adjacent.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nε: distance threshold;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#Adjancency-matrix-based-graph-Laplacian-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Adjancency-matrix-based graph Laplacian methods",
    "category": "section",
    "text": "sparse_adjacency\nsparse_adjacency_list"
},

{
    "location": "Laplace.html#CoherentStructures.diffusion_coordinates",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.diffusion_coordinates",
    "category": "function",
    "text": "diffusion_coordinates(P,n_coords) -> (Σ::Vector, Ψ::Matrix)\n\nCompute the (time-coupled) diffusion coordinates Ψ and the coordinate weights Σ for a linear map P. n_coords determines the number of diffusion coordinates to be computed.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#CoherentStructures.diffusion_distance",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.diffusion_distance",
    "category": "function",
    "text": "diffusion_distance(diff_coord) -> SymmetricMatrix\n\nReturns the distance matrix of pairs of points whose diffusion distances correspond to the diffusion coordinates given by diff_coord.\n\n\n\n\n\n"
},

{
    "location": "Laplace.html#Diffusion-coordinate-like-functions-1",
    "page": "Graph Laplacian-based methods",
    "title": "Diffusion-coordinate-like functions",
    "category": "section",
    "text": "diffusion_coordinates\ndiffusion_distance"
},

]}
