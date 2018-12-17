var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#CoherentStructures.jl-1",
    "page": "Home",
    "title": "CoherentStructures.jl",
    "category": "section",
    "text": "Tools for computing Lagrangian Coherent Structures in Julia"
},

{
    "location": "#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "CoherentStructures.jl is a toolbox for computing Lagrangian Coherent Structures in aperiodic flows in Julia. It has been developed in Oliver Junge\'s research group at TUM, Germany, by (in alphabetical order)Alvaro de Diego (@adediego)\nDaniel Karrasch (@dkarrasch)\nNathanael Schilling (@natschil)Contributions from colleagues in the field are most welcome via raising issues or, even better, via pull requests."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "First install the JuAFEM.jl package.Then, run the following in the Julia REPL:]add https://github.com/CoherentStructures/CoherentStructures.jl.git"
},

{
    "location": "generated/rot_double_gyre/#",
    "page": "Rotating double gyre",
    "title": "Rotating double gyre",
    "category": "page",
    "text": ""
},

{
    "location": "generated/rot_double_gyre/#Rotating-Double-Gyre-1",
    "page": "Rotating double gyre",
    "title": "Rotating Double Gyre",
    "category": "section",
    "text": "The (computable) notebook for this example can be found here."
},

{
    "location": "generated/rot_double_gyre/#Description-1",
    "page": "Rotating double gyre",
    "title": "Description",
    "category": "section",
    "text": "The rotating double gyre model was introduced by Mosovsky & Meiss. It can be derived from the stream functionpsi(xyt)=(1s(t))psi_P +s(t)psi_F  psi_P (x y) = sin(2pi x) sin(pi y)  psi_F (x y) = sin(pi x) sin(2pi y)where s is (usually taken to be) a cubic interpolating function satisfying s(0) = 0 and s(1) = 1. It therefore interpolates two double-gyre-type velocity fields, from horizontally to vertically arranged counter-rotating gyres. The corresponding velocity field is provided by the package and callable as rot_double_gyre.(Image: )"
},

{
    "location": "generated/rot_double_gyre/#FEM-Based-Methods-1",
    "page": "Rotating double gyre",
    "title": "FEM-Based Methods",
    "category": "section",
    "text": "The following code demonstrates how to use these methods.using CoherentStructures, Arpack\nLL = [0.0, 0.0]; UR = [1.0, 1.0];\nctx, _ = regularTriangularGrid((50, 50), LL, UR)\n\nA = x -> mean_diff_tensor(rot_double_gyre, x, [0.0, 1.0], 1.e-10, tolerance= 1.e-4)\nK = assembleStiffnessMatrix(ctx, A)\nM = assembleMassMatrix(ctx)\nλ, v = eigs(-K, M, which=:SM);This velocity field is given by the rot_double_gyre function. The third argument to mean_diff_tensor is a vector of time instances at which we evaluate (and subsequently average) the pullback diffusion tensors. The fourth parameter is the step size δ used for the finite-difference scheme, tolerance is passed to the ODE solver from DifferentialEquations.jl. In the above, A(x) approximates the mean diffusion tensor given byA(x) = sum_t in mathcal T(DPhi^t(x))^-1 (DPhi^t x)^-TThe eigenfunctions saved in v approximate those of Delta^dynimport Plots\nres = [plot_u(ctx, v[:,i], 100, 100, colorbar=:none, clim=(-3,3)) for i in 1:6];\nfig = Plots.plot(res..., margin=-10Plots.px)(Image: )Looking at the spectrum, there appears a gap after the third eigenvalue.spectrum_fig = Plots.scatter(1:6, real.(λ))(Image: )We can use the Clustering.jl package to compute coherent structures from the first two nontrivial eigenfunctions:using Clustering\n\nctx2, _ = regularTriangularGrid((200, 200))\nv_upsampled = sample_to(v, ctx, ctx2)\n\nnumclusters=2\nres = kmeans(permutedims(v_upsampled[:,2:numclusters+1]), numclusters + 1)\nu = kmeansresult2LCS(res)\nres = Plots.plot([plot_u(ctx2, u[:,i], 200, 200, color=:viridis, colorbar=:none) for i in [1,2,3]]...)(Image: )"
},

{
    "location": "generated/rot_double_gyre/#Geodesic-vortices-1",
    "page": "Rotating double gyre",
    "title": "Geodesic vortices",
    "category": "section",
    "text": "Here, we demonstrate how to calculate black-hole vortices, see Geodesic elliptic material vortices for references and details.using Distributed\nnprocs() == 1 && addprocs()\n\n@everywhere begin\n    using CoherentStructures, OrdinaryDiffEq, StaticArrays\n    import AxisArrays\n    const AA = AxisArrays\n    const q = 51\n    const tspan = range(0., stop=1., length=q)\n    ny = 101\n    nx = 101\n    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0\n    xspan = range(xmin, stop=xmax, length=nx)\n    yspan = range(ymin, stop=ymax, length=ny)\n    P = AA.AxisArray(SVector{2}.(xspan, yspan\'), xspan, yspan)\n    const δ = 1.e-6\n    mCG_tensor = u -> av_weighted_CG_tensor(rot_double_gyre, u, tspan, δ;\n            tolerance=1e-6, solver=Tsit5())\nend\n\nC̅ = pmap(mCG_tensor, P; batch_size=ny)\np = LCSParameters(3*max(step(xspan), step(yspan)), 0.5, true, 60, 0.7, 1.5, 1e-4)\nvortices, singularities = ellipticLCS(C̅, p; outermost=true)The results are then visualized as follows.using Plots\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)\nfig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));\n            aspect_ratio=1, color=:viridis, leg=true,\n            title=\"DBS field and transport barriers\")\nscatter!(getcoords(singularities), color=:red)\nfor vortex in vortices\n    plot!(vortex.curve, color=:yellow, w=3, label=\"T = $(round(vortex.p, digits=2))\")\n    scatter!(vortex.core, color=:yellow)\nend(Image: )This page was generated using Literate.jl."
},

{
    "location": "generated/standard_map/#",
    "page": "Standard map",
    "title": "Standard map",
    "category": "page",
    "text": ""
},

{
    "location": "generated/standard_map/#The-standard-map-1",
    "page": "Standard map",
    "title": "The standard map",
    "category": "section",
    "text": "The (computable) notebook for this example can be found here.The standard mapf(xy) = (x+y+asin(x)y+asin(x))is an area-preserving map on the 2-torus 02pi^2 resulting from a symplectic time-discretization of the planar pendulum.  For a = 0971635, its phase space shows the characteristic mixture of regular (periodic or quasi-periodic) and chaotic motion.  Here, we repeat the experiment in Froyland & Junge (2015) and compute coherent structures.We first visualize the phase space by plotting 500 iterates of 50 random seed points.using Random\n\na = 0.971635\nf(a,x) = (mod(x[1] + x[2] + a*sin(x[1]), 2π),\n          mod(x[2] + a*sin(x[1]), 2π))\n\nX = []\nfor i in 1:50\n    Random.seed!(i)\n    x = 2π*rand(2)\n    for i in 1:500\n        x = f(a,x)\n        push!(X,x)\n    end\nend\n\nusing Plots\ngr(aspect_ratio=1, legend=:none)\nfig = scatter([x[1] for x in X], [x[2] for x in X], markersize=1)(Image: )Approximating the Dynamic Laplacian by FEM methods is straightforward:using Arpack, CoherentStructures, Tensors\n\nDf(a,x) = Tensor{2,2}((1.0+a*cos(x[1]), a*cos(x[1]), 1.0, 1.0))\n\nn, ll, ur = 100, [0.0,0.0], [2π,2π]               # grid size, domain corners\nctx, _ = regularTriangularGrid((n,n), ll, ur)\npred(x,y) = peuclidean(x[1], y[1], 2π) < 1e-9 &&\n            peuclidean(x[2], y[2], 2π) < 1e-9\nbd = boundaryData(ctx, pred)                      # periodic boundary\n\nI = one(Tensor{2,2})                              # identity matrix\nDf2(x) = Df(a,f(a,x))⋅Df(a,x)                     # consider 2. iterate\ncg(x) = 0.5*(I + dott(inv(Df2(x))))               # avg. inv. Cauchy-Green tensor\n\nK = assembleStiffnessMatrix(ctx, cg, bdata=bd)\nM = assembleMassMatrix(ctx, bdata=bd)\nλ, v = eigs(K, M, which=:SM)\n\nusing Printf\ntitle = [ @sprintf(\"\\\\lambda = %.3f\",λ[i]) for i = 1:4 ]\np = [ plot_u(ctx, v[:,i], bdata=bd, title=title[i],\n             clim=(-0.25,0.25), cb=false) for i in 1:4 ]\nfig = plot(p...)(Image: )This page was generated using Literate.jl."
},

{
    "location": "generated/bickley/#",
    "page": "Bickley jet",
    "title": "Bickley jet",
    "category": "page",
    "text": ""
},

{
    "location": "generated/bickley/#Bickley-Jet-1",
    "page": "Bickley jet",
    "title": "Bickley Jet",
    "category": "section",
    "text": "The (computable) notebook for this example can be found here.The Bickley jet flow is a kinematic idealized model of a meandering zonal jet flanked above and below by counterrotating vortices. It was introduced by Rypina et al.; cf. also del‐Castillo‐Negrete and Morrison.The Bickley jet is described by a time-dependent velocity field arising from a stream-function. The corresponding velocity field is provided by the package and callable as bickleyJet.Instead of using the bickleyJet function to get this velocity field, we could also use the @velo_from_stream macro:using CoherentStructures\nbickley = @velo_from_stream stream begin\n    stream = psi₀ + psi₁\n    psi₀   = - U₀ * L₀ * tanh(y / L₀)\n    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term\n\n    re_sum_term =  Σ₁ + Σ₂ + Σ₃\n\n    Σ₁  =  ε₁ * cos(k₁*(x - c₁*t))\n    Σ₂  =  ε₂ * cos(k₂*(x - c₂*t))\n    Σ₃  =  ε₃ * cos(k₃*(x - c₃*t))\n\n    k₁ = 2/r₀      ; k₂ = 4/r₀    ; k₃ = 6/r₀\n\n    ε₁ = 0.0075    ; ε₂ = 0.15    ; ε₃ = 0.3\n    c₂ = 0.205U₀   ; c₃ = 0.461U₀ ; c₁ = c₃ + (√5-1)*(c₂-c₃)\n\n    U₀ = 62.66e-6  ; L₀ = 1770e-3 ; r₀ = 6371e-3\nendNow, bickley is a callable function with the standard OrdinaryDiffEq signature (u, p, t) with state u, (unused) parameter p and time t."
},

{
    "location": "generated/bickley/#Geodesic-vortices-1",
    "page": "Bickley jet",
    "title": "Geodesic vortices",
    "category": "section",
    "text": "Here we briefly demonstrate how to find material barriers to diffusive transport; see Geodesic elliptic material vortices for references and details.using Distributed\nnprocs() == 1 && addprocs()\n\n@everywhere begin\n    using CoherentStructures, OrdinaryDiffEq, Tensors, StaticArrays\n    import AxisArrays\n    const AA = AxisArrays\n    q = 81\n    const tspan = range(0., stop=3456000., length=q)\n    ny = 61\n    nx = (22ny) ÷ 6\n    xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0\n    xspan = range(xmin, stop=xmax, length=nx)\n    yspan = range(ymin, stop=ymax, length=ny)\n    P = AA.AxisArray(SVector{2}.(xspan, yspan\'), xspan, yspan)\n    const δ = 1.e-6\n    const DiffTensor = SymmetricTensor{2,2}([2., 0., 1/2])\n    mCG_tensor = u -> av_weighted_CG_tensor(bickleyJet, u, tspan, δ;\n              D=DiffTensor, tolerance=1e-6, solver=Tsit5())\nend\n\nC̅ = pmap(mCG_tensor, P; batch_size=ny)\np = LCSParameters(3*max(step(xspan), step(yspan)), 2.0,true, 60, 0.7, 1.5, 1e-4)\nvortices, singularities = ellipticLCS(C̅, p)The result is visualized as follows:import Plots\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)\nfig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));\n                    aspect_ratio=1, color=:viridis, leg=true,\n                    xlims=(0, 6.371π), ylims=(-3, 3),\n                    title=\"DBS field and transport barriers\")\nPlots.scatter!(fig,getcoords(singularities), color=:red)\nfor vortex in vortices\n    Plots.plot!(fig,vortex.curve, color=:yellow, w=3, label=\"T = $(round(vortex.p, digits=2))\")\n    Plots.scatter!(fig,vortex.core, color=:yellow)\nend(Image: )"
},

{
    "location": "generated/bickley/#FEM-based-Methods-1",
    "page": "Bickley jet",
    "title": "FEM-based Methods",
    "category": "section",
    "text": "Assume we have setup the bickley function using the @velo_from_stream macro as described above. We are working on a periodic domain in one direction:LL = [0.0, -3.0]; UR = [6.371π, 3.0]\nctx, _ = regularP2TriangularGrid((50, 15), LL, UR, quadrature_order=2)\npredicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 6.371π) < 1e-10\nbdata = CoherentStructures.boundaryData(ctx, predicate, []);Using a FEM-based method to compute coherent structures:using Arpack\ncgfun = (x -> mean_diff_tensor(bickley, x, range(0.0, stop=40*3600*24, length=81),\n     1.e-8; tolerance=1.e-5))\n\nK = assembleStiffnessMatrix(ctx, cgfun, bdata=bdata)\nM = assembleMassMatrix(ctx, bdata=bdata)\nλ, v = eigs(K, M, which=:SM, nev= 10)\n\nimport Plots\nplot_real_spectrum(λ)K-means clustering yields the coherent vortices.using Clustering\nctx2, _ = regularTriangularGrid((200, 60), LL, UR)\nv_upsampled = sample_to(v, ctx, ctx2, bdata=bdata)\n\nfunction iterated_kmeans(numiterations, args...)\n    best = kmeans(args...)\n    for i in 1:(numiterations - 1)\n        cur = kmeans(args...)\n        if cur.totalcost < best.totalcost\n            best = cur\n        end\n    end\n    return best\nend\n\nn_partition = 8\nres = iterated_kmeans(20, permutedims(v_upsampled[:,2:n_partition]), n_partition)\nu = kmeansresult2LCS(res)\nu_combined = sum([u[:,i]*i for i in 1:n_partition])\nfig = plot_u(ctx2, u_combined, 400, 400;\n    color=:rainbow, colorbar=:none, title=\"$n_partition-partition of Bickley jet\")(Image: )This page was generated using Literate.jl."
},

{
    "location": "generated/ocean_flow/#",
    "page": "Geostrophic ocean flow",
    "title": "Geostrophic ocean flow",
    "category": "page",
    "text": ""
},

{
    "location": "generated/ocean_flow/#Geostrophic-Ocean-Flow-1",
    "page": "Geostrophic ocean flow",
    "title": "Geostrophic Ocean Flow",
    "category": "section",
    "text": "The (computable) notebook for this example can be found here.For a more realistic application, we consider an unsteady ocean surface velocity data set obtained from satellite altimetry measurements produced by SSALTO/DUACS and distributed by AVISO. The particular space-time window has been used several times in the literature.Below is a video showing advection of the initial 90-day DBS field for 90 days.<video controls=\"\" height=\"100%\" width=\"100%\">\n <source src=\"https://raw.githubusercontent.com/natschil/misc/master/videos/ocean_flow.mp4\" type=\"video/mp4\" />\nYour browser does not support the video tag.\n</video>"
},

{
    "location": "generated/ocean_flow/#Geodesic-vortices-1",
    "page": "Geostrophic ocean flow",
    "title": "Geodesic vortices",
    "category": "section",
    "text": "Here, we demonstrate how to detect material barriers to diffusive transport.using Distributed\nnprocs() == 1 && addprocs()\n\n@everywhere using CoherentStructures, OrdinaryDiffEq, StaticArraysNext, we load and interpolate the velocity data sets.using JLD2\nJLD2.@load(\"Ocean_geostrophic_velocity.jld2\")\nconst VI = interpolateVF(Lon, Lat, Time, UT, VT)Since we want to use parallel computing, we set up the integration LCSParameters on all workers, i.e., @everywhere.begin\n    import AxisArrays\n    const AA = AxisArrays\n    q = 91\n    t_initial = minimum(Time)\n    t_final = t_initial + 90\n    const tspan = range(t_initial, stop=t_final, length=q)\n    xmin, xmax, ymin, ymax = -4.0, 7.5, -37.0, -28.0\n    nx = 300\n    ny = floor(Int, (ymax - ymin) / (xmax - xmin) * nx)\n    xspan = range(xmin, stop=xmax, length=nx)\n    yspan = range(ymin, stop=ymax, length=ny)\n    P = AA.AxisArray(SVector{2}.(xspan, yspan\'), xspan, yspan)\n    const δ = 1.e-5\n    mCG_tensor = u -> av_weighted_CG_tensor(interp_rhs, u, tspan, δ;\n        p=VI, tolerance=1e-6, solver=Tsit5())\nendNow, compute the averaged weighted Cauchy-Green tensor field and extract elliptic LCSs.C̅ = pmap(mCG_tensor, P; batch_size=ny)\np = LCSParameters(5*max(step(xspan), step(yspan)), 2.5, true, 60, 0.5, 2.0, 1e-4)\nvortices, singularities = ellipticLCS(C̅, p)Finally, the result is visualized as follows.using Plots\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)\nfig = Plots.heatmap(xspan, yspan, permutedims(log10.(traceT));\n            aspect_ratio=1, color=:viridis, leg=true,\n            title=\"DBS field and transport barriers\")\nscatter!(getcoords(singularities), color=:red)\nfor vortex in vortices\n    plot!(vortex.curve, color=:yellow, w=3, label=\"T = $(round(vortex.p, digits=2))\")\n    scatter!(vortex.core, color=:yellow)\nend(Image: )"
},

{
    "location": "generated/ocean_flow/#FEM-based-methods-1",
    "page": "Geostrophic ocean flow",
    "title": "FEM-based methods",
    "category": "section",
    "text": "Here we showcase how the adaptive TO method can be used to calculate coherent sets.First we setup the problem.using CoherentStructures\nimport JLD2, OrdinaryDiffEq, Plots\n\n#Import and interpolate ocean dataset\n#The @load macro initializes Lon,Lat,Time,UT,VT\n\nJLD2.@load(\"Ocean_geostrophic_velocity.jld2\")\n\nVI = interpolateVF(Lon, Lat, Time, UT, VT)\n\n#Define a flow function from it\nt_initial = minimum(Time)\nt_final = t_initial + 90\ntimes = [t_initial, t_final]\nflow_map = u0 -> flow(interp_rhs, u0, times;\n    p=VI, tolerance=1e-5, solver=OrdinaryDiffEq.BS5())[end]Next we set up the domain. We want to use zero Dirichlet boundary conditions here.LL = [-4.0, -34.0]\nUR = [6.0, -28.0]\nctx, _  = regularTriangularGrid((150, 90), LL, UR)\nbdata = getHomDBCS(ctx, \"all\");For the TO method, we seek generalized eigenpairs involving the bilinear forma_h(uv) = frac12 left(a_0(uv) + a_1(I_h u I_h v) right)Here, a_0 is the weak form of the Laplacian on the initial domain, and a_1 is the weak form of the Laplacian on the final domain. The operator I_h is an interpolation operator onto the space of test functions on the final domain.For the adaptive TO method, we use pointwise nodal interpolation (i.e. collocation) and the mesh on the final domain is obtained by doing a Delaunay triangulation on the images of the nodal points of the initial domain. This results in the representation matrix of I_h being the identity, so in matrix form we get:S = 05(S_0 + S_1)where S_0 is the stiffness matrix for the triangulation at initial time, and S_1 is the stiffness matrix for the triangulation at final time.M = assembleMassMatrix(ctx, bdata=bdata)\nS0 = assembleStiffnessMatrix(ctx)\nS1 = adaptiveTOCollocationStiffnessMatrix(ctx, flow_map)Averaging matrices and applying boundary conditions yieldsS = applyBCS(ctx, 0.5(S0 + S1), bdata);We can now solve the eigenproblem.using Arpack\n\nλ, v = eigs(S, M, which=:SM, nev=6);We upsample the eigenfunctions and then cluster.using Clustering\n\nctx2, _ = regularTriangularGrid((200, 120), LL, UR)\nv_upsampled = sample_to(v, ctx, ctx2, bdata=bdata)\n\nfunction iterated_kmeans(numiterations, args...)\n    best = kmeans(args...)\n    for i in 1:(numiterations - 1)\n        cur = kmeans(args...)\n        if cur.totalcost < best.totalcost\n            best = cur\n        end\n    end\n    return best\nend\n\nn_partition = 4\nres = iterated_kmeans(20, permutedims(v_upsampled[:,1:(n_partition-1)]), n_partition)\nu = kmeansresult2LCS(res)\nu_combined = sum([u[:,i] * i for i in 1:n_partition])\nfig = plot_u(ctx2, u_combined, 200, 200;\n    color=:viridis, colorbar=:none, title=\"$n_partition-partition of Ocean Flow\")(Image: )This page was generated using Literate.jl."
},

{
    "location": "basics/#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "basics/#Basics-1",
    "page": "Basics",
    "title": "Basics",
    "category": "section",
    "text": ""
},

{
    "location": "basics/#Dynamical-Systems-Utilities-1",
    "page": "Basics",
    "title": "Dynamical Systems Utilities",
    "category": "section",
    "text": "CurrentModule = CoherentStructures"
},

{
    "location": "basics/#CoherentStructures.@define_stream",
    "page": "Basics",
    "title": "CoherentStructures.@define_stream",
    "category": "macro",
    "text": "@define_stream(name::Symbol, code::Expr)\n\nDefine a scalar stream function on R^2. The defining code can be a series of definitions in an enclosing begin ... end-block and is treated as a series of symbolic substitutions. Starting from the symbol name, substitutions are performed until the resulting expression only depends on x, y and t.\n\nThe symbol name is not brought into the namespace. To access the resulting vector field and variational equation  use @velo_from_stream name and @var_velo_from_stream name\n\nThis is a convenience macro for the case where you want to use @velo_from_stream and @var_velo_from_stream without typing the code twice. If you only use one, you might as well use @velo_from_stream name code or @var_velo_from_stream directly.\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.@velo_from_stream",
    "page": "Basics",
    "title": "CoherentStructures.@velo_from_stream",
    "category": "macro",
    "text": "@velo_from_stream(name::Symbol, [code::Expr])\n\nGet the velocity field corresponding to a stream function on R^2. The defining code can be a series of definitions (in an enclosing begin ... end-block and is treated as a series of symbolic substitutions. Starting from the symbol name, substitutions are performed until the resulting expression only depends on x, y and t.\n\nThe macro returns an anonymous function with signature (u,p,t) that returns an SVector{2} corresponding to the vector field at position u at time t. The parameter slot is not used and can be filled with nothing when calling.\n\nThe macro can be called without the code if the stream name has been defined beforehand via @define_stream.\n\nnote: Sign convention\nWe follow the \"oceanographic\" sign convention, whereby the velocity v is derived from the stream function psi by v = (-partial_ypsi partial_xpsi)\n\nExamples\n\njulia> using CoherentStructures\n\njulia> f = @velo_from_stream Ψ_ellipse begin\n               Ψ_ellipse = a*x^2 + b*y^2\n               a = t\n               b = 3\n           end\n(#3) #1 (generic function with 1 method)\n\njulia> f([1.0,1.0], nothing, 1.0)\n2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:\n -6.0\n  2.0\n\njulia> using CoherentStructures\n\njulia> @define_stream Ψ_circular begin\n           Ψ_circular = f(x) + g(y)\n           # naming of function variables\n           # does not matter:\n           f(a) = a^2\n           g(y) = y^2\n       end\n\njulia> f2 = @velo_from_stream Ψ_circular\n(#5) #1 (generic function with 1 method)\n\njulia> f2([1.0,1.0], nothing, 0.0)\n2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:\n -2.0\n  2.0\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.@var_velo_from_stream",
    "page": "Basics",
    "title": "CoherentStructures.@var_velo_from_stream",
    "category": "macro",
    "text": "@var_velo_from_stream(name::Symbol, [code::Expr])\n\nGet the (state and tangent space) velocity field corresponding to a stream function on R^2. The defining code can be a series of definitions (in an enclosing begin ... end-block and is treated as a series of symbolic substitutions. Starting from the symbol name, substitutions are performed until the resulting expression only depends on x, y and t.\n\nThe macro returns an anonymous function with signature (U,p,t) that returns an SMatrix{2,3}: in the first column, one has the usual velocity, in the second to third column, one has the linearized velocity, both at position u = U[:,1] at time t. The parameter slot is not used and can be filled with nothing when calling.\n\nThe macro can be called without the code if the stream name has been defined beforehand via @define_stream.\n\nnote: Sign convention\nWe follow the \"oceanographic\" sign convention, whereby the velocity v is derived from the stream function psi by v = (-partial_ypsi partial_xpsi)\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.@vorticity_from_stream",
    "page": "Basics",
    "title": "CoherentStructures.@vorticity_from_stream",
    "category": "macro",
    "text": "@vorticity_from_stream(name::Symbol, [code::Expr])\n\nGet the vorticity field as a function of (x, y, t) corresponding to a stream function on R^2.\n\nnote: Sign convention\nThe vorticity omega of the velocity field v = (v_x v_y) is defined as derived from the stream function psi by omega = partial_x v_x - partial_y v_y) = trace(nabla^2psi), i.e., the trace of the Hessian of the stream function.\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.interpolateVF",
    "page": "Basics",
    "title": "CoherentStructures.interpolateVF",
    "category": "function",
    "text": "interpolateVF(xspan, yspan, tspan, u, v, interpolation_type=ITP.BSpline(ITP.Cubic(ITP.Free())))) -> VI\n\nxspan, yspan and tspan span the space-time domain on which the velocity-components u and v are given. u corresponds to the x- or eastward component, v corresponds to the y- or northward component. For interpolation, the Interpolations.jl package is used; see their documentation for how to declare other interpolation types.\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.interp_rhs",
    "page": "Basics",
    "title": "CoherentStructures.interp_rhs",
    "category": "function",
    "text": "interp_rhs(u, p, t) -> SVector{2}\n\nDefines a 2D vector field that is readily usable for trajectory integration from vector field interpolants of the x- and y-direction, resp. It assumes that the interpolants are provided as a 2-tuple (UI, VI) via the parameter p. Here, UI and VI are the interpolants for the x- and y-components of the velocity field.\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.interp_rhs!",
    "page": "Basics",
    "title": "CoherentStructures.interp_rhs!",
    "category": "function",
    "text": "interp_rhs!(du, u, p, t) -> Vector\n\nDefines a mutating/inplace 2D vector field that is readily usable for trajectory integration from vector field interpolants of the x- and y-direction, resp. It assumes that the interpolants are provided as a 2-tuple (UI, VI) via the parameter p. Here, UI and VI are the interpolants for the x- and y-components of the velocity field.\n\n\n\n\n\n"
},

{
    "location": "basics/#Definition-of-vector-fields-1",
    "page": "Basics",
    "title": "Definition of vector fields",
    "category": "section",
    "text": "CoherentStructures.jl is set up for handling two- and three-dimensional dynamical systems only. For such low-dimensional flows it is advantageous (for top performance) to obey the following syntax:function vectorfield2d(u, p, t)\n    du1 = ... # equation for $\\dot{x}$\n    du2 = ... # equation for $\\dot{y}$\n    return StaticArrays.SVector{2}(du1, du2)\nendand correspondingly for three-dimensional ODEs:function vectorfield3d(u, p, t)\n    du1 = ... # equation for $\\dot{x}$\n    du2 = ... # equation for $\\dot{y}$\n    du3 = ... # equation for $\\dot{z}$\n    return StaticArrays.SVector{3}(du1, du2, du3)\nendFurthermore, there are convenience macros to define two-dimensional velocity and vorticity fields from stream functions.@define_stream\n@velo_from_stream\n@var_velo_from_stream\n@vorticity_from_streamIn fact, two of the predefined velocity fields, the rotating double gyre rot_double_gyre, and the Bickley jet flow bickleyJet, are generated from these macros.Another typical use case is when velocities are given as a data set. In this case, one first interpolates the velocity components with interpolateVF to obtain callable interpolation functions, say, UI and VI. The corresponding vector field is then interp_rhs, into which the velocity interpolants enter via the parameter argument p; see below for examples.interpolateVF\ninterp_rhs\ninterp_rhs!"
},

{
    "location": "basics/#CoherentStructures.flow",
    "page": "Basics",
    "title": "CoherentStructures.flow",
    "category": "function",
    "text": "flow(odefun,  u0, tspan; tolerance, p, solver) -> Vector{Vector}\n\nSolve the ODE with right hand side given by odefun and initial value u0. p is a parameter passed to odefun. tolerance is passed as both relative and absolute tolerance to the solver, which is determined by solver.\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.parallel_flow",
    "page": "Basics",
    "title": "CoherentStructures.parallel_flow",
    "category": "function",
    "text": "parallel_flow(flow_fun,P) -> Array\n\nApply the flow_fun to each element in P in parallel, if possible. Returns a 2D array with dimensions ((space dim x no. of time instances) x no. of trajectories), in which each column corresponds to a concatenated trajectory, i.e., represented in delay coordinates.\n\n\n\n\n\n"
},

{
    "location": "basics/#Flow-maps-1",
    "page": "Basics",
    "title": "Flow maps",
    "category": "section",
    "text": "flowparallel_flow"
},

{
    "location": "basics/#CoherentStructures.linearized_flow",
    "page": "Basics",
    "title": "CoherentStructures.linearized_flow",
    "category": "function",
    "text": "linearized_flow(odefun, x, tspan,δ; ...) -> Vector{Tensor{2,2}}\n\nCalculate derivative of flow map by finite differences if δ != 0. If δ==0, attempts to solve variational equation (odefun is assumed to be the rhs of variational equation in this case). Return time-resolved linearized flow maps.\n\n\n\n\n\n"
},

{
    "location": "basics/#Linearized-flow-map-1",
    "page": "Basics",
    "title": "Linearized flow map",
    "category": "section",
    "text": "linearized_flow"
},

{
    "location": "basics/#CoherentStructures.CG_tensor",
    "page": "Basics",
    "title": "CoherentStructures.CG_tensor",
    "category": "function",
    "text": "CG_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor\n\nReturns the classic right Cauchy–Green strain tensor. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nkwargs...: are passed to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.mean_diff_tensor",
    "page": "Basics",
    "title": "CoherentStructures.mean_diff_tensor",
    "category": "function",
    "text": "mean_diff_tensor(odefun, u, tspan, δ; kwargs...) -> SymmetricTensor\n\nReturns the averaged diffusion tensor at a point along a set of times. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nkwargs...: are passed to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.av_weighted_CG_tensor",
    "page": "Basics",
    "title": "CoherentStructures.av_weighted_CG_tensor",
    "category": "function",
    "text": "av_weighted_CG_tensor(odefun, u, tspan, δ; D, kwargs...) -> SymmetricTensor\n\nReturns the transport tensor of a trajectory, aka  time-averaged, di ffusivity-structure-weighted version of the classic right Cauchy–Green strain tensor. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nD: (constant) diffusion tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.pullback_tensors",
    "page": "Basics",
    "title": "CoherentStructures.pullback_tensors",
    "category": "function",
    "text": "pullback_tensors(odefun, u, tspan, δ; D, kwargs...) -> Tuple(Vector{SymmetricTensor},Vector{SymmetricTensor})\n\nReturns the time-resolved pullback tensors of both the diffusion and the metric tensor along a trajectory. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nD: (constant) diffusion tensor, metric tensor is computed via inversion; defaults to eye(2)\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.pullback_metric_tensor",
    "page": "Basics",
    "title": "CoherentStructures.pullback_metric_tensor",
    "category": "function",
    "text": "pullback_metric_tensor(odefun, u, tspan, δ; G, kwargs...) -> Vector{SymmetricTensor}\n\nReturns the time-resolved pullback tensors of the metric tensor along a trajectory, aka right Cauchy-Green strain tensor. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nG: (constant) metric tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.pullback_diffusion_tensor",
    "page": "Basics",
    "title": "CoherentStructures.pullback_diffusion_tensor",
    "category": "function",
    "text": "pullback_diffusion_tensor(odefun, u, tspan, δ; D, kwargs...) -> Vector{SymmetricTensor}\n\nReturns the time-resolved pullback tensors of the diffusion tensor along a trajectory. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nD: (constant) diffusion tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.pullback_SDE_diffusion_tensor",
    "page": "Basics",
    "title": "CoherentStructures.pullback_SDE_diffusion_tensor",
    "category": "function",
    "text": "pullback_SDE_diffusion_tensor(odefun, u, tspan, δ; B, kwargs...) -> Vector{SymmetricTensor}\n\nReturns the time-resolved pullback tensors of the diffusion tensor in SDEs. Derivatives are computed with finite differences.\n\nodefun: RHS of the ODE\nu: initial value of the ODE\ntspan: set of time instances at which to save the trajectory\nδ: stencil width for the finite differences\nB: (constant) SDE tensor\nkwargs... are passed through to linearized_flow\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.tensor_invariants",
    "page": "Basics",
    "title": "CoherentStructures.tensor_invariants",
    "category": "function",
    "text": "tensor_invariants(T) -> λ₁, λ₂, ξ₁, ξ₂, traceT, detT\n\nReturns pointwise invariants of the 2D symmetric tensor field T, i.e., smallest and largest eigenvalues, corresponding eigenvectors, trace and determinant.\n\nExample\n\nT = [SymmetricTensor{2,2}(rand(3)) for i in 1:10, j in 1:20]\nλ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(T)\n\nAll output variables have the same array arrangement as T; e.g., λ₁ is a 10x20 array with scalar entries.\n\n\n\n\n\n"
},

{
    "location": "basics/#Cauchy-Green-and-other-pullback-tensors-1",
    "page": "Basics",
    "title": "Cauchy-Green and other pullback tensors",
    "category": "section",
    "text": "CG_tensormean_diff_tensorav_weighted_CG_tensorpullback_tensors\npullback_metric_tensor\npullback_diffusion_tensor\npullback_SDE_diffusion_tensorA second-order symmetric two-dimensional tensor (field) may be diagonalized (pointwise), ie., an eigendecomposition is computed, by the following function.tensor_invariants"
},

{
    "location": "basics/#CoherentStructures.PEuclidean",
    "page": "Basics",
    "title": "CoherentStructures.PEuclidean",
    "category": "type",
    "text": "PEuclidean(L)\n\nCreate a Euclidean metric on a rectangular periodic domain. Periods per dimension are contained in the vector L. For dimensions without periodicity put Inf in the respective component.\n\nExample\n\njulia> using Distances\n\njulia> x, y, L = [0.0, 0.0], [0.7, 0.0], [0.5, Inf]\n([0.0, 0.0], [0.7, 0.0], [0.5, Inf])\n\njulia> evaluate(PEuclidean(L), x, y)\n0.19999999999999996\n\n\n\n\n\n"
},

{
    "location": "basics/#CoherentStructures.STmetric",
    "page": "Basics",
    "title": "CoherentStructures.STmetric",
    "category": "type",
    "text": "STmetric(Smetric, dim, p)\n\nCreates a spatiotemporal, averaged in time metric.\n\nProperties\n\nSmetric is a metric as defined in the Distances package, e.g., Euclidean, PEuclidean, or Haversine;\ndim corresponds to the spatial dimension;\np corresponds to the kind of average applied to the vector of spatial distances:\np = Inf: maximum\np = 2: mean squared average\np = 1: arithmetic mean\np = -1: harmonic mean (does not yield a metric!)\np = -Inf: minimum (does not yield a metric!)\n\nExample\n\njulia> x, y = rand(10), rand(10)\n\njulia> Distances.evaluate(STmetric(Distances.Euclidean(),2,1),x,y)\n\n\n\n\n\n"
},

{
    "location": "basics/#Distance-computations-1",
    "page": "Basics",
    "title": "Distance computations",
    "category": "section",
    "text": "To compute distances w.r.t. standard metrics, there exists the excellent Distance.jl package. The Euclidean distance between two points is computed by any of the following lines:using Distances\nx, y = rand(3), rand(3)\nevaluate(Euclidean(),x,y)\neuclidean(x,y)Other metrics of potential interest include Haversine(r), the geodesic distance of two points on the sphere with radius r. In CoherentStructures.jl, there are two more types of metrics implemented:PEuclidean\nSTmetricThat is, the distance on a periodic torus/cylinder, and a spatiotemporal metric that interprets vectors as concatenated trajectories, applies the spatial metric to each time instance and reduces the vector of spatial distances by computing its l_p-mean. Notably, p may be any \"real\" number, including Inf and -Inf for the maximum- and \"minimum\"-norm. The spatiotemporal metric is a mathematical metric only for pgeq 1, in which case it smoothly operates with efficient sparsification methods like BallTree and inrange as implemented in the NearestNeighbors.jl package."
},

{
    "location": "fem/#",
    "page": "FEM-based methods",
    "title": "FEM-based methods",
    "category": "page",
    "text": ""
},

{
    "location": "fem/#FEM-Based-Methods-1",
    "page": "FEM-based methods",
    "title": "FEM-Based Methods",
    "category": "section",
    "text": "These methods rely on the theory outlined by Froyland\'s Dynamical Laplacian and the Geometric Heat Flow of Karrasch & Keller.The Laplace-like operators are best discretized by finite-element-based methods, see this paper by Froyland & Junge.This involves the discretization of the average of a one-parameter family of Laplace operators of the form:Delta^dyn = sum_t in mathcal T P_t^* Delta P_tfor a finite series of times mathcal T, where P_t is the transfer-operator for the flow at time t (in volume-preserving flows).The resulting operator is both symmetric and uniformly elliptic. Eigenfunctions of Delta^dyn can be used to find Lagrangian Coherent Structures.See the Examples section for examples of how these methods can be used."
},

{
    "location": "fem/#Features-1",
    "page": "FEM-based methods",
    "title": "Features",
    "category": "section",
    "text": ""
},

{
    "location": "fem/#CG-and-TO-methods-1",
    "page": "FEM-based methods",
    "title": "CG and TO methods",
    "category": "section",
    "text": "The standard Galerkin formulation of the weak dynamical Laplace is referred to as the CG-method here, due to the fact that the inverse Cauchy-Green tensor appears in the weak formulation. This gives a bilinear form overline a(uv) = sum_t in mathcal Ta^t(P_t u P_t v) Here P_t is the Transfer-Operator (or pushforward) to time-t, and a^t is the weak-form of the Laplacian on the range of the time-t map being considered.There is also a range of transfer operator-based approaches implemented here. These approximate the weak form of the Dynamical-Laplace by a bilinear-form:tilde a_h(uv) = sum_t in mathcal T a^t(I_hP_t u I_h P_t v)where I_h is a suitable interpolation operator depending on the mesh-width h. Options for I_h implemented in this package are:collocation (pointwise interpolation):\npoints used are mesh points from domain grid (\"adaptive TO\"),\npoints used are arbitrary (\"non-adaptive TO\");\nthe L^2-orthogonal projection onto an FEM-space:\nusing the forward-flow map (currently gives poor results),\nusing the inverse flow map.Note that the L^2-Galerkin methods currently perform very poorly on larger problems.For more details, see Froyland & Junge, 2018."
},

{
    "location": "fem/#Grids-1",
    "page": "FEM-based methods",
    "title": "Grids",
    "category": "section",
    "text": "Various types of regular and irregular meshes (with Delaunay triangulation using VoronoiDelaunay.jl ) are supported. These are based on the corresponding elements from JuAFEM.jl and include:triangular P1-Lagrange elements in 2D (all methods)\nquadrilateral P1-Lagrange elements in 2D (all methods except adaptive TO)\ntriangular and quadrilateral P2-Lagrange elements in 2D (all methods except adaptive TO)\ntetrahedral P1-Lagrange elements in 3D (only CG method tested, non-adaptive TO might work also)"
},

{
    "location": "fem/#The-gridContext-Type-1",
    "page": "FEM-based methods",
    "title": "The gridContext Type",
    "category": "section",
    "text": "The FEM-based methods of CoherentStructures.jl rely heavily on the JuAFEM.jl package. This package is very low-level and does not provide point-location/plotting functionality. To be able to more conveniently work with the specific types of grids that we need, all necessary variables for a single grid are combined in a gridContext structure - including the grid points, the quadrature formula used and the type of element used (e.g. Triangular P1, Quadrilateral P2, etc..). This makes it easier to assemble mass/stiffness matrices, and provides an interface for point-location and plotting.In this documentation, the variable name ctx is exclusively used for gridContext objects.See also Constructing Grids in the FEM-API section."
},

{
    "location": "fem/#Node-ordering-and-dof-ordering-1",
    "page": "FEM-based methods",
    "title": "Node ordering and dof ordering",
    "category": "section",
    "text": "Finite Element methods work with degrees of freedom (dof), which are elements of some dual space. For nodal finite elements, these correspond to evaluation functionals at the nodes of the grid.The nodes of the grid can be obtained in the following way [n.x for n in ctx.grid.nodes]. However, most of the methods of this package do not return results in this order, but instead use JuAFEM.jl\'s dof-ordering.See also the documentation in dof2node and CoherentStructures.gridContextWhen working with (non-natural) Boundary Conditions, the ordering is further changed, due to there being fewer degrees of freedom in total."
},

{
    "location": "fem/#Assembly-1",
    "page": "FEM-based methods",
    "title": "Assembly",
    "category": "section",
    "text": "See Stiffness and Mass Matrices from the FEM-API section."
},

{
    "location": "fem/#Evaluating-Functions-in-the-Approximation-Space-1",
    "page": "FEM-based methods",
    "title": "Evaluating Functions in the Approximation Space",
    "category": "section",
    "text": "given a series of coefficients that represent a function in the approximation space, to evaluate a function at a point, use the evaluate_function_from_node_or_cellvals or evaluate_function_from_dofvals functions.using CoherentStructures #hide\nusing Plots, Tensors\nctx, _ = regularP2TriangularGrid((10, 10))\nu = zeros(ctx.n)\nu[45] = 1.0\nPlots.heatmap(range(0, stop=1, length=200),range(0, stop=1, length=200),\n    (x, y) -> evaluate_function_from_dofvals(ctx, u, Vec(x, y)))For more details, consult the API: evaluate_function_from_dofvals, evaluate_function_from_node_or_cellvals, evaluate_function_from_node_or_cellvals_multiple"
},

{
    "location": "fem/#Nodal-Interpolation-1",
    "page": "FEM-based methods",
    "title": "Nodal Interpolation",
    "category": "section",
    "text": "To perform nodal interpolation of a grid, use the nodal_interpolation function."
},

{
    "location": "fem/#Boundary-Conditions-1",
    "page": "FEM-based methods",
    "title": "Boundary Conditions",
    "category": "section",
    "text": "To use something other than the natural homogeneous von Neumann boundary conditions, the CoherentStructures.boundaryData type can be used. This currently supports combinations of homogeneous Dirichlet and periodic boundary conditions.Homogeneous Dirichlet BCs require rows and columns of the stiffness/mass matrices to be deleted\nPeriodic boundary conditions require rows and columns of the stiffness/mass matrices to be added to each other.This means that the coefficient vectors for elements of the approximation space that satisfy the boundary conditions are potentially smaller and in a different order. Given a bdata argument, functions like plot_u will take this into account."
},

{
    "location": "fem/#Constructing-Boundary-Conditions-1",
    "page": "FEM-based methods",
    "title": "Constructing Boundary Conditions",
    "category": "section",
    "text": "Natural von-Neumann boundary conditions can be constructed with: boundaryData() and are generally the defaultHomogeneous Dirichlet boundary conditions can be constructed with the getHomDBCS(ctx[, which=\"all\"]) function. The optional which parameter is a vector of strings, corresponding to JuAFEM face-sets, e.g. getHomDBCS(ctx, which=[\"left\", \"right\"])Periodic boundary conditions are constructed by calling boundaryData(ctx,predicate,[which_dbc=[]]). The argument predicate is a function that should return true if and only if two points should be identified. Due to floating-point rounding errors, note that using exact comparisons (==) should be avoided. Only points that are in JuAFEM.jl boundary facesets are considered. If this is too restrictive, use the boundaryData(dbc_dofs, periodic_dofs_from, periodic_dofs_to) constructor.For details, see boundaryData."
},

{
    "location": "fem/#Example-1",
    "page": "FEM-based methods",
    "title": "Example",
    "category": "section",
    "text": "Here we apply homogeneous DBC to top and bottom, and identify the left and right side:using CoherentStructures\nctx, _ = regularQuadrilateralGrid((10, 10))\npredicate = (p1, p2) -> abs(p1[2] - p2[2]) < 1e-10 && peuclidean(p1[1], p2[1], 1.0) < 1e-10\nbdata = boundaryData(ctx, predicate, [\"top\", \"bottom\"])\nu = ones(nBCDofs(ctx, bdata))\nu[20] = 2.0; u[38] = 3.0; u[56] = 4.0\nplot_u(ctx, u, 200, 200, bdata=bdata, colorbar=:none)To apply boundary conditions to a stiffness/mass matrix, use the applyBCS function. Note that assembleStiffnessMatrix and assembleMassMatrix take a bdata argument that does this internally."
},

{
    "location": "fem/#Plotting-and-Videos-1",
    "page": "FEM-based methods",
    "title": "Plotting and Videos",
    "category": "section",
    "text": "There are some helper functions that exist for making plots and videos of functions on grids. These rely on the Plots.jl library. Plotting recipes are unfortunately not implemented.The simplest way to plot is using the plot_u function. Plots and videos of eulerian plots like f circ Phi^0_t can be made with the plot_u_eulerian and  eulerian_videos functions."
},

{
    "location": "fem/#Parallelisation-1",
    "page": "FEM-based methods",
    "title": "Parallelisation",
    "category": "section",
    "text": "Many of the plotting functions support parallelism internally. Tensor fields can be constructed in parallel, and then passed to assembleStiffnessMatrix. For an example that does this, see TODO: Add this example"
},

{
    "location": "fem/#FEM-API-1",
    "page": "FEM-based methods",
    "title": "FEM-API",
    "category": "section",
    "text": "CurrentModule = CoherentStructures"
},

{
    "location": "fem/#CoherentStructures.assembleStiffnessMatrix",
    "page": "FEM-based methods",
    "title": "CoherentStructures.assembleStiffnessMatrix",
    "category": "function",
    "text": "assembleStiffnessMatrix(ctx,A,[p; bdata])\n\nAssemble the stiffness-matrix for a symmetric bilinear form\n\na(uv) = int nabla u(x)cdot A(x)nabla v(x)f(x) dx\n\nThe integral is approximated using quadrature. A is a function that returns a SymmetricTensor{2,dim} and has one of the following forms:\n\nA(x::Vector{Float64})\nA(x::Vec{dim})\nA(x::Vec{dim}, index::Int, p). Here x is equal to ctx.quadrature_points[index], and p is that which is passed to assembleStiffnessMatrix\n\nThe ordering of the result is in dof order, except that boundary conditions from bdata are applied. The default is natural boundary conditions.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.assembleMassMatrix",
    "page": "FEM-based methods",
    "title": "CoherentStructures.assembleMassMatrix",
    "category": "function",
    "text": "assembleMassMatrix(ctx;[bdata,lumped=false])\n\nAssemble the mass matrix\n\nM_ij = int varphi_j(x) varphi_i(x) f(x)dlambda^d\n\nThe integral is approximated using numerical quadrature. The values of f(x) are taken from ctx.mass_weights, and should be ordered in the same way as ctx.quadrature_points\n\nThe result is ordered in a way so as to be usable with a stiffness matrix with boundary data bdata.\n\nReturns a lumped mass matrix if lumped==true.\n\nExample\n\nctx.mass_weights = map(f, ctx.quadrature_points)\nM = assembleMassMatrix(ctx)\n\n\n\n\n\n"
},

{
    "location": "fem/#Stiffness-and-Mass-Matrices-1",
    "page": "FEM-based methods",
    "title": "Stiffness and Mass Matrices",
    "category": "section",
    "text": "assembleStiffnessMatrix\nassembleMassMatrix"
},

{
    "location": "fem/#Constructing-Grids-1",
    "page": "FEM-based methods",
    "title": "Constructing Grids",
    "category": "section",
    "text": "There are several helper functions available for constructing grids. The simplest is:"
},

{
    "location": "fem/#CoherentStructures.regular1dGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regular1dGrid",
    "category": "function",
    "text": "regular1dGrid(numnodes,left=0.0,right=1.0; [quadrature_order, PC=false])\n\nCreate a regular grid with numnodes nodes on the interval [left,right] in 1d. If PC==false, uses P1-Lagrange basis functions. If PC=true, uses piecewise-constant basis functions.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regular1dP2Grid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regular1dP2Grid",
    "category": "function",
    "text": "regular1dP2Grid(numnodes, [left,right; quadrature_order])\n\nCreate a regular grid with numnodes non-interior nodes on the interval [left,right]. Uses P2-Lagrange elements.\n\n\n\n\n\n"
},

{
    "location": "fem/#In-1D-1",
    "page": "FEM-based methods",
    "title": "In 1D",
    "category": "section",
    "text": "regular1dGrid\nregular1dP2Grid"
},

{
    "location": "fem/#CoherentStructures.regular2dGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regular2dGrid",
    "category": "function",
    "text": "regular2dGrid(gridType, numnodes, LL=[0.0,0.0],UR=[1.0,1.0];quadrature_order=default_quadrature_order)\n\nConstructs a regular grid. gridType should be from CoherentStructures.regular2dGridTypes\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularTriangularGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularTriangularGrid",
    "category": "function",
    "text": "regularTriangularGrid(numnodes, LL,UR;[quadrature_order, PC=false])\n\nCreate a regular triangular grid on a rectangle; does not use Delaunay triangulation internally. If\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularDelaunayGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularDelaunayGrid",
    "category": "function",
    "text": "regularDelaunayGrid(numnodes=(25,25), LL, UR; [quadrature_order,on_torus=false, nudge_epsilon=1e-5,PC=false])\n\nCreate a regular grid on a square with lower left corner LL and upper-right corner UR. Internally uses Delauny Triangulation. If on_torus==true, uses a periodic Delaunay triangulation. To avoid degenerate special cases, all nodes are given a random nudge, the strength of which depends on numnodes and nudge_epsilon. If PC==true, returns a piecewise constant grid. Else returns a P1-Lagrange grid.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.irregularDelaunayGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.irregularDelaunayGrid",
    "category": "function",
    "text": "irregularDelaunayGrid(nodes_in; [on_torus=true,LL,UR,PC=false,...])\n\nTriangulate the nodes nodes_in and return a gridContext and bdata for them. If on_torus==true, the triangulation is done on a torus. If PC==true, return a mesh with piecewise constant shape-functions, else P1 Lagrange.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularP2TriangularGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2TriangularGrid",
    "category": "function",
    "text": "regularP2TriangularGrid(numnodes=(25,25), LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)\n\nCreate a regular P2 triangular grid on a Rectangle. Does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularP2DelaunayGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2DelaunayGrid",
    "category": "function",
    "text": "regularP2DelaunayGrid(numnodes=(25,25),LL=[0.0,0.0],UR=[1.0,1.0],quadrature_order=default_quadrature_order)\n\nCreate a regular P2 triangular grid with numnodes being the number of (non-interior) nodes in each direction.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularQuadrilateralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularQuadrilateralGrid",
    "category": "function",
    "text": "regularP2QuadrilateralGrid(numnodes, LL,UR;[quadrature_order, PC=false]\n\nCreate a regular P1 quadrilateral grid on a Rectangle. If PC==true, use piecewise constant shape functions. Else use P1 Lagrange.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularP2QuadrilateralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2QuadrilateralGrid",
    "category": "function",
    "text": "regularP2QuadrilateralGrid(numnodes=(25,25), LL=[0.0,0.0], UR=[1.0,1.0], quadrature_order=default_quadrature_order)\n\nCreate a regular P2 quadrilateral grid on a rectangle.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularTetrahedralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularTetrahedralGrid",
    "category": "function",
    "text": "regularTetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)\n\nCreate a regular P1 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.regularP2TetrahedralGrid",
    "page": "FEM-based methods",
    "title": "CoherentStructures.regularP2TetrahedralGrid",
    "category": "function",
    "text": "regularP2TetrahedralGrid(numnodes=(10,10,10), LL=[0.0,0.0,0.0], UR=[1.0,1.0,1.0], quadrature_order=default_quadrature_order3D)\n\nCreate a regular P2 tetrahedral grid on a Cuboid in 3D. Does not use Delaunay triangulation internally.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.gridContext",
    "page": "FEM-based methods",
    "title": "CoherentStructures.gridContext",
    "category": "type",
    "text": "struct gridContext<dim>\n\nStores everything needed as \"context\" to be able to work on a FEM grid based on the JuAFEM package. Adds a point-locator API which facilitates plotting functions defined on the grid within Julia.\n\nFields\n\ngrid::JuAFEM.Grid, ip::JuAFEM.Interpolation,ip_geom::JuAFEM.Interpolation, qr::JuAFEM.QuadratureRule - See the JuAFEM package\nloc::CellLocator object used for point-location on the grid.\nnode_to_dof::Vector{Int}  lookup table for dof index of a node (for Lagrange elements)\ndof_to_node::Vector{Int}  inverse of nodetodof\ncell_to_dof::Vector{Int}  lookup table for dof index of a cell (for piecewise constant elements)\ndof_to_cell::Vector{Int}  inverse of celltodof\nnum_nodes::Int number of nodes on the grid\nnum_cells::Int number of elements (e.g. triangles,quadrilaterals, ...) on the grid\nn number of degrees of freedom (== num_nodes for Lagrange Elements, and == num_cells for piecewise constant elements)\nquadrature_points::Vector{Vec{dim,Float64}} All quadrature points on the grid, in a fixed order.\nmass_weights::Vector{Float64} Weighting for stiffness/mass matrices\nspatialBounds If available, the corners of a bounding box of a domain. For regular grids, the bounds are tight.\nnumberOfPointsInEachDirection For regular grids, how many (non-interior) nodes make up the regular grid.\ngridType A string describing what kind of grid this is (e.g. \"regular triangular grid\")\n\n\n\n\n\n"
},

{
    "location": "fem/#In-2D-1",
    "page": "FEM-based methods",
    "title": "In 2D",
    "category": "section",
    "text": "regular2dGridSupported values for the gridType argument are:using CoherentStructures #hide\nCoherentStructures.regular2dGridTypesThe following functions are conceptually similar:regularTriangularGrid\nregularDelaunayGrid\nirregularDelaunayGrid\nregularP2TriangularGrid\nregularP2DelaunayGrid\nregularQuadrilateralGrid\nregularP2QuadrilateralGridIn 3D we haveregularTetrahedralGrid\nregularP2TetrahedralGridAll of these methods return a gridContext object and a boundaryData object. The latter is only relevant when using a Delaunay grid with on_torus==true.CoherentStructures.gridContext"
},

{
    "location": "fem/#CoherentStructures.boundaryData",
    "page": "FEM-based methods",
    "title": "CoherentStructures.boundaryData",
    "category": "type",
    "text": "mutable struct boundaryData\n\nRepresent (a combination of) homogeneous Dirichlet and periodic boundary conditions. Fields:\n\ndbc_dofs list of dofs that should have homogeneous Dirichlet boundary conditions. Must be sorted.\nperiodic_dofs_from and periodic_dofs_to are both Vector{Int}. The former must be strictly increasing, both must be the same length. periodic_dofs_from[i] is identified with periodic_dofs_to[i]. periodic_dofs_from[i] must be strictly larger than periodic_dofs_to[i]. Multiple dofs can be identified with the same dof. If some dof is identified with another dof and one of them is in dbc_dofs, both points must be in dbc_dofs\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.getHomDBCS",
    "page": "FEM-based methods",
    "title": "CoherentStructures.getHomDBCS",
    "category": "function",
    "text": "getHomDBCS(ctx,which=\"all\")\n\nReturn boundaryData object corresponding to homogeneous Dirichlet Boundary Conditions for a set of facesets. which=\"all\" is shorthand for [\"left\",\"right\",\"top\",\"bottom\"].\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.undoBCS",
    "page": "FEM-based methods",
    "title": "CoherentStructures.undoBCS",
    "category": "function",
    "text": "undoBCS(ctx, u, bdata)\n\nGiven a vector u in dof order with boundary conditions applied, return the corresponding u in dof order without the boundary conditions.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.applyBCS",
    "page": "FEM-based methods",
    "title": "CoherentStructures.applyBCS",
    "category": "function",
    "text": "applyBCS(ctx_row, K, bdata_row; [ctx_col, bdata_col, bdata_row, add_vals=true])\n\nApply the boundary conditions from bdata_row and bdata_col to the sparse matrix K. Only applies boundary conditions accross columns (rows) if bdata_row==nothing (bdata_col==nothing) If add_vals==true, then\n\n\n\n\n\n"
},

{
    "location": "fem/#Boundary-Conditions-API-1",
    "page": "FEM-based methods",
    "title": "Boundary Conditions API",
    "category": "section",
    "text": "boundaryData\ngetHomDBCS\nundoBCS\napplyBCS"
},

{
    "location": "fem/#CoherentStructures.dof2node",
    "page": "FEM-based methods",
    "title": "CoherentStructures.dof2node",
    "category": "function",
    "text": "dof2node(ctx,u)\n\nInterprets u as an array of coefficients ordered in dof order, and reorders them to be in node order.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.getDofCoordinates",
    "page": "FEM-based methods",
    "title": "CoherentStructures.getDofCoordinates",
    "category": "function",
    "text": "getDofCoordinates(ctx,dofindex)\n\nReturn the coordinates of the node corresponding to the dof with index dofindex\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.evaluate_function_from_dofvals",
    "page": "FEM-based methods",
    "title": "CoherentStructures.evaluate_function_from_dofvals",
    "category": "function",
    "text": "evaluate_function_from_dofvals(ctx, dofvals, x_in; outside_value=0.0,project_in=fals)\n\nEvaluate the function at point xin with coefficients of dofs given by dofvals (in dof-order). Return `outsidevalueif point is out of bounds. Project the point into the domain ifprojectin==true. For evaluation at many points, or for many dofvals, the functionevaluatefunctionfromdofvals_multiple` is more efficient.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.evaluate_function_from_node_or_cellvals",
    "page": "FEM-based methods",
    "title": "CoherentStructures.evaluate_function_from_node_or_cellvals",
    "category": "function",
    "text": "evaluate_function_from_node_or_cellvals(ctx, vals, x_in; outside_value=0, project_in=false)\n\nLike evaluate_function_from_dofvals, but the coefficients from vals are assumed to be in node order. This is more efficient than evaluate_function_from_dofvals.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.evaluate_function_from_node_or_cellvals_multiple",
    "page": "FEM-based methods",
    "title": "CoherentStructures.evaluate_function_from_node_or_cellvals_multiple",
    "category": "function",
    "text": "evaluate_function_from_node_or_cellvals_multiple(ctx, vals, xin; is_diag=false, kwargs...)\n\nLike evaluate_function_from_dofvals_multiple but uses node- (or cell- if piecewise constant interpolation) ordering for vals, which makes it slightly more efficient. If vals is a diagonal matrix, set is_diag to true for much faster evaluation.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.nodal_interpolation",
    "page": "FEM-based methods",
    "title": "CoherentStructures.nodal_interpolation",
    "category": "function",
    "text": "nodal_interpolation(ctx,f)\n\nPerform nodal interpolation of a function. Returns a vector of coefficients in dof order\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.getH",
    "page": "FEM-based methods",
    "title": "CoherentStructures.getH",
    "category": "function",
    "text": "getH(ctx)\n\nReturn the mesh width of a regular grid.\n\n\n\n\n\n"
},

{
    "location": "fem/#Helper-functions-1",
    "page": "FEM-based methods",
    "title": "Helper functions",
    "category": "section",
    "text": "dof2node\ngetDofCoordinatesevaluate_function_from_dofvals\nevaluate_function_from_node_or_cellvals\nevaluate_function_from_node_or_cellvals_multiplenodal_interpolationgetH"
},

{
    "location": "fem/#Plotting-API-1",
    "page": "FEM-based methods",
    "title": "Plotting API",
    "category": "section",
    "text": ""
},

{
    "location": "fem/#CoherentStructures.plot_u",
    "page": "FEM-based methods",
    "title": "CoherentStructures.plot_u",
    "category": "function",
    "text": "plot_u(ctx, dof_vals, nx, ny; bdata=nothing, kwargs...)\n\nPlot the function with coefficients (in dof order, possible boundary conditions in bdata) given by dof_vals on the grid ctx. The domain to be plotted on is given by ctx.spatialBounds. The function is evaluated on a regular nx by ny grid, the resulting plot is a heatmap. Keyword arguments are passed down to plot_u_eulerian, which this function calls internally.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.plot_u_eulerian",
    "page": "FEM-based methods",
    "title": "CoherentStructures.plot_u_eulerian",
    "category": "function",
    "text": "plot_u_eulerian(ctx,dof_vals,inverse_flow_map,\n    LL,UR,nx,ny,\n    euler_to_lagrange_points=nothing, only_get_lagrange_points=false,\n    z=nothing,\n    postprocessor=nothing,\n    bdata=nothing, ....)\n\nPlot a heatmap of a function in eulerian coordinates, i.e. the pushforward of f. This is given by f circ Phi^-1, f is a function defined on the grid ctx, represented by coefficients given by dof_vals (with possible boundary conditions given in bdata)\n\nThe argument inverse_flow_map is Phi^-1.\n\nThe resulting plot is on a regular nx by ny grid on the grid with lower left corner LL and upper right corner UR.\n\nPoints that fall outside of the domain represented by ctx are plotted as NaN, which results in transparency.\n\nOne can pass values to be plotted directly by providing them in an array in the argument z. postprocessor can modify the values being plotted, return_scalar_field results in these values being returned.  See the source code for further details.  Additional arguments are passed to Plots.heatmap\n\nInverse flow maps are computed in parallel if there are multiple workers.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.eulerian_videos",
    "page": "FEM-based methods",
    "title": "CoherentStructures.eulerian_videos",
    "category": "function",
    "text": " eulerian_videos(ctx, us, inverse_flow_map_t, t0,tf, nx,ny,nt, LL,UR, num_videos=1;\n    extra_kwargs_fun=nothing, ...)\n\nCreate num_videos::Int videos in eulerian coordinates, i.e. where the time t is varied, plot f_i circ Phi_t^0 for f_1 dots.\n\nus(i,t) is a vector of dofs to be plotted at time t for the ith video.\n\ninverse_flow_map_t(t,x) is Phi_t^0(x)\n\nt0, tf  are initial and final time. Spatial bounds are given by LL,UR\n\nnx,ny,nt give the number of points in each direction.\n\nextra_kwargs_fun(i,t) can be used to provide additional keyword arguments to Plots.heatmap()\n\nAdditional kwargs are passed down to plot_eulerian_video\n\nAs much as possible is done in parallel.\n\nReturns a Vector of iterables result. Call Plots.animate(result[i]) to get an animation.\n\n\n\n\n\n"
},

{
    "location": "fem/#CoherentStructures.eulerian_video",
    "page": "FEM-based methods",
    "title": "CoherentStructures.eulerian_video",
    "category": "function",
    "text": "eulerian_video(ctx, u, inverse_flow_map_t,t0,tf, nx, ny, nt, LL, UR;extra_kwargs_fun=nothing,...)\n\nLike eulerian_videos, but u(t) is a vector of dofs, and extra_kwargs_fun(t) gives extra keyword arguments. Returns only one result, on which Plots.animate() can be applied.\n\n\n\n\n\n"
},

{
    "location": "fem/#FEM-1",
    "page": "FEM-based methods",
    "title": "FEM",
    "category": "section",
    "text": "plot_u\nplot_u_eulerian\neulerian_videos\neulerian_video"
},

{
    "location": "fem/#CoherentStructures.plot_ftle",
    "page": "FEM-based methods",
    "title": "CoherentStructures.plot_ftle",
    "category": "function",
    "text": "plot_ftle(odefun,p,tspan,LL,UR,nx,ny;\n    δ=1e-9,tolerance=1e-4,solver=OrdinaryDiffEq.BS5(),\n    existing_plot=nothing,flip_y=false, check_inbounds=always_true, pass_on_errors=false)\n\nMake a heatmap of a FTLE field using finite differences. If existing_plot is given a value, plot using heatmap! on top of it. If flip_y is true, then flip the y-coordinate (needed sometimes due to a bug in Plots). Points where check_inbounds(x[1],x[2],p) == false are set to NaN (i.e. transparent). Unless pass_on_errors is set to true, errors from calculating FTLE values are caught and ignored.\n\n\n\n\n\n"
},

{
    "location": "fem/#Other-plotting-utilities-1",
    "page": "FEM-based methods",
    "title": "Other plotting utilities",
    "category": "section",
    "text": "plot_ftle"
},

{
    "location": "fem/#Defaults-1",
    "page": "FEM-based methods",
    "title": "Defaults",
    "category": "section",
    "text": "const default_quadrature_order=5\nconst default_solver = OrdinaryDiffEq.BS5()"
},

{
    "location": "elliptic/#",
    "page": "Geodesic vortices",
    "title": "Geodesic vortices",
    "category": "page",
    "text": ""
},

{
    "location": "elliptic/#Geodesic-elliptic-material-vortices-1",
    "page": "Geodesic vortices",
    "title": "Geodesic elliptic material vortices",
    "category": "section",
    "text": "CurrentModule = CoherentStructuresThe following functions implement an LCS methodology developed in the following papers:Haller & Beron-Vera, 2012\nHaller & Beron-Vera, 2013\nKarrasch, Huhn, and Haller, 2015The present code was originally inspired by Alireza Hadjighasem\'s MATLAB implementation, which was written in the context of the SIAM Review paper, but has been significantly modified and improved throughout. Depending on the indefinite metric tensor field used, the functions below yield the following types of coherent structures:black-hole/Lagrangian coherent vortices (Haller & Beron-Vera, 2012)\nelliptic objective Eulerian coherent structures (OECSs) (Serra & Haller, 2016)\nmaterial diffusive transport barriers (Haller, Karrasch, and Kogelbauer, 2018)The general procedure is the following. Assume T is the symmetric tensor field of interest, say, (i) the Cauchy-Green strain tensor field C, (ii) the rate-of-strain tensor field S, or (iii) the averaged diffusion-weighted Cauchy-Green tensor field barC_D; cf. the references above. Denote by 0lambda_1leqlambda_2 the eigenvalue and by xi_1 and xi_2 the corresponding eigenvector fields of T. Then the direction fields of interest are given byeta_lambda^pm = sqrtfraclambda_2 - lambdalambda_2-lambda_1xi_1 pm sqrtfraclambda - lambda_1lambda_2-lambda_1xi_2Tensor singularities are defined as points at which lambda_2=lambda_1, i.e., at which the two characteristic directions xi_1 and xi_2 are not well-defined. As described and exploited in Karrasch et al., 2015, non-negligible tensor singularities express themselves by an angle gap when tracking (the angle of) tensor eigenvector fields along closed paths surrounding the singularity. Our approach here avoids computing singularities directly, but rather computes the index for each grid cell and then combines nearby singularities, i.e., adds non-vanishing indices of nearby grid cells.In summary, the implementation consists of the following steps:compute the index for each grid cell and combine nearby singular grid cells to \"singularity candidates\";\nlook for elliptic singularity candidates (and potentially isolated wedge pairs);\nplace an eastwards oriented Poincaré section at the pair center;\nfor each point on the discretized Poincaré section, scan through the given parameter interval such that the corresponding η-orbit closes at that point;\nif desired: for each Poincaré section, take the outermost closed orbit as the coherent vortex barrier (if there exist any)."
},

{
    "location": "elliptic/#Function-documentation-1",
    "page": "Geodesic vortices",
    "title": "Function documentation",
    "category": "section",
    "text": ""
},

{
    "location": "elliptic/#The-meta-functions-ellipticLCS-and-constrainedLCS-1",
    "page": "Geodesic vortices",
    "title": "The meta-functions ellipticLCS and constrainedLCS",
    "category": "section",
    "text": "The fully automated high-level functions are:ellipticLCS\nconstrainedLCSOne of their arguments is a list of parameters used in the LCS detection. This list is combined in a data type called LCSParameters. The output is a list of EllipticBarriers and a list of Singularitys. There is an option to retrieve all closed barriers (outermost=false), in contrast to extracting only the outermost vortex boundaries (outermost=true), which is more efficient.The meta-functions consist of two steps: first, the index theory-based determination of where to search for closed orbits,, cf. Index theory-based placement of Poincaré sections; second, the closed orbit computation, cf. Closed orbit detection."
},

{
    "location": "elliptic/#CoherentStructures.LCSParameters",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.LCSParameters",
    "category": "type",
    "text": "struct LCSParameters\n\nContainer for parameters used in elliptic LCS computations.\n\nFields\n\nindexradius::Float64=0.1: radius for singularity type detection\nboxradius::Float64=0.5: \"radius\" of localization square for closed orbit detection\ncombine_pairs=true: whether isolated singularity pairs should be merged\nn_seeds::Int64=40: number of seed points on the Poincaré section\npmin::Float64=0.7: lower bound on the parameter in the eta-field\npmax::Float64=1.3: upper bound on the parameter in the eta-field\nrdist::Float64=1e-4: required return distances for closed orbits\n\n\n\n\n\n"
},

{
    "location": "elliptic/#CoherentStructures.EllipticBarrier",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.EllipticBarrier",
    "category": "type",
    "text": "struct EllipticBarrier\n\nThis is a container for coherent vortex boundaries. An object vortex of type EllipticBarrier can be easily plotted by plot(vortex.curve), or plot!([figure, ]vortex.curve) if it is to be overlaid over an existing plot.\n\nFields\n\ncurve: a list of tuples, contains the coordinates of coherent vortex boundary points;\ncore: location of the vortex core;\np: contains the parameter value of the direction field eta_lambda^pm, for the curve is a closed orbit;\ns: a Bool value, which encodes the sign in the formula of the direction field eta_lambda^pm via the formula (-1)^s.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#Specific-types-1",
    "page": "Geodesic vortices",
    "title": "Specific types",
    "category": "section",
    "text": "These are the specifically introduced types for elliptic LCS computations.LCSParameters\nEllipticBarrierAnother one is Singularity, which comes along with some convenience functions.Singularity\ngetcoords\ngetindices"
},

{
    "location": "elliptic/#CoherentStructures.singularity_detection",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.singularity_detection",
    "category": "function",
    "text": "singularity_detection(T, combine_distance; combine_isolated_wedges=true) -> Vector{Singularity}\n\nCalculates line-field singularities of the first eigenvector of T by taking a discrete differential-geometric approach. Singularities are calculated on each cell. Singularities with distance less or equal to combine_distance are combined by averaging the coordinates and adding the respective indices. If combine_pairs is `true, pairs of singularities that are mutually the closest ones are included in the final list.\n\nReturns a vector of Singularitys. Returned indices correspond to doubled indices to get integer values.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#CoherentStructures.critical_point_detection",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.critical_point_detection",
    "category": "function",
    "text": "critical_point_detection(vs, combine_distance, γ; combine_pairs=true)\n\nComputes critical points of a vector/line field vs, given as an AxisArray. Critical points with distance less or equal to combine_distance are combined by averaging the coordinates and adding the respective indices. The parameter γ should be chosen π for line fields and 2π for vector fields; cf. compute_singularities. If combine_pairs istrue, pairs of singularities that are mutually the closest ones are included in the final list.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#CoherentStructures.compute_singularities",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.compute_singularities",
    "category": "function",
    "text": "compute_singularities(α, modulus) -> Vector{Singularity}\n\nComputes critical points/singularities of vector and line fields, respectively. α is a scalar field (array) which is assumed to contain some consistent angle representation of the vector/line field. Choose modulus=2π for vector fields, and modulus=π for line fields.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#CoherentStructures.combine_singularities",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.combine_singularities",
    "category": "function",
    "text": "combine_singularities(sing_coordinates, sing_indices, combine_distance) -> Vector{Singularity}\n\nThis function does the equivalent of: Build a graph where singularities are vertices, and two vertices share an edge iff the coordinates of the corresponding vertices (given by sing_coordinates) have a distance leq combine_distance. Find all connected components of this graph, and return a list of their mean coordinate and sum of sing_indices\n\n\n\n\n\n"
},

{
    "location": "elliptic/#CoherentStructures.combine_isolated_pairs",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.combine_isolated_pairs",
    "category": "function",
    "text": "combine_isolated_pairs(singularities)\n\nDetermines singularities which are mutually closest neighbors and combines them as one, while adding their indices.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#Index-theory-based-placement-of-Poincaré-sections-1",
    "page": "Geodesic vortices",
    "title": "Index theory-based placement of Poincaré sections",
    "category": "section",
    "text": "This is performed by discrete_singularity_detection for line fields (such as eigenvector fields of symmetric positive-definit tensor fields) and by critical_point_detection for classic vector fields.singularity_detection\ncritical_point_detectionThis function takes three steps.compute_singularities\ncombine_singularities\ncombine_isolated_pairsFrom all virtual/merged singularities those with a suitable index are selected. Around each elliptic singularity the tensor field is localized and passed on for closed orbit detection."
},

{
    "location": "elliptic/#CoherentStructures.compute_returning_orbit",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.compute_returning_orbit",
    "category": "function",
    "text": "compute_returning_orbit(vf, seed::SVector{2}, save::Bool=false)\n\nComputes returning orbits under the velocity field vf, originating from seed. The optional argument save controls whether intermediate locations of the returning orbit should be saved.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#CoherentStructures.compute_closed_orbits",
    "page": "Geodesic vortices",
    "title": "CoherentStructures.compute_closed_orbits",
    "category": "function",
    "text": "compute_closed_orbits(ps, ηfield, cache; rev=true, pmin=0.7, pmax=1.5, rdist=1e-4)\n\nCompute the outermost closed orbit for a given Poincaré section ps, a vector field constructor ηfield, and an LCScache cache. Keyword argumentspminandpmaxcorrespond to the range of shift parameters in which closed orbits are sought;revdetermines whether closed orbits are sought from the outside inwards (true) or from the inside outwards (false).rdist` sets the required return distance for an orbit to be considered as closed.\n\n\n\n\n\n"
},

{
    "location": "elliptic/#Closed-orbit-detection-1",
    "page": "Geodesic vortices",
    "title": "Closed orbit detection",
    "category": "section",
    "text": "compute_returning_orbit\ncompute_closed_orbits"
},

{
    "location": "Laplace/#",
    "page": "Graph Laplacian-based methods",
    "title": "Graph Laplacian-based methods",
    "category": "page",
    "text": ""
},

{
    "location": "Laplace/#Graph-Laplacian/diffusion-maps-based-LCS-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Graph Laplacian/diffusion maps-based LCS methods",
    "category": "section",
    "text": "CurrentModule = CoherentStructuresCite a couple of important papers:Shi & Malik, Normalized cuts and image segmentation, 2000\nCoifman & Lafon, Diffusion maps, 2006\nMarshall & Hirn, Time coupled diffusion maps, 2018In the LCS context, we havesomewhat related Froyland & Padberg-Gehle, 2015\nHadjighasem et al., 2016\nBanisch & Koltai, 2017\nRypina et al., 2017/Padberg-Gehle & Schneide, 2018\nDe Diego et al., 2018"
},

{
    "location": "Laplace/#Function-documentation-1",
    "page": "Graph Laplacian-based methods",
    "title": "Function documentation",
    "category": "section",
    "text": ""
},

{
    "location": "Laplace/#CoherentStructures.KNN",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.KNN",
    "category": "type",
    "text": "KNN(k)\n\nDefines the KNN (k-nearest neighbors) sparsification method. In this approach, first k nearest neighbors are sought. In the final graph Laplacian, only those particle pairs are included which are contained in some k-neighborhood.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.mutualKNN",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.mutualKNN",
    "category": "type",
    "text": "mutualKNN(k)\n\nDefines the mutual KNN (k-nearest neighbors) sparsification method. In this approach, first k nearest neighbors are sought. In the final graph Laplacian, only those particle pairs are included which are mutually contained in each others k-neighborhood.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.neighborhood",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.neighborhood",
    "category": "type",
    "text": "neighborhood(ε)\n\nDefines the ε-neighborhood sparsification method. In the final graph Laplacian, only those particle pairs are included which have distance less than ε.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#Sparsification-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Sparsification methods",
    "category": "section",
    "text": "Two commonly used sparsification methods are implemented for use with various graph Laplacian methods, see below.KNN\nmutualKNN\nneighborhoodOther sparsification methods can be implemented by defining a corresponding sparseaffinitykernel instance."
},

{
    "location": "Laplace/#CoherentStructures.diff_op",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.diff_op",
    "category": "function",
    "text": "diff_op(data, sp_method, kernel = gaussian_kernel; α=1.0, metric=Euclidean\"()\")\n\nReturn a diffusion/Markov matrix P.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nsp_method: employed sparsification method (neighborhood or mutualKNN);\nkernel: diffusion kernel, e.g., x -> exp(-x*x/4σ);\nα: exponent in diffusion-map normalization;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.sparse_diff_op_family",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_diff_op_family",
    "category": "function",
    "text": "sparse_diff_op_family(data, sp_method, kernel=gaussian_kernel, dim=2; op_reduce, α, metric)\n\nReturn a list of sparse diffusion/Markov matrices P.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nsp_method: sparsification method;\nkernel: diffusion kernel, e.g., x -> exp(-x*x/4σ);\ndim: the columns are interpreted as concatenations of dim- dimensional points, to which metric is applied individually;\nop_reduce: time-reduction of diffusion operators, e.g. mean or P -> prod(LMs.LinearMap,Iterators.reverse(P)) (default)\nα: exponent in diffusion-map normalization;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.sparse_diff_op",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_diff_op",
    "category": "function",
    "text": "sparse_diff_op(data, sp_method, kernel; α=1.0, metric=Euclidean()) -> SparseMatrixCSC\n\nReturn a sparse diffusion/Markov matrix P.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nsp_method: sparsification method;\nkernel: diffusion kernel, e.g., x -> exp(-x*x) (default);\nα: exponent in diffusion-map normalization;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.sparseaffinitykernel",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparseaffinitykernel",
    "category": "function",
    "text": "sparseaffinitykernel(data, sp_method, kernel, metric=Euclidean()) -> SparseMatrixCSC\n\nReturn a sparse matrix W where w_ij = k(x_i x_j). The x_i are taken from the columns of data. Entries are only calculated for pairs determined by the sparsification method sp_method. Default metric is Euclidean().\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.kde_normalize!",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.kde_normalize!",
    "category": "function",
    "text": "kde_normalize!(A, α = 1.0)\n\nNormalize rows and columns of A in-place with the respective row-sum to the α-th power; i.e., return a_ij=a_ijq_i^alphaq_j^alpha, where q_k = sum_ell a_kell. Default for α is 1.0.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.wLap_normalize!",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.wLap_normalize!",
    "category": "function",
    "text": "wLap_normalize!(A)\n\nNormalize rows of A in-place with the respective row-sum; i.e., return a_ij=a_ijq_i.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#Diffusion-maps-type-graph-Laplacian-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Diffusion-maps type graph Laplacian methods",
    "category": "section",
    "text": "diff_op\nsparse_diff_op_family\nsparse_diff_op\nsparseaffinitykernel\nkde_normalize!\nwLap_normalize!"
},

{
    "location": "Laplace/#CoherentStructures.sparse_adjacency",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_adjacency",
    "category": "function",
    "text": "sparse_adjacency(data, ε[, dim]; metric) -> SparseMatrixCSC\n\nReturn a sparse adjacency matrix A with integer entries 0 or 1. If the third argument dim is passed, then data is interpreted as concatenated points of length dim, to which metric is applied individually. Otherwise, metric is applied to the whole columns of data.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nε: distance threshold;\ndim: the columns of data are interpreted as concatenations of dim- dimensional points, to which metric is applied individually;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.sparse_adjacency_list",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.sparse_adjacency_list",
    "category": "function",
    "text": "sparse_adjacency_list(data, ε; metric=Euclidean()) -> idxs::Vector{Vector}\n\nReturn two lists of indices of data points that are adjacent.\n\nArguments\n\ndata: 2D array with columns correspdonding to data points;\nε: distance threshold;\nmetric: distance function w.r.t. which the kernel is computed, however, only for point pairs where metric(x_i x_j)leq varepsilon.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#Adjancency-matrix-based-graph-Laplacian-methods-1",
    "page": "Graph Laplacian-based methods",
    "title": "Adjancency-matrix-based graph Laplacian methods",
    "category": "section",
    "text": "sparse_adjacency\nsparse_adjacency_list"
},

{
    "location": "Laplace/#CoherentStructures.diffusion_coordinates",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.diffusion_coordinates",
    "category": "function",
    "text": "diffusion_coordinates(P,n_coords) -> (Σ::Vector, Ψ::Matrix)\n\nCompute the (time-coupled) diffusion coordinates Ψ and the coordinate weights Σ for a linear map P. n_coords determines the number of diffusion coordinates to be computed.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#CoherentStructures.diffusion_distance",
    "page": "Graph Laplacian-based methods",
    "title": "CoherentStructures.diffusion_distance",
    "category": "function",
    "text": "diffusion_distance(diff_coord) -> SymmetricMatrix\n\nReturns the distance matrix of pairs of points whose diffusion distances correspond to the diffusion coordinates given by diff_coord.\n\n\n\n\n\n"
},

{
    "location": "Laplace/#Diffusion-coordinate-like-functions-1",
    "page": "Graph Laplacian-based methods",
    "title": "Diffusion-coordinate-like functions",
    "category": "section",
    "text": "diffusion_coordinates\ndiffusion_distance"
},

]}
