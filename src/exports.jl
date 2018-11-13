export

	#advection_diffusion.jl
	FEM_heatflow,
	implicitEulerStepFamily,
	ADimplicitEulerStep,

	#ellipticLCS.jl
	EllipticVortex,
	LCSParameters,
	singularity_location_detection,
	singularity_type_detection,
	detect_elliptic_region,
	set_Poincaré_section,
	compute_returning_orbit,
	compute_outermost_closed_orbit,
	ellipticLCS,

	#diffusion_operators.jl
	KNN,
	mutualKNN,
	neighborhood,
	DM_heatflow,
	diff_op,
	sparse_diff_op_family,
	sparse_diff_op,
	sparseaffinitykernel,
	kde_normalize!,
	wlap_normalize!,
	sparse_adjacency_family,
	sparse_adjacency,
	stationary_distribution,
	diffusion_coordinates,
	diffusion_distance,

	#dynamicmetrics
	PEuclidean,
	peuclidean,
	STmetric,
	stmetric,

	#FEMassembly.jl
	assembleStiffnessMatrix,
	assembleMassMatrix,

	#gridfunctions.jl
	regular1dGrid,
	regular1dP2Grid,
	regularTriangularGrid,
	regularDelaunayGrid,
	irregularDelaunayGrid,
	randomDelaunayGrid,
	regularP2TriangularGrid,
	regularP2DelaunayGrid,
	regularQuadrilateralGrid,
	regularP2QuadrilateralGrid,
	regularTetrahedralGrid,
	regularP2TetrahedralGrid,
	regularGrid,
	periodicDelaunayGrid,
	randomPeriodicDelaunayGrid,
	aperiodicDelaunayGrid,
	evaluate_function_from_dofvals,
	evaluate_function_from_nodevals,
	locatePoint,
	nodal_interpolation,
    sample_to,
	undoBCS,
	doBCS,
	applyBCS,
	getHomDBCS,
	boundaryData,
	nDofs,
	getDofCoordinates,

	#numericalExperiments.jl
	makeOceanFlowTestCase,
	makeDoubleGyreTestCase,
	experimentResult,
	runExperiment!,
	plotExperiment,

	#ulam,jl
	ulam,

	#plotting.jl
	plot_u,
	plot_u!,
	plot_spectrum,
	plot_real_spectrum,
	plot_u_eulerian,
	plot_ftle,
	eulerian_videos,
	eulerian_video,

	#pullbacktensors.jl
	flow,
	parallel_flow,
	linearized_flow,
	parallel_tensor,
	mean_diff_tensor,
	CG_tensor,
	pullback_tensors,
	pullback_metric_tensor,
	pullback_diffusion_tensor,
	pullback_diffusion_tensor_function,
	pullback_SDE_diffusion_tensor,
	av_weighted_CG_tensor,

	#streammacros.jl
    @define_stream,
    @velo_from_stream,
    @var_velo_from_stream,
	@vorticity_from_stream,

	#TO.jl
	nonAdaptiveTOCollocation,
	adaptiveTOCollocation,
	adaptiveTOCollocationStiffnessMatrix,
	L2GalerkinTO,
	L2GalerkinTOFromInverse,
	adaptiveTOCollocation,

	#util.jl
	tensor_invariants,
	dof2node,
	kmeansresult2LCS,
	getH,
	unzip,

	#velocityfields.jl
	rot_double_gyre,
	rot_double_gyre!,
	rot_double_gyreEqVari,
	rot_double_gyreEqVari!,
	bickleyJet,
	bickleyJet!,
	bickleyJetEqVari,
	bickleyJetEqVari!,
	interpolateVF,
	interp_rhs,
	interp_rhs!,
	standardMap,
	standardMapInv,
	DstandardMap,
	abcFlow,
	cylinder_flow
