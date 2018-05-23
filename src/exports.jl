export
	#advection_diffusion.jl
	ADimplicitEulerStep,

	#ellipticLCS.jl
	singularity_location_detection,
	singularity_type_detection,
	detect_elliptic_region,
	set_Poincaré_section,
	compute_returning_orbit,
	compute_outermost_closed_orbit,
	ellipticLCS,

	#diffusion_operators.jl
	diff_op,
	sparse_diff_op_family
	sparse_diff_op,
	sparseaffinitykernel,
	α_normalize!,
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
	regularTriangularGrid,
	regularDelaunayGrid,
	regularP2TriangularGrid,
	regularP2DelaunayGrid,
	regularQuadrilateralGrid,
	regularP2QuadrilateralGrid,
	regularGrid,
	evaluate_function_from_dofvals,
	evaluate_function_from_nodevals,
	locatePoint,
	nodal_interpolation,
	undoBCS,
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

	#plotting.jl
	plot_u,
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

	#TO.jl
	nonAdaptiveTO,
	adaptiveTO,
	L2GalerkinTO,

	#util.jl
	tensor_invariants,
	dof2node,
	kmeansresult2LCS,
	interp_rhs,
	interp_rhs!,
	getH,

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
	interpolateVFPeriodic
