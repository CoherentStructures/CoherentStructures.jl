export
	#pullbacktensors.jl
	flow,
	ad_flow,
	linearized_flow,
	mean_diff_tensor,
	pullback_tensors,
	pullback_metric_tensor,
	pullback_diffusion_tensor,
	pullback_diffusion_tensor_function,
	pullback_SDE_diffusion_tensor,

	#velocityfields.jl
	rot_double_gyre!,
	rot_double_gyre,
	transientGyresEqVari!,
	transientGyresEqVari,
	bickleyJet!,
	bickleyJet,
	bickleyJetEqVari!,
	bickleyJetEqVari,
	interpolateVF,
	interpolateVFPeriodic,

	#ellipticLCS.jl
	singularity_location_detection,
	singularity_type_detection,
	detect_elliptic_region,
	set_Poincaré_section,
	compute_returning_orbit,
	compute_outermost_closed_orbit,
	ellipticLCS,

	#gridfunctions.jl
	regularTriangularGrid,
	regularDelaunayGrid,
	regularP2TriangularGrid,
	regularP2DelaunayGrid,
	regularQuadrilateralGrid,
	regularP2QuadrilateralGrid,
	regularGrid,
	evaluate_function_from_dofvals,
	evaluate_function_from_u,
	locatePoint,
	nodal_interpolation,
	upsample2DBCS,
	getH,

	#plotting.jl
	plot_u,
	plot_spectrum,
	plot_real_spectrum,
	plot_u_eulerian,
	plot_ftle,

	#FEMassembly.jl
	assembleStiffnessMatrix,
	assembleMassMatrix,

	#TO.jl
	nonAdaptiveTO,
	adaptiveTO,
	L2GalerkinTO,

	#numericalExperiments.jl
	makeOceanFlowTestCase,
	makeDoubleGyreTestCase,
	experimentResult,
	runExperiment!,
	plotExperiment,

	#util.jl
	tensor_invariants,
	dof2U,
	kmeansresult2LCS,
	interp_rhs,
	interp_rhs!,

	#advection_diffusion.jl
	ADimplicitEulerStep,

	#field_from_hamiltonian.jl
	@makefields
