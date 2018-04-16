export
	#PullbackTensors.jl
	flow,
	ad_flow,
	linearized_flow,
	invCGTensor,
	pullback_tensors,
	pullback_metric_tensor,
	pullback_diffusion_tensor,
	pullback_diffusion_tensor_function,
	pullback_SDE_diffusion_tensor,

	#velocityFields.jl
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

	#GridFunctions.jl
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
	dof2U,
	kmeansresult2LCS,
	interp_rhs,
	interp_rhs!,

	#advection_diffusion.jl
	ADimplicitEulerStep,

	#field_from_hamiltonian.jl
	@makefields
