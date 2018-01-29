export
	#PullbackTensors.jl
	flow,
	linearized_flow,
	invCGTensor,
	pullback_tensors,
	pullback_metric_tensor,
	pullback_diffusion_tensor,

	#velocityFields.jl
	rot_double_gyre2,
	transientGyresEqVari,
	bickleyJetEqVari,
	interpolateVF,
	oceanVF,
	interp_rhs,

	#GridFunctions.jl
	regularTriangularGrid,
	regularDelaunayGrid,
	regularP2TriangularGrid,
	regularP2DelaunayGrid,
	regularQuadrilateralGrid,
	regularP2QuadrilateralGrid,

	#plotting.jl
	plot_u,
	plot_spectrum,

	#FEMassembly.jl
	assembleStiffnessMatrix,
	assembleMassMatrix,

	#TO.jl
	nonAdaptiveTO,
	adaptiveTO,
	L2GalerkinTO
