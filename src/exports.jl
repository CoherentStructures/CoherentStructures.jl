export
	#PullbackTensors.jl
	flow2D,
	LinearizedFlowMap,
	invCGTensor,

	#velocityFields.jl
	rot_double_gyre2,
	transientGyresEqVari,
	bickleyJetEqVari,
	interpolateOceanFlow,
	oceanVF,

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


