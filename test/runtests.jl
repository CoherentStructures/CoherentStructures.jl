using Test
using CoherentStructures

include("fem_tests.jl")

include("test_pullback_tensors.jl")

include("test_util.jl")

include("test_elliptic.jl")

include("test_dyn_metric.jl")

include("test_diff_ops.jl")

include("test_odesolvers.jl")
