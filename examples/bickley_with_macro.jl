using CoherentStructures, OrdinaryDiffEq, Arpack

nx, ny = 100, 31
LL = [0.0, -3.0]; UR=[6.371π, 3.0]
ctx, _ = regularTriangularGrid((nx, ny), LL, UR, quadrature_order=2)
predicate = (x,y) -> (abs(x[2] - y[2]) < 1e-10) && (peuclidean(x[1],y[1],6.371π) < 1e-10)
bdata = CoherentStructures.BoundaryData(ctx, predicate, [])

cgfun = (x -> mean_diff_tensor(bickleyJet, x, range(0.0,stop=40*3600*24,length=81),
     1.e-8, tolerance=1.e-6, solver=Tsit5()))

@time K = assembleStiffnessMatrix(ctx, cgfun, bdata=bdata)
@time M = assembleMassMatrix(ctx, bdata=bdata)
@time λ, v = Arpack.eigs(K, M, which=:SM, nev= 10)

plot_u(ctx, v[:,2], 4nx, 4ny, bdata=bdata)
plot_real_spectrum(λ)

using Clustering
n_partition = 7
res = kmeans(permutedims(v[:,2:n_partition]),n_partition)
u = kmeansresult2LCS(res)

sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx, sum([u[:,i]*i for i in 1:n_partition]),200,200,color=:viridis,bdata=bdata)
