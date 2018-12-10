using Random

a = 0.971635
f(a,x) = (mod(x[1] + x[2] + a*sin(x[1]), 2π),
          mod(x[2] + a*sin(x[1]), 2π))

X = []
for i in 1:50
    Random.seed!(i)
    x = 2π*rand(2)
    for i in 1:500
        x = f(a,x)
        push!(X,x)
    end
end

using Plots
scatter([x[1] for x in X], [x[2] for x in X], markersize=1)

using Arpack, CoherentStructures, Tensors

Df(a,x) = Tensor{2,2}((1.0+a*cos(x[1]), a*cos(x[1]), 1.0, 1.0))

n, ll, ur = 100, [0.0,0.0], [2π,2π]               # grid size, domain corners
ctx = regularTriangularGrid((n,n), ll, ur)
pred(x,y) = peuclidean(x[1], y[1], 2π) < 1e-9 &&
            peuclidean(x[2], y[2], 2π) < 1e-9
bd = boundaryData(ctx, pred)        # periodic boundary

I = one(Tensor{2,2})               # identity matrix
Df2(x) = Df(a,f(a,x))⋅Df(a,x)       # consider 2. iterate
cg(x) = 0.5*(I + dott(inv(Df2(x))))

K = assembleStiffnessMatrix(ctx, cg, bdata=bd)
M = assembleMassMatrix(ctx, bdata=bd)
λ, v = eigs(K, M, which=:SM)

using Printf
title = [ @sprintf("\\lambda = %.3f",λ[i]) for i = 1:4 ]
p = [ plot_u(ctx, v[:,i], bdata=bd, title=title[i],
             clim=(-0.25,0.25), cb=false) for i in 1:4 ]
plot(p...)
