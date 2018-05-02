using CoherentStructures

# after this, 'bickley' will reference a Dictionary of functions
# access it via the desired signature. e.g. F = bickley[:(dU, U, p, t)]
# for the right side of the equation of variation.
bickley = @makefields from stream begin
    stream = psi₀ + psi₁
    psi₀   = - U₀ * L₀ * tanh(y / L₀)
    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term

    re_sum_term =  Σ₁ + Σ₂ + Σ₃

    Σ₁  =  ε₁ * cos(k₁*(x - c₁*t))
    Σ₂  =  ε₂ * cos(k₂*(x - c₂*t))
    Σ₃  =  ε₃ * cos(k₃*(x - c₃*t))

    k₁ = 2/r₀      ; k₂ = 4/r₀    ; k₃ = 6/r₀

    ε₁ = 0.0075    ; ε₂ = 0.15    ; ε₃ = 0.3
    c₁ = 0.1446U₀  ; c₂ = 0.205U₀ ; c₃ = 0.461U₀

    U₀ = 62.66e-6  ; L₀ = 1770e-3 ; r₀ = 6371e-3
end

LL = [0.0,-3.0]; UR=[6.371π,3.0]
ctx = regularTriangularGrid((100,30),LL,UR,quadrature_order=1)
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && (abs((p1[1] - p2[1])%6.371π) < 1e-10)
bdata = CoherentStructures.boundaryData(ctx,predicate,[])

bickley_field = bickley[:(du,u,p,t)]
cgfun = (x -> mean(pullback_diffusion_tensor(bickley_field, x,linspace(0.0,40*3600*24,81),
     1.e-8,tolerance=1.e-5)))

@time K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
@time M = assembleMassMatrix(ctx,bdata=bdata)
@time λ, v = eigs(K,M,which=:SM, nev= 10)

plot_u(ctx,v[:,1],bdata=bdata)

using Clustering
n_partition = 7
res = kmeans(v[:,2:n_partition]',n_partition)
u = kmeansresult2LCS(res)

sum([u[:,i]*i for i in 1:n_partition])
plot_u(ctx, sum([u[:,i]*i for i in 1:n_partition]),200,200,color=:viridis,bdata=bdata) 
