#include("../src/field_from_hamiltonian.jl")
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

#using ServerPlots
#using Plots;pyplot()
#quiv = let f = bickley[:(x,y,t)]; (x,y)-> 10e2*f(x,y,0.0) end
#xs = linspace(0,1,20)
#quiver(xs, xs', quiver = quiv)


xmin = 0.0; xmax = 6.371π; ymin = -3.; ymax = 3.0
ctx = regularTriangularGrid((100,30),[xmin,ymin],[xmax,ymax],quadrature_order=1)
field = bickley[:(du,u,p,t)]

p = (62.66e-6, 1770e-3, 9.058543015644972e-6, 1.28453e-5, 2.888626e-5,
         0.0075, 0.15, 0.3, 0.31392246115209543, 0.6278449223041909, 0.9417673834562862)
function difference(x,y,t)
    du = [0.0,0.0]
    bickleyJet!(du,[x,y],p,t)
    du2 = [0.0,0.0]
    field(du2,[x,y],p,t)
    return norm(du-du2)
end
Plots.heatmap(
    linspace(xmin,xmax,200),linspace(ymin,ymax,200), (x,y) -> difference(x,y,40*3600*24.0)
    )

cgfun = (x -> mean(pullback_diffusion_tensor(field, x,linspace(0.0,40*3600*24,81),
     1.e-8,tolerance=1.e-4)))

#cgfun = (x -> mean(pullback_diffusion_tensor(bickleyJet!, x,linspace(0.0,40*3600*24,81),
#     1.e-8,tolerance=1.e-4,p=p)))
predicate = (p1,p2) -> abs(p1[2] - p2[2]) < 1e-10 && (abs((p1[1] - p2[1])%6.371π) < 1e-10)
bdata = CoherentStructures.boundaryData(ctx,predicate,[])
@time K = assembleStiffnessMatrix(ctx,cgfun,bdata=bdata)
@time M = assembleMassMatrix(ctx,bdata=bdata)
@time λ, v = eigs(K,M,which=:SM, nev= 10)
plot_u(ctx,v[:,4],bdata=bdata)

for i in 1:10
    title = "Eigenvector with eigenvalue $(λ[i])"
    Plots.display(plot_u(ctx,v[:,i],title=title,bdata=bdata))
end
