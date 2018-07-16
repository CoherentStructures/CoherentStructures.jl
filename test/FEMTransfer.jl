#(c) 2018 Nathanael Schilling
#Plots for (more general) L^2-Galerkin approximations of the transfer-operator

using CoherentStructures

begin
    #ctx = regularTriangularGrid((64,64), [0.0,0.0],[1.,1.],quadrature_order=5)
    ctx = regularP2QuadrilateralGrid((16,16), [0.0,0.0],[1.,1.],quadrature_order=6)
    pred  = (x,y) -> (peuclidean(x,y,[2π,2π]) < 1e-9)
    bdata = boundaryData(ctx,pred) #Periodic boundary
end

length(ctx.quadrature_points)

function sm(u)
    ϵ = 1
    ρ = 0.3
    u2 = u[2]+0.5
    return StaticArrays.SVector{2,Float64}((
        mod(u[1] + ϵ*u2 + ϵ^2*ρ*sin(2π*u[1]), 1),
        mod(u[2] + ϵ*ρ*sin(2π*u[1]) , 1)
        ))
end

function sminv(x)
    ϵ = 1
    ρ = 0.3
    return StaticArrays.SVector{2,Float64}((
        mod(x[1] - x[2] - 0.5, 1),
        mod(x[2] - ρ*sin(2π*(x[1] - x[2] - 0.5)) , 1)
        ))
end



ALPHA = applyBCS(ctx,L2GalerkinTOFromInverse(ctx,sminv),bdata)
#ALPHA = applyBCS(ctx,L2GalerkinTOFromInverse(ctx,sm),bdata)
ALPHA2 = applyBCS(ctx,L2GalerkinTO(ctx,sm),bdata)
M = assembleMassMatrix(ctx,bdata=bdata)

lambda, V = eigs(0.5*(ALPHA + ALPHA2'),M, which=:LR, nev=100)
lambda, V = eigs(0.5*(ALPHA + ALPHA'),M, which=:LR, nev=100)
lambda, V = eigs(0.5*(ALPHA + ALPHA'),M, which=:LR, nev=100)
plot_spectrum(lambda)
lambda[1:5]
plot_u(ctx,real.(V[:,1])/maximum(abs(V[:,1])),bdata=bdata,color=:rainbow)

M\((ALPHA + ALPHA2'*ones(size(ALPHA)[1]))
norm( M\( 0.5*(ALPHA + ALPHA2')*ones(size(ALPHA)[1])) - ones(size(ALPHA)[1]))
norm( M\( (ALPHA)*ones(size(ALPHA)[1])) - ones(size(ALPHA)[1]))
norm( M\( (ALPHA)*ones(size(ALPHA)[1])) - ones(size(ALPHA)[1]))

Plots.plot([ plot_u(ctx,real.(V[:,i]/maximum(real.(V[:,i]))),bdata=bdata,colorbar=false,@sprintf("%.1e" ,abs(1-lambda[i])),clim=[-1,1],200,200,color=:rainbow) for i in 1:10]...,
    margin=-10.0Plots.px)
Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/SMTOevecsP2.pdf")


ALPHA2 = applyBCS(ctx, nonAdaptiveTO(ctx,standardMapInv),bdata)

lambda, V= eigs(ALPHA2,which=:LM,nev=100)

Plots.plot([ plot_u(ctx,real.(V[:,i]),bdata=bdata,colorbar=false,clim=[-1,1],100,100) for i in 1:56]...,
    margin=-10.0Plots.px)

Plots.pdf("/home/nathanael/Documents/TUM/topmath/plots/ALPAHevecsP1.pdf")
