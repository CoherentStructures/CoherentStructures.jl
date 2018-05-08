# Geodesic elliptic material vortices

```@meta
CurrentModule = CoherentStructures
```
The following functions implement an LCS methodology developed in the following papers:
   * [Haller & Beron-Vera, 2012](https://dx.doi.org/10.1016/j.physd.2012.06.012)
   * [Haller & Beron-Vera, 2013](https://dx.doi.org/10.1017/jfm.2013.391)
   * [Karrasch, Huhn, and Haller, 2015](https://dx.doi.org/10.1098/rspa.2014.0639)
The present code is structurally inspired--albeit partially significantly
improved--by Alireza Hadjighasem's MATLAB implementation [github project](https://github.com/Hadjighasem/Elliptic_LCS_2D),
which was written in the context of the [SIAM Review paper](https://doi.org/10.1137/140983665). Depending on the indefinite metric
tensor field used, the functions below yield the following types of coherent structures:
   * black-hole/Lagrangian coherent vortices ([Haller & Beron-Vera, 2012](https://doi.org/10.1017/jfm.2013.391))
   * elliptic objective Eulerian coherent structures (OECSs) ([Serra & Haller, 2016](https://dx.doi.org/10.1063/1.4951720))
   * material diffusive transport barriers (Haller, Karrasch, and Kogelbauer, 2018)
The general procedure is the following. Assume $T$ is the symmetric tensor field of interest, say, (i) the Cauchy-Green strain tensor field $C$, (ii) the rate-of-strain tensor field $S$, or (iii) the averaged diffusion-weighted Cauchy-Green tensor field $\bar{C}_D$; cf. the references above. Denote by $0<\lambda_1\leq\lambda_2$ the eigenvalue and by $\xi_1$ and $\xi_2$ the corresponding eigenvector fields of $T$. Then the direction fields of interest are given by

$\eta_{\lambda}^{\pm} := \sqrt{\frac{\lambda_2 - \lambda}{\lambda_2-\lambda_1}}\xi_1 \pm \sqrt{\frac{\lambda - \lambda_1}{\lambda_2-\lambda_1}}\xi_2.$

Tensor singularities are defined as points at which $\lambda_2=\lambda_1$, i.e., at which the two characteristic directions $\xi_1$ and $\xi_2$ are not well-defined.
Then, the algorithm put forward in [Karrasch et al., 2015](https://dx.doi.org/10.1098/rspa.2014.0639) consists of the following steps:
   1. locate singularities of the tensor field $T$;
   2. determine the type of the singularities (only non-degenerate types like wedges and trisectors are detected);
   3. look for wedge pairs that are reasonably isolated from other singularities;
   4. place an eastwards oriented Poincaré section at the pair center;
   5. for each point on the discretized Poincaré section, scan through the parameter space such that the corresponding η-orbit closes at that point;
   6. for each Poincaré section, take the outermost closed orbit as the coherent vortex barrier (if there exist any).

## Example

```@example 3
# addprocs()

# @everywhere
begin
    using CoherentStructures
    import Tensors, OrdinaryDiffEq
    ########################## integration set up ############################
    const q = 81
    const tspan = collect(linspace(0.,3456000.,q))
    const ny = 120
    const nx = div(ny*24,6)
    const N = nx*ny
    const xmin, xmax, ymin, ymax = 0.0 - 2.0, 6.371π + 2.0, -3.0, 3.0
    xspan, yspan = linspace(xmin,xmax,nx), linspace(ymin,ymax,ny)
    P = vcat.(xspan,yspan')
    const δ = 1.e-6
    const DiffTensor = Tensors.SymmetricTensor{2,2}([2., 0., 1/2])
    mCG_tensor = u -> av_weighted_CG_tensor(bickleyJet,u,tspan,δ,D = DiffTensor,tolerance=1e-6,solver=OrdinaryDiffEq.Tsit5())
end

C̅ = map(mCG_tensor,P)

LCSparams = (.1, 0.5, 0.04, 0.5, 1.8, 60)
vals, signs, orbits = ellipticLCS(C̅,xspan,yspan,LCSparams);
```
The result is visualized as follows:
```@example 3
import Plots
λ₁, λ₂, ξ₁, ξ₂, traceT, detT = tensor_invariants(C̅)
l₁ = min.(λ₁,quantile(λ₁[:],0.999))
l₁ = max.(λ₁,1e-2)
l₂ = min.(λ₂,quantile(λ₂[:],0.995))
begin
    fig = Plots.heatmap(xspan,yspan,log10.(l₁.+l₂)',aspect_ratio=1,color=:viridis,
            title="DBS-field and transport barriers",xlims=(0., 6.371π),ylims=(-3., 3.),leg=true)
    for i in eachindex(orbits)
        Plots.plot!(orbits[i][1,:],orbits[i][2,:],w=3,label="T = $(round(vals[i],2))")
    end
end
Plots.plot(fig)
```

## Function documentation

The fully automated meta-function is the following:

```@docs
ellipticLCS
```

Essentially, it calls sequentially the following functions.

```@docs
singularity_location_detection
```

```@docs
singularity_type_detection
```

```@docs
detect_elliptic_region
```

```@docs
set_Poincaré_section
```

```@docs
compute_outermost_closed_orbit
```
