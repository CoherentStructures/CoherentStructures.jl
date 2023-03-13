# # Coupled multiscale modeling of cancellous bone
#
# ## Introduction
#
# Multiscale modeling of bone has important applications in the early detection, 
# diagnosis and treatment of bone diseases. The most common one is osteoporosis. 
# This bone disease affects especially the elderly population and manifests itself
# by reducing the volume percentage of cortical bone, one of the two main components
# of mammal bone besides bone marrow. Simulating the behavior of bone can give important
# insights into the physical interactions of the system. 
#
# ## Material model
#
# The model consists of two scales (macroscopic and microscopic scale) and uses the
# finite element square method ($\text{FE}^2$) to connect the scales. Starting point for
# the microscopic scale is the following thermodynamic energy functional, which considers
# the phases cortical bone (subscript b) and bone marrow (subscript m):
# ```math
# \Pi = \int_{\Omega_b} \Psi_{b}(\varepsilon,\textbf{E},\textbf{B}) \text{d}V + \int_{\Omega_m} \Psi_{m}(\varepsilon,\varepsilon^i,\textbf{E},\textbf{B}) + \mathcal{C} \text{d}V + \int_{\Omega_m} \int_{t} \Delta(\dot{\varepsilon}^i,\dot{\textbf{A}}) \text{d}t \text{d}V + \int_{\Omega} \Psi_{g}(\nabla \cdot \textbf{A}) \text{d}V - W_{\text{ext}} \; \text{.}
# ```
# The functional consists of the energy densities $\Psi_{b}$ and $\Psi_{m}$
# of both phases, a volume constraint $\mathcal{C}$, dissipation and gauge functionals
# ($\Delta$ and $\Psi_{g}$) and the potential of the generalized external forces
# $W_{\text{ext}}$. The energy densities for both phases are
# ```math
# \Psi_b = \frac{1}{2} (\varepsilon \cdot C_b \cdot \varepsilon - \textbf{E} \cdot \xi_b \cdot \textbf{E} + \textbf{B} \cdot \mu_b^{-1} \cdot \textbf{B}) - \textbf{e}_b \cdot \varepsilon \cdot \textbf{E} \quad \text{and}
# ```
# ```math
# \Psi_m = \frac{1}{2} ((\varepsilon-\varepsilon^i) \cdot C_m \cdot (\varepsilon-\varepsilon^i) - \textbf{E} \cdot \xi_m \cdot \textbf{E} + \textbf{B} \cdot \mu_\mathrm{m}^{-1} \cdot \textbf{B}) \; \text{,}
# ```
# consisting of quadratic energies for mechanical, electric and magnetic effects.
# A piezoelectric energy term is included in the cortical bone phase. For the bone
# marrow phase, an inelastic strain $\varepsilon^i$ is introduced. Here, $C$ is 
# the mechanical stiffness tensor, $\xi$ is the permittivity tensor, $\mu^{-1}$ is
# the inverse permeability tensor and $\textbf{e}_b$ is the piezoelectric tensor. The resulting
# strong form of the problem is
# ```math
# \nabla \cdot \sigma + \textbf{f} = 0 \quad \text{in} \; \Omega
# ```
# ```math
#  \sigma \cdot \textbf{n} = \textbf{t} \quad \text{on} \; \partial \Omega
# ```
# ```math
#  \nabla \cdot \textbf{D} = q_v \quad \text{in} \; \Omega
# ```
# ```math
#  \textbf{D} \cdot \textbf{n} = -q_s \quad \text{on} \; \partial \Omega
# ```
# ```math
#  \nabla \times \textbf{H} = \dot{\textbf{D}} + \textbf{J} + \gamma \nabla(\nabla \cdot \textbf{A}) + \textbf{j}_v \quad \text{in} \; \Omega
# ```
# ```math
# \textbf{H} \times \textbf{n} = \textbf{j}_s - \gamma (\nabla \cdot \textbf{A}) \textbf{n} \quad \text{on} \; \partial \Omega \; \text{,}
# ```
# recovering the mechanical equilibrium condition, two Maxwell equations (the other
# two are implicitly fulfilled due to the choice of potentials) and boundary conditions, 
# including the gauge. By using variational calculus, the problem can be transformed into
# the weak form and the finite element method can be applied.
#
# ## Usage of the CoherentStructures.jl package
#
# On the microscopic scale, periodic boundary conditions (PBCs) are desired to obtain good
# results independent of the relative position of inclusions in the used representative
# volume element (RVE). The computer program used for the simulations is written in Julia
# and uses mainly the [Ferrite.jl]( https://github.com/Ferrite-FEM/Ferrite.jl) package. During
# the development in 2020, the past version of Ferrite.jl (called JuAFEM.jl back then) did not
# support periodic boundary conditions inherently. The ‘GridContext‘ struct of
# ‘CoherentStructures.jl‘ expands the ‘Grid‘-type of Ferrite and allows to find nodes on
# opposite sides (stored via ‘BCTable‘), which is necessary to apply periodic boundary conditions.
# As the model required seven degrees of freedom (DoFs) in each node, it is necessary to split
# the resulting solution vector after solving the linear system up into each individual DoF and
# apply the function ‘undoBCS‘ to receive the complete solution vector. Since 2022, Ferrite
# directly supports PBCs, also allowing more complicated cases directly in the package.
# For a detailed explanation, see the following forum post: 
# [Ferrite-periodic-boundary-conditions](https://discourse.julialang.org/t/ferrite-jl-periodic-boundary-conditions/39387).
#
# ## Code availability:
#
# The code can be found on [Github](https://github.com/blaszm/bone_fe2_c).
#
# ## Related publications:
#
# 1.	[Multiscale modeling of cancellous bone](https://link.springer.com/article/10.1007/s10237-021-01525-6)
# 2.	[Inverse modeling using artificial neural networks](https://onlinelibrary.wiley.com/doi/full/10.1002/zamm.202100541)
