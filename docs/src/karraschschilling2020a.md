# Material barriers in Turbulent flow (Karrasch & Schilling 2020)

We need to use several packages. Below shows to install specific versions that should work. You can also try it without pinnin the specific versions. You'll only need to do this the first time you run this example. Due to julia-specific factors, all commands used here will take much longer the first time they are run.
```julia
import Pkg
Pkg.add("FourierFlows")
Pkg.add("GeophysicalFlows")
Pkg.add("Plots")
Pkg.add(Pkg.PackageSpec(url="https://github.com/KristofferC/JuAFEM.jl.git"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/CoherentStructures/CoherentStructures.jl.git"), version="0.2")
Pkg.pin(Pkg.PackageSpec(name="FourierFlows", version="0.4.1"))
Pkg.pin(Pkg.PackageSpec(name="GeophysicalFlows", version="0.3.3"))
```
