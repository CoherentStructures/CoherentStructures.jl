Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625
ENV["GKSwstype"] = "100"

using Documenter, CoherentStructures

#Before running this, make sure that Distances,JLD2, Printf, Random, OrdinaryDiffEq and Clustering packages are
#installed and added to your current environment (]add )
makedocs(
    format=:html,
    sitename="CoherentStructures.jl",
    pages = Any[
        "Home" => "index.md"
        "Examples" => "examples.md"
        "Basics" => "basics.md"
        "Methods" => [
            "FEM-based methods" => "fem.md"
            "Geodesic vortices" => "elliptic.md"
            "Graph Laplacian-based methods" => "Laplace.md"
            ]
    ]
    )

deploydocs(
    repo = "github.com/CoherentStructures/CoherentStructures.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
