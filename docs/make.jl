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
