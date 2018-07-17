using Documenter, CoherentStructures
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
