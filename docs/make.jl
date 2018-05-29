using Documenter, CoherentStructures
makedocs(
    format=:html,
    sitename="CoherentStructures.jl",
    pages = [
        "Home" => "index.md"
        "FEM-based methods" => "fem.md"
        "Geodesic vortices" => "elliptic.md"
        "Graph Laplacian-based methods" => "Laplace.md"
        "API" => "api.md"
    ]
    )
