using Documenter, CoherentStructures

makedocs(
    format=:html,
    sitename="CoherentStructures.jl",
    pages = [
        "Home" => "index.md"
        "FEM-based methods" => "fem.md"
        "Dynamical Systems Utilities" => "util.md"
    ]
    )

makedocs()
