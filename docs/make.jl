using Documenter, CoherentStructures
makedocs(
    format=:html,
    sitename="CoherentStructures.jl",
    pages = [
        "Home" => "index.md"
        "FEM-based methods" => "fem.md"
        "Test Cases" => "examples.md"
        "API" => "api.md"
    ]
    )
