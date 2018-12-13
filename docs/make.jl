if Base.HOME_PROJECT[] != nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end
ENV["GKSwstype"] = "100"

using Documenter, CoherentStructures

# Before running this, make sure that Plots, Tensors, Distances, JLD2, Printf,
# Random, OrdinaryDiffEq and Clustering packages are installed and added to your
# current environment (]add )
makedocs(
    format = :html,
    sitename="CoherentStructures.jl",
    pages = Any[
        "Home" => "index.md"
        "Examples" => [
            "Rotating Double Gyre" => "rot_double_gyre.md"
            "Geostrophic Ocean Flow" => "ocean_flow.md"
            "Bickley Jet" => "bickley.md"
            "Standard Map" => "standard_map.md"
            ]
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
    branch = "gh-pages",
    devbranch = "master",
    devurl ="dev",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
