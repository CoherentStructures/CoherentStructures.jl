using Documenter
using Literate
using CoherentStructures
using Plots # to not capture precompilation output

ENV["GKSwstype"] = "100"

# generate the example notebooks for the documentation
OUTPUT = joinpath(@__DIR__, "..", "src/generated")

Literate.markdown(joinpath(@__DIR__, "..", "src", "src/bickley.jl"), OUTPUT)
Literate.notebook(joinpath(@__DIR__, "..", "src", "src/bickley.jl"), OUTPUT)
Literate.script(joinpath(@__DIR__, "..", "src", "src/bickley.jl"), OUTPUT)

Literate.markdown(joinpath(@__DIR__, "..", "src", "src/ocean_flow.jl"), OUTPUT)
Literate.notebook(joinpath(@__DIR__, "..", "src", "src/ocean_flow.jl"), OUTPUT)
Literate.script(joinpath(@__DIR__, "..", "src", "src/ocean_flow.jl"), OUTPUT)

Literate.markdown(joinpath(@__DIR__, "..", "src", "src/rot_double_gyre.jl"), OUTPUT)
Literate.notebook(joinpath(@__DIR__, "..", "src", "src/rot_double_gyre.jl"), OUTPUT)
Literate.script(joinpath(@__DIR__, "..", "src", "src/rot_double_gyre.jl"), OUTPUT)

Literate.markdown(joinpath(@__DIR__, "..", "src", "src/standard_map.jl"), OUTPUT)
Literate.notebook(joinpath(@__DIR__, "..", "src", "src/standard_map.jl"), OUTPUT)
Literate.script(joinpath(@__DIR__, "..", "src", "src/standard_map.jl"), OUTPUT)

# replace links (if any)
# travis_tag = get(ENV, "TRAVIS_TAG", "")
# folder = isempty(travis_tag) ? "latest" : travis_tag
# url = "https://nbviewer.jupyter.org/github/CoherentStructures/CoherentStructures.jl/blob/gh-pages/$(folder)/"
# if get(ENV, "HAS_JOSH_K_SEAL_OF_APPROVAL", "") == "true"
#     str = read(joinpath(@__DIR__, "src/filename.md"), String)
#     str = replace(str, "[notebook.ipynb](generated/notebook.ipynb)." => "[notebook.ipynb]($(url)generated/notebook.ipynb).")
#     write(joinpath(@__DIR__, "src/filename.md"), str)
# end


# Before running this, make sure that Plots, Tensors, Distances, JLD2,
# OrdinaryDiffEq and Clustering packages are installed and added to your
# current environment (]add )
makedocs(
    format = Documenter.HTML(),
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
    target = "build",
    deps = nothing,
    make = nothing,
)
