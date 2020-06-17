using Documenter
using Literate
using CoherentStructures
using StreamMacros
using Dates
ENV["GKSwstype"] = "100"
using Plots # to not capture precompilation output

if !isdir("/tmp/natschil_misc")
    if ("DEPLOY_KEY_2" ∈ keys(ENV))
        run(`bash -c 'echo $DEPLOY_KEY_2 | tr -d " " | base64 --decode > /tmp/mykey'`)
        run(`chmod 0600 /tmp/mykey`)
        run(`git config --global user.email "autdeploy@example.com"`)
        run(`git config --global user.name "Automatic Deploy"`)
        run(`ssh-agent bash -c 'ssh-add /tmp/mykey; git clone git@github.com:natschil/misc.git  /tmp/natschil_misc/'`)
    else
        mkdir("/tmp/natschil_misc")
        mkdir("/tmp/natschil_misc/autogen")
    end
end

# generate the example notebooks for the documentation
OUTPUT = joinpath(@__DIR__, "src/generated")

function mypreprocess(content, whatkind)
    global cont = content
    while true
        current_location = findfirst("DISPLAY_PLOT",content)
        if current_location === nothing
            break
        end
        @assert content[current_location[end] + 1] == '('
        closing_bracket = findfirst(")",content[current_location[end]:end])
        @assert closing_bracket !== nothing
        closing_bracket  = closing_bracket .+ (current_location[end]-1)

        args = content[(current_location[end]+2):(closing_bracket[1]-1)]
        @assert findfirst(",",args) !== nothing
        figname = args[1:(findfirst(",",args)[1]-1)]
        file_name = args[(findfirst(",",args)[1]+2):end]

        @assert length(file_name) > 1
        @assert length(figname) > 1

        if whatkind === :markdown
            linkloc="https://raw.githubusercontent.com/natschil/misc/master/autogen/" * file_name *".png"
            inner_text = "# ![]($linkloc)"
        elseif whatkind === :notebook || whatkind === :julia_norun
            inner_text = "Plots.plot($figname)"
        elseif whatkind === :julia_run
            inner_text = "Plots.png($figname,\"/tmp/natschil_misc/autogen/$file_name.png\")"
        end

        content = content[1:(current_location[1]-1)] * inner_text * content[(closing_bracket[1]+1):end]
    end
    if whatkind === :julia_run
        content = replace(content, "addprocs()" => "addprocs(exeflags=\"--project=docs/\")")
        content = replace(content, "OCEAN_FLOW_FILE" => "\"docs/examples/Ocean_geostrophic_velocity.jld2\"")
    else
        content = replace(content, "OCEAN_FLOW_FILE" => "\"Ocean_geostrophic_velocity.jld2\"")
    end
    return content
end

preprocess_markdown = x -> mypreprocess(x, :markdown)
preprocess_notebook = x -> mypreprocess(x, :notebook)
preprocess_script = x -> mypreprocess(x, :julia_norun)
preprocess_script2 = x -> mypreprocess(x, :julia_run)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/bickley.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/bickley.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/bickley.jl"), OUTPUT;
    preprocess=preprocess_script)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/bickley.jl"), "/tmp/";
    preprocess=preprocess_script2)

run(`julia --project=docs/ /tmp/bickley.jl`)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/trajectories.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/trajectories.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/trajectories.jl"), OUTPUT;
    preprocess=preprocess_script)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/trajectories.jl"), "/tmp/";
    preprocess=preprocess_script2)

run(`julia --project=docs/ /tmp/trajectories.jl`)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/ocean_flow.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/ocean_flow.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/ocean_flow.jl"), OUTPUT;
    preprocess=preprocess_script)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/ocean_flow.jl"), "/tmp/";
    preprocess=preprocess_script2)

run(`julia --project=docs/ /tmp/ocean_flow.jl`)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/rot_double_gyre.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/rot_double_gyre.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/rot_double_gyre.jl"), OUTPUT;
    preprocess=preprocess_script)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/rot_double_gyre.jl"), "/tmp/";
    preprocess=preprocess_script2)

run(`julia --project=docs/ /tmp/rot_double_gyre.jl`)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/standard_map.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/standard_map.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/standard_map.jl"), OUTPUT;
    preprocess=preprocess_script)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/standard_map.jl"), "/tmp/";
    preprocess=preprocess_script2)

run(`julia --project=docs/ /tmp/standard_map.jl`)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/diffbarriers.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/diffbarriers.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/diffbarriers.jl"), OUTPUT;
    preprocess=preprocess_script)

Literate.markdown(joinpath(@__DIR__, "..", "docs/examples/turbulence.jl"), OUTPUT;
    documenter=false, preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "docs/examples/turbulence.jl"), OUTPUT;
    execute=false, preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "docs/examples/turbulence.jl"), OUTPUT;
    preprocess=preprocess_script)

# replace links (if any)
# travis_tag = get(ENV, "TRAVIS_TAG", "")
# folder = isempty(travis_tag) ? "latest" : travis_tag
# url = "https://nbviewer.jupyter.org/github/CoherentStructures/CoherentStructures.jl/blob/gh-pages/$(folder)/"
# if get(ENV, "HAS_JOSH_K_SEAL_OF_APPROVAL", "") == "true"
#     str = read(joinpath(@__DIR__, "src/file_name.md"), String)
#     str = replace(str, "[notebook.ipynb](generated/notebook.ipynb)." => "[notebook.ipynb]($(url)generated/notebook.ipynb).")
#     write(joinpath(@__DIR__, "src/file_name.md"), str)
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
            "Rotating double gyre" => "generated/rot_double_gyre.md"
            "Standard map" => "generated/standard_map.md"
            "Bickley jet" => "generated/bickley.md"
            "Geostrophic ocean flow" => "generated/ocean_flow.md"
            "Working with trajectories" => "generated/trajectories.md"
            ]
        "Related publications" => [
            "Overview" => "examples_pub_overview.md"
            "Material diffusion barriers" => "generated/diffbarriers.md"
            "Diffusion barriers in turbulence" => "generated/turbulence.md"
        ]
        "Basics" => "basics.md"
        "Methods" => [
            "FEM-based methods" => "fem.md"
            "Geodesic vortices" => "elliptic.md"
            "Graph Laplacian-based methods" => "Laplace.md"
            ]
            "Miscellaneous" => [
                "Creating animations" => "videos.md"
            ]
    ]
    )

if "DEPLOY_KEY_2" ∈ keys(ENV)
    GREF = ENV["GITHUB_REF"]
    if ("GITHUB_REF" ∈ keys(ENV)) &&  (ENV["GITHUB_REF"] ∈ ["master", "refs/heads/master"])
        run(`git -C /tmp/natschil_misc/ add /tmp/natschil_misc/autogen`)
        curdate = Dates.now()
        run(`git -C /tmp/natschil_misc/ commit -m "Autogen $curdate"`)

        #run(`bash -c 'echo $DEPLOY_KEY_2 | tr -d " " | base64 --decode > /tmp/mykey'`)
        #run(`chmod 0600 /tmp/mykey`)
        run(`ssh-agent bash -c 'ssh-add /tmp/mykey; git -C /tmp/natschil_misc/ push'`)
    end

    deploydocs(
        repo = "github.com/CoherentStructures/CoherentStructures.jl.git",
        push_preview=true,
    )
end
