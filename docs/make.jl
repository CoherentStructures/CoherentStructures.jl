using Documenter
using Literate
using CoherentStructures
using Dates
ENV["GKSwstype"] = "100"
using Plots # to not capture precompilation output

if !isdir("/tmp/natschil_misc")
    run(`bash -c 'echo $DEPLOY_KEY_2 | base64 --decode > /tmp/mykey'`)
    run(`chmod 0600 /tmp/mykey`)
    run(`ssh-agent bash -c 'ssh-add /tmp/mykey; git clone git@github.com:natschil/misc.git  /tmp/natschil_misc/'`)
end



# generate the example notebooks for the documentation
OUTPUT = joinpath(@__DIR__, "src/generated")

function mypreprocess(content,whatkind)
    global cont = content
    while true
        current_location = findfirst("DISPLAY_PLOT",content)
        if current_location == nothing
            break
        end
        @assert content[current_location[end] + 1] == '('
        closing_bracket = findfirst(")",content[current_location[end]:end])
        @assert closing_bracket != nothing
        closing_bracket  = closing_bracket .+ (current_location[end]-1)

        args = content[(current_location[end]+2): (closing_bracket[1]-1)]
        @assert findfirst(",",args) != nothing
        figname = args[1:(findfirst(",",args)[1]-1)]
        filename = args[(findfirst(",",args)[1]+1):end]

        @assert length(filename) > 1
        @assert length(figname) > 1

        if whatkind == :markdown
            linkloc="https://raw.githubusercontent.com/natschil/misc/master/autogen/" * filename *".png"
            inner_text = "# ![]($linkloc)"
        elseif whatkind == :notebook || whatkind == :julia_norun
            inner_text = "Plots.plot($figname)"
        elseif whatkind == :julia_run
            inner_text = "Plots.png($figname,\"/tmp/natschil_misc/autogen/$filename.png\")"
        end

        content = content[1:(current_location[1]-1)] * inner_text * content[(closing_bracket[1]+1):end]
    end
    if whatkind == :julia_run
        content = replace(content,"addprocs()" => "addprocs(exeflags=\"--project=docs/\")")
    end
    return content
end

preprocess_markdown = x->mypreprocess(x,:markdown)
preprocess_notebook = x->mypreprocess(x,:notebook)
preprocess_script = x->mypreprocess(x,:julia_norun)
preprocess_script2 = x->mypreprocess(x,:julia_run)

Literate.markdown(joinpath(@__DIR__, "..", "examples/bickley.jl"), OUTPUT;
    documenter=false,preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "examples/bickley.jl"), OUTPUT;
    execute=false,preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "examples/bickley.jl"), OUTPUT;
    preprocess=preprocess_script
    )
Literate.script(joinpath(@__DIR__, "..", "examples/bickley.jl"), "/tmp/";
    preprocess=preprocess_script2
    )

run(`julia --project=docs/ /tmp/bickley.jl`)


Literate.markdown(joinpath(@__DIR__, "..", "examples/ocean_flow.jl"), OUTPUT;
    documenter=false,preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "examples/ocean_flow.jl"), OUTPUT;
    execute=false,preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "examples/ocean_flow.jl"), OUTPUT;
    preprocess=preprocess_script
    )
Literate.script(joinpath(@__DIR__, "..", "examples/ocean_flow.jl"), "/tmp/";
    preprocess=preprocess_script2
    )

#cd() necessary because velocity data is in examples/
cd("../examples")
run(`julia --project=../docs/ /tmp/ocean_flow.jl`)
cd("../docs")


Literate.markdown(joinpath(@__DIR__, "..", "examples/rot_double_gyre.jl"), OUTPUT;
    documenter=false,preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "examples/rot_double_gyre.jl"), OUTPUT;
    execute=false,preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "examples/rot_double_gyre.jl"), OUTPUT;
    preprocess=preprocess_script
    )
Literate.script(joinpath(@__DIR__, "..", "examples/rot_double_gyre.jl"), "/tmp/";
    preprocess=preprocess_script2
    )

run(`julia --project=docs/ /tmp/rot_double_gyre.jl`)


Literate.markdown(joinpath(@__DIR__, "..", "examples/standard_map.jl"), OUTPUT;
    documenter=false,preprocess=preprocess_markdown)
Literate.notebook(joinpath(@__DIR__, "..", "examples/standard_map.jl"), OUTPUT;
    execute=false,preprocess=preprocess_notebook)
Literate.script(joinpath(@__DIR__, "..", "examples/standard_map.jl"), OUTPUT;
    preprocess=preprocess_script
    )
Literate.script(joinpath(@__DIR__, "..", "examples/standard_map.jl"), "/tmp/";
    preprocess=preprocess_script2
    )
1
run(`julia --project=docs/ /tmp/standard_map.jl`)



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
            "Rotating double gyre" => "generated/rot_double_gyre.md"
            "Standard map" => "generated/standard_map.md"
            "Bickley jet" => "generated/bickley.md"
            "Geostrophic ocean flow" => "generated/ocean_flow.md"
            ]
        "Basics" => "basics.md"
        "Methods" => [
            "FEM-based methods" => "fem.md"
            "Geodesic vortices" => "elliptic.md"
            "Graph Laplacian-based methods" => "Laplace.md"
            ]
    ]
    )

run(`git -C /tmp/natschil_misc/ add /tmp/natschil_misc/autogen`)
curdate = Dates.now()
run(`git -C /tmp/natschil_misc/ commit -m "Autogen $curdate"`)

run(`bash -c 'echo $DEPLOY_KEY_2 | base64 --decode > /tmp/mykey'`)
run(`chmod 0600 /tmp/mykey`)
run(`ssh-agent bash -c 'ssh-add /tmp/mykey; git -C /tmp/natschil_misc/ push'`)

deploydocs(
    repo = "github.com/CoherentStructures/CoherentStructures.jl.git"
)
