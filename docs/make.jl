using Documenter
using Literate
using CoherentStructures
using Dates
ENV["GKSwstype"] = "100"
using Plots # to not capture precompilation output

ENV["DEPLOY_KEY_2"] = "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFB
QUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUNGd0FBQUFkemMyZ3RjbgpOaEFBQUFB
d0VBQVFBQUFnRUF4ZUFMeGhEZ1VMV25yVGEvZzZSbkgyY1EyMmhMWm5Sc0d6QS9xUWx3d3l4c2F0
SGZJbWFKCml2U3FrRUJrcE5MeFdTdFV5cXg0WjZ1VCthUWx0QXVMa0hLdDlyLzJudmE0WnhFSS9E
OGRMd2J6cDl6Y2RRNDRpdThoOEYKbUtYWENQdVl4TFhpUHBQZEJ4T2lSS0dZdzRjbDRQK2JsRXRR
SksvL1JzMmExQkhyZDVheGJMK0ViMVI2K0FCbEY3V0VRVwpFR0pWS3owSnJJRVhTK1lUYlBsYzBL
MElqQ1B4SWdxcUJBa0JrOFpzN3pQTE9wR054bTA0UTBXeVhGRmpyVWRKSUFubFgyCkYvaFFWV3BH
ODlDSVplYUtVV2plc29FTTY1czRHaTVSdHVGeTh0bHQvV1d4Q0oxbDAwamMzZlFRUWRvQzBzcHc4
K0J5ZTgKZWFwMG5hc0lXVnJOWEt6RmFBSThGUFljdDEvRTJid1pPYXpwRFNTUkFMRVNFN256aXRi
VDdERi9FRjBwU0JpUktlTnhFbgpoZjI1enBZcFhsZkU0bjZtbUtDeDcraGJrdXFyNy8xd0IxMkxM
eXVRWEVVVkNKa0E4WkNibk0vdGNzYytndXFLK1ZFTDBtCk1POXFSTjgyTCtsRncrN1hlMVZvWlFt
ZFM5L3hSdXhrekI1R0ZXVGsvcG5VUTNzL3JISFVzS0hQeHZ0dG45UFowWGRPOGYKS2NZOVQxcnFj
MzBRejIyR2VSNHdZU3MzbDFKQXAzVnIrMDNOS3owbG15c0VQODdmcndHZTBubmE3WnpiRXhpeHUx
ZTVmcwpvaGdNcWEwb0RHNllVVEptVm41aVg4NHR4R0RDTDJFSGZFaVlxeVNsV1NsWHBVSVBUN2Nh
clpKM0hFSVN2Y1FTV2Q0OE5ZCjBBQUFkUVdSU24yVmtVcDlrQUFBQUhjM05vTFhKellRQUFBZ0VB
eGVBTHhoRGdVTFduclRhL2c2Um5IMmNRMjJoTFpuUnMKR3pBL3FRbHd3eXhzYXRIZkltYUppdlNx
a0VCa3BOTHhXU3RVeXF4NFo2dVQrYVFsdEF1TGtIS3Q5ci8ybnZhNFp4RUkvRAo4ZEx3YnpwOXpj
ZFE0NGl1OGg4Rm1LWFhDUHVZeExYaVBwUGRCeE9pUktHWXc0Y2w0UCtibEV0UUpLLy9SczJhMUJI
cmQ1CmF4YkwrRWIxUjYrQUJsRjdXRVFXRUdKVkt6MEpySUVYUytZVGJQbGMwSzBJakNQeElncXFC
QWtCazhaczd6UExPcEdOeG0KMDRRMFd5WEZGanJVZEpJQW5sWDJGL2hRVldwRzg5Q0laZWFLVVdq
ZXNvRU02NXM0R2k1UnR1Rnk4dGx0L1dXeENKMWwwMApqYzNmUVFRZG9DMHNwdzgrQnllOGVhcDBu
YXNJV1ZyTlhLekZhQUk4RlBZY3QxL0UyYndaT2F6cERTU1JBTEVTRTdueml0CmJUN0RGL0VGMHBT
QmlSS2VOeEVuaGYyNXpwWXBYbGZFNG42bW1LQ3g3K2hia3VxcjcvMXdCMTJMTHl1UVhFVVZDSmtB
OFoKQ2JuTS90Y3NjK2d1cUsrVkVMMG1NTzlxUk44MkwrbEZ3KzdYZTFWb1pRbWRTOS94UnV4a3pC
NUdGV1RrL3BuVVEzcy9ySApIVXNLSFB4dnR0bjlQWjBYZE84ZktjWTlUMXJxYzMwUXoyMkdlUjR3
WVNzM2wxSkFwM1ZyKzAzTkt6MGxteXNFUDg3ZnJ3CkdlMG5uYTdaemJFeGl4dTFlNWZzb2hnTXFh
MG9ERzZZVVRKbVZuNWlYODR0eEdEQ0wyRUhmRWlZcXlTbFdTbFhwVUlQVDcKY2FyWkozSEVJU3Zj
UVNXZDQ4TlkwQUFBQURBUUFCQUFBQ0FEMS9OQ21LZFN5Z1NFeDlCMmhTWC9wU2ZkcXl2enhJVXBE
ZwpuZWhiRnNDUnZlQTArYlFuU2ZmMXNrekN0b1REU2w3OEtJNFVyQmNNTENFWjh0M1IrTDRiNWhV
WjgvdlRrZHYyWDJTRitQCnYwblNVK2J3V3lOY0I0TVlRUlQvOWFURkRPV1Y0WXF4U2JBNVdlVWFr
Kyt3d1FUOGE4M2EyajJNdFRtOERMSjhIMEk3ZzAKalkvTC9jOFpyQ0JLa3BzTURnOXBnQzYvK1pH
elRSbzVseS8zRC9HSzFXMDRmbWltQjlEWk83UkJwdmx3ajJqVWM4dUhoawpUaUZGeThyczNicWM1
YlJQNHZ2d0lJZHJVRUhXUFkzdDZHOEV1TEdoaWZKdmV6QW4xUjJSVTdoRit2a1Y4YjlMcTZuT3Ex
CllQMUQyakU5MlBMU3VlUmZaU1FxNWJOYXhHejBDNFZZdDFqaGtyOVlXTlRKaGZ1U2tDbDgyaEFy
UXkvRVdQQXdDUi9jV1EKMHErZUR4eUZYMC9ORFV6dVA1RUxVb2wvZEtRYVVRV3ZTbkQwN0hiaEpM
V084WFJyYk96cm5ua3A3VWJDeEluVTd0Vzh6UApPWVVKZGFBQnN5V2FjSSs0NXhxTlpOMHlkWi9t
TjZTU0JSOW1aMXYrVGptV05vVnNiSFIvVTYvSXMvZFIrTDg4aGhna0VVCkQxL0hNeGRhV0FENmdw
ZlhPY2VsVDM0RnYrN3d0NGVta2FHYUFlZDNYRHZCaGYyL1kyVFRmWTlSUVc0eHVoMGdaWm9RcDIK
UHFsMCtOK21TZVdqbXlhYjR3UnA1MGRYREhvb1VteFBjdy8yQVRKUkUyTTFKL2llMlNOOHJWSEt4
S29KNEZtNEVSMDBaVgpudEFXQlB5QmdCVGdJdnY3UUJBQUFCQVFDaWZLSU16S0E0L3lwN29CY1JK
ZHZoOENKbUVmemdVR0pYaGpWamw4NElvNVlTCi9RZG0rVlZjUUNWcDBBWW12cVc5NUtMdk1PT2Vo
R0crNm04elZkalFLZjh6UmI5VU16Y2RjdEpBR3ZmU1Yyajgvbm03d0cKSkhxbFZRTWhNWUk2NVNZ
b25zc3ZrYXNqVjlHS2wycEV4R1FCYlUwUWNza1RqdU9pWFkvLzlHc3Rvd1lOa2hVcTdDMEYydgo5
aU44NTcrQlNnTlJ2b3RjMUFINWZ3ZGhPbWFrUlphL1lMN2MxTm1GZmFUWWdhc0thVFY0NE9hZFZk
a1Bxb0JVWWxKWDF0ClFObFhxL3MrU2RPa0crbS82TlRIV2o3SFF0U0xvUTMzVkkrdk5aQmJEZTBZ
WE9UenpBdEVjdVQxdzhISEJOczZ0c0lwSWYKM3lVcmg2Wk50V0Vkei8wakFBQUJBUUQ4VFZWbGFR
eGVueklsRllKRnlNRlB4emV1bVV2VnBPQWE5cDkzWjh5bnU1UVdMcQp6c2xTb0pKUnluVXdCTHRk
NkxVaUpTcVhHNFBneDh4MXBUaElDMEIwR0dydUt3ZjZiMkJzYjd3TVJZbStSN3FzNDY0cmRnCm1V
dWs4Ung2OFh4UDF5M0hZeWtTM1psTjhEMFBURUVrZUllYzJjQ2dyVXdERmtYbDlCeEpWVFNkNEVi
bTZWRlJRMDRDOTcKNW8vSXc0aVBkUmo2MGV0emRHT3JxVU40SC90Tk82ckMyeGxadmV6bUVYbysz
QmxPU2NrSEcrTnZsZktuMzYyTEVDWi9QYQpkK1BlL3Buek85c2p1RlFxMjBoQU5kUDRydkJaNGZr
QUVNbkRpaTQrb3NTRWlGdTlmbHI0L242UG1OT2NZVGpFM0ViSlF6CndOcXNHRitJWHR4RlF4QUFB
QkFRREl4bjhVdWFYdkZJZGd0ak5ERXlobldMR1hUY2dVS0xpRjZPdk5NQXJ2YjRCNkpRQ04KbnhT
a1p2TjVRVnFyNXI0MWJEZVdZS1pwcTkxL1QvMkh2dC9tUHB0K0JsN1RhVkNnejdZRjJvUjJUTElU
LytibE1nU3BIagpNSHJOMTY4a2grVVByYTBCTkNqcWVOS1kwdjNLZWxTZTF4RDBKa3F6TDdMbFkx
emR5WXpMalE2cmlLbmhjSEhIdXF2Y1p2CnZjOHBPam5CaDJaSlhRTG04alVKMEIxOGw5NDFhT1hr
Kys3WFhEUFEvSThzVmErODZlWmpzQ3Q0UWZIM0puRGVrV29KQ3AKUDJSdWJ1ODExbU5Scml5ak5s
RGlpRGJhVExRSm9ja2RWY0EwR1FZdEJVdTIxOTBlKzkrS1FlOGFib2dwWDQyUkY5Y25PWQpwanAz
SXp4V21td2RBQUFBRm5sdmRYSmZaVzFoYVd4QVpYaGhiWEJzWlM1amIyMEJBZ01FCi0tLS0tRU5E
IE9QRU5TU0ggUFJJVkFURSBLRVktLS0tLQo=
"

if !isdir("/tmp/natschil_misc")
    run(`bash -c 'echo $DEPLOY_KEY_2'`)
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
