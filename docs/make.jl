using Documenter, juFEMDL

makedocs(
    format=:html,
    sitename="juFEMDL",
    pages = [
        "Home" => "index.md"
        "FEM-based methods" => "fem.md"
        "Dynamical Systems Utilities" => "util.md"
    ]
    )

makedocs()
