using Documenter, SelfFulfillingNKPC

DocMeta.setdocmeta!(SelfFulfillingNKPC, :DocTestSetup,
                    :(using SelfFulfillingNKPC); recursive=true)

makedocs(;
    modules = [SelfFulfillingNKPC],
    authors = "Niloufar Eslahi",
    sitename = "SelfFulfillingNKPC.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://nilufarslh.github.io/NKPC_SelfFullfilling_Econ622-Project/stable/",
        edit_link = "main",
        repolink  = "https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project",
        assets = String[],
    ),
    pages = [
        "Home"            => "index.md",
        "Methodology"     => "methodology.md",
        "Reproducibility" => "reproducibility.md",
        "API Reference"   => "api.md",
    ],
    checkdocs = :none,
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(;
    repo = "github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project",
    devbranch = "main",
    push_preview = true,
)
