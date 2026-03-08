using TKM3D
using Documenter

DocMeta.setdocmeta!(TKM3D, :DocTestSetup, :(using TKM3D); recursive=true)

makedocs(;
    modules=[TKM3D],
    authors="Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename="TKM3D.jl",
    format=Documenter.HTML(;
        canonical="https://ArrogantGao.github.io/TKM3D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ArrogantGao/TKM3D.jl",
    devbranch="main",
)
