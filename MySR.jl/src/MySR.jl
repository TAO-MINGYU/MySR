"""
    MySR

Julia backend of the MySR toolkit: a thin, physics-flavoured wrapper around
`SymbolicRegression.jl` (the "srjl" engine merged into MySR).

The Python frontend (`mysr` package) drives this module through
`search_from_files`, which mirrors the PySR-style bridge documented in
`Reference/PySR/Distilled_PySR_SR.jl/03_julia_bridge.md`.
"""
module MySR

using DelimitedFiles
using JSON
using SymbolicRegression

export search_from_files

"""
    search_from_files(X_path, y_path, config_path, out_path)

Load features `X_path` (n×d CSV), targets `y_path` (n CSV) and a JSON
`config_path`, run `SymbolicRegression.equation_search`, and write a JSON
hall-of-fame (`[{equation, loss, complexity}, ...]`) to `out_path`.
"""
function search_from_files(
    X_path::String, y_path::String, config_path::String, out_path::String
)
    X = readdlm(X_path, ',', Float64)
    y = vec(readdlm(y_path, ',', Float64))
    cfg = JSON.parsefile(config_path)

    hof = equation_search(
        X, y;
        niterations=get(cfg, "niterations", 40),
        populations=get(cfg, "populations", 20),
        population_size=get(cfg, "population_size", 33),
        ncycles_per_iteration=get(cfg, "ncycles_per_iteration", 550),
        maxsize=get(cfg, "maxsize", 20),
        maxdepth=get(cfg, "maxdepth", nothing),
        parsimony=get(cfg, "parsimony", 1e-4),
        loss=get(cfg, "loss", "L2DistLoss()"),
        binary_operators=get(cfg, "binary_operators", ["+", "-", "*", "/"]),
        unary_operators=get(cfg, "unary_operators", ["cos", "exp", "log", "sqrt"]),
        tournament_selection_n=get(cfg, "tournament_selection_n", 10),
        tournament_selection_p=get(cfg, "tournament_selection_p", 0.86),
        seed=get(cfg, "seed", nothing),
    )

    eqs = hof.equations
    out = [
        Dict(
            "equation" => string(row.equation),
            "loss" => row.loss,
            "complexity" => row.complexity,
        ) for row in eachrow(eqs)
    ]

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, out)
    end
    return out
end

end # module
