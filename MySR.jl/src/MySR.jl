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

# String operator name -> Julia function (this engine only accepts Function objects).
const BINARY_OPS = Dict(
    "+" => +, "-" => -, "*" => *, "/" => /, "^" => ^, "pow" => ^,
)
const UNARY_OPS = Dict(
    "cos" => cos, "sin" => sin, "tan" => tan, "exp" => exp,
    "log" => log, "sqrt" => sqrt, "tanh" => tanh, "abs" => abs,
    "square" => x -> x^2, "cube" => x -> x^3,
)

function _ops(names::Vector{Any}, table::Dict{String,Function})
    out = Function[]
    for n in names
        key = string(n)
        if haskey(table, key)
            push!(out, table[key])
        else
            error("unknown operator: $n")
        end
    end
    return out
end

"""
    search_from_files(X_path, y_path, config_path, out_path)

Load features `X_path` (n×d CSV), targets `y_path` (n CSV) and a JSON
`config_path`, run `SymbolicRegression.equation_search`, and write a JSON
hall-of-fame (`[{equation, loss, complexity}, ...]`) to `out_path`.
"""
function search_from_files(
    X_path::String, y_path::String, config_path::String, out_path::String
)
    X = permutedims(readdlm(X_path, ',', Float64))  # SR.jl expects [features, rows]
    y = vec(readdlm(y_path, ',', Float64))
    cfg = JSON.parsefile(config_path)

    opt = Options(;
        binary_operators=_ops(get(cfg, "binary_operators", ["+", "-", "*", "/"]), BINARY_OPS),
        unary_operators=_ops(get(cfg, "unary_operators", ["cos", "exp", "log", "sqrt"]), UNARY_OPS),
        maxsize=get(cfg, "maxsize", 20),
        maxdepth=get(cfg, "maxdepth", nothing),
        parsimony=get(cfg, "parsimony", 1e-4),
        populations=get(cfg, "populations", 20),
        population_size=get(cfg, "population_size", 33),
        ncycles_per_iteration=get(cfg, "ncycles_per_iteration", 550),
        tournament_selection_n=get(cfg, "tournament_selection_n", 10),
        tournament_selection_p=get(cfg, "tournament_selection_p", 0.86),
        seed=get(cfg, "seed", nothing),
    )

    hof = equation_search(
        X, y;
        niterations=get(cfg, "niterations", 40),
        options=opt,
    )

    members = hof.members
    out = [
        Dict(
            "equation" => string(m.tree),
            "loss" => (isnan(m.loss) || isinf(m.loss)) ? missing : m.loss,
            "complexity" => compute_complexity(m.tree, opt),
        ) for m in members
    ]

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, out)
    end
    return out
end

end # module
