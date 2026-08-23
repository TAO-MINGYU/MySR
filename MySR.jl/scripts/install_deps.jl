# Install MySR.jl dependencies (SymbolicRegression + JSON).
using Pkg
Pkg.add(["SymbolicRegression", "JSON"])
Pkg.instantiate()
Pkg.precompile()
println("JULIA_DEPS_OK")
