#!/usr/bin/env julia
# Driver invoked by the Python frontend (mysr/julia_bridge.py).
# Usage: julia --project=MySR.jl run_from_files.jl <X.csv> <y.csv> <config.json> <out.json>
using MySR

out = MySR.search_from_files(ARGS[1], ARGS[2], ARGS[3], ARGS[4])
println("DONE: $(length(out)) equations")
