"""Functions for initializing the Julia environment and installing deps."""

import os
from typing import Any, Callable, cast, overload

import numpy as np
from juliacall import convert as jl_convert  # type: ignore
from numpy.typing import NDArray

from .deprecated import init_julia, install
from .julia_import import AnyValue, jl

jl_convert = cast(Callable[[Any, Any], Any], jl_convert)

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")

Serialization = jl.Serialization
PythonCall = jl.PythonCall

jl.seval("using MySRCore.SymbolicRegression: plus, sub, mult, div, pow")

# Compile the predicate once during bridge initialization. ``jl_is_function``
# is used repeatedly while validating operators and losses; evaluating the
# same Julia closure on every call needlessly reparses and recompiles it.
_JL_IS_FUNCTION = cast(Callable[[Any], bool], jl.seval("op -> op isa Function"))


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    if filename is None:
        return None
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def _load_cluster_manager(cluster_manager: str):
    if cluster_manager == "slurm":
        jl.seval("using Distributed: addprocs")
        jl.seval("using SlurmClusterManager: SlurmManager")
        return jl.seval("""
            (numprocs; kws...) -> begin
                # A large single-node allocation can take longer than the
                # upstream 60-second default to report every worker through
                # srun.  Keep the default bounded but scale it with the
                # requested worker count; callers may override it explicitly
                # with MYSR_SLURM_LAUNCH_TIMEOUT.
                launch_timeout = max(
                    60.0,
                    something(
                        tryparse(Float64, get(ENV, "MYSR_SLURM_LAUNCH_TIMEOUT", "")),
                        0.5 * numprocs,
                    ),
                )
                manager = SlurmManager(launch_timeout=launch_timeout)
                manager.ntasks == numprocs || error(
                    "Requested $numprocs processes, but Slurm allocation has $(manager.ntasks) tasks. " *
                    "Set Slurm `--ntasks`/`--ntasks-per-node` and `procs` to the same value."
                )
                addprocs(manager; kws...)
            end
            """)
    jl.seval(f"using ClusterManagers: addprocs_{cluster_manager}")
    return jl.seval(f"addprocs_{cluster_manager}")


def jl_array(x, dtype=None):
    if x is None:
        return None
    elif dtype is None:
        return jl_convert(jl.Array, x)
    else:
        return jl_convert(jl.Array[dtype], x)


def jl_dict(x):
    return jl_convert(jl.Dict, x)


def jl_named_tuple(d):
    return jl.NamedTuple({jl.Symbol(k): v for k, v in d.items()})


def jl_is_function(f) -> bool:
    return bool(_JL_IS_FUNCTION(f))


def jl_serialize(obj: Any) -> NDArray[np.uint8]:
    buf = jl.IOBuffer()
    Serialization.serialize(buf, obj)
    return np.array(jl.take_b(buf))


@overload
def jl_deserialize(s: NDArray[np.uint8]) -> AnyValue: ...
@overload
def jl_deserialize(s: None) -> None: ...
def jl_deserialize(s):
    if s is None:
        return s
    buf = jl.IOBuffer()
    jl.write(buf, jl_array(s))
    jl.seekstart(buf)
    return Serialization.deserialize(buf)
