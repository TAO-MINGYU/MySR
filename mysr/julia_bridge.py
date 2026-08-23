"""Bridge between the Python frontend and the Julia backend.

Writes data + config to temporary files, launches ``julia --project=<MySR.jl>``
with a small driver script, and parses the returned hall of fame JSON.

This follows the PySR-style split (see Reference/PySR/Distilled_PySR_SR.jl/
03_julia_bridge.md): Python orchestrates, Julia searches.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# MySR.jl lives at the repo root next to the `mysr` package.
_MYSR_JL = Path(__file__).resolve().parent.parent / "MySR.jl"
_DRIVER = _MYSR_JL / "scripts" / "run_from_files.jl"


def find_julia() -> str:
    # 1) sibling of the running python (conda env: julia lives in env/bin)
    candidate = Path(sys.executable).resolve().parent / "julia"
    if candidate.exists():
        return str(candidate)
    # 2) PATH
    exe = shutil.which("julia")
    if exe:
        return exe
    raise RuntimeError(
        "julia executable not found; activate the env_mysr conda env"
    )


def run_search(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    if not _DRIVER.exists():
        raise FileNotFoundError(
            f"Julia driver not found: {_DRIVER} (is MySR.jl present in the repo?)"
        )

    tmp = Path(tempfile.mkdtemp(prefix="mysr_"))
    try:
        x_path = tmp / "X.csv"
        y_path = tmp / "y.csv"
        cfg_path = tmp / "config.json"
        out_path = tmp / "hall_of_fame.json"

        np.savetxt(x_path, X, delimiter=",", fmt="%.10g")
        np.savetxt(y_path, y, delimiter=",", fmt="%.10g")
        cfg_path.write_text(json.dumps(config), encoding="utf-8")

        cmd = [
            find_julia(),
            "--threads=auto",
            f"--project={_MYSR_JL}",
            str(_DRIVER),
            str(x_path),
            str(y_path),
            str(cfg_path),
            str(out_path),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if verbose:
            print(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia backend failed (exit {proc.returncode}):\n{proc.stderr}"
            )

        if not out_path.exists():
            raise RuntimeError("Julia backend finished without hall_of_fame.json")
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
