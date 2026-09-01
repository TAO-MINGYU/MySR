"""User-facing dimensional specifications for MySR.

MySR accepts dimensions, not physical-unit expressions. A dimension is
represented by seven exponents in the fixed DynamicQuantities basis
(length, mass, time, current, temperature, luminosity, amount). For
readability, mappings may use either these canonical names or the aliases
(L, M, T, I, Theta, J, N).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

DIMENSION_BASIS = (
    "length",
    "mass",
    "time",
    "current",
    "temperature",
    "luminosity",
    "amount",
)
DIMENSION_ALIASES = {
    "l": "length",
    "length": "length",
    "m": "mass",
    "mass": "mass",
    "t": "time",
    "time": "time",
    "i": "current",
    "current": "current",
    "theta": "temperature",
    "temperature": "temperature",
    "j": "luminosity",
    "luminosity": "luminosity",
    "n": "amount",
    "amount": "amount",
}
DimensionVector = tuple[float, ...]
DIMENSION_TOLERANCE = 1.0e-12


def _is_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _canonical_name(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError("dimension mapping keys must be strings")
    key = name.strip().lower()
    try:
        return DIMENSION_ALIASES[key]
    except KeyError as error:
        allowed = ", ".join(DIMENSION_BASIS)
        raise ValueError(
            f"unknown dimension basis '{name}'; use one of: {allowed}"
        ) from error


def normalize_dimension(value: Any, *, name: str = "dimension") -> DimensionVector:
    """Normalize one dimension to a seven-component exponent vector.

    Accepted values are mappings such as {"mass": 1, "length": 2} or
    numeric sequences of length seven in DIMENSION_BASIS order. Strings are
    deliberately rejected: MySR no longer accepts unit strings.
    """

    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a dimension mapping or a length-7 numeric vector; "
            "unit strings are not accepted"
        )
    if isinstance(value, Mapping):
        result = [0.0] * len(DIMENSION_BASIS)
        index = {basis: i for i, basis in enumerate(DIMENSION_BASIS)}
        for key, exponent in value.items():
            basis = _canonical_name(key)
            if not _is_scalar(exponent) or not np.isfinite(float(exponent)):
                raise TypeError(f"{name}[{key!r}] must be a finite real exponent")
            result[index[basis]] = float(exponent)
        return tuple(result)

    # NumPy arrays are not guaranteed to register as collections.abc.Sequence
    # across NumPy versions, so handle one-dimensional arrays explicitly.
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional length-7 vector")
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a dimension mapping or a length-7 numeric vector"
        )
    if len(value) != len(DIMENSION_BASIS):
        raise ValueError(
            f"{name} vectors must have exactly {len(DIMENSION_BASIS)} exponents "
            f"in {DIMENSION_BASIS} order"
        )
    result = []
    for exponent in value:
        if not _is_scalar(exponent) or not np.isfinite(float(exponent)):
            raise TypeError(f"{name} exponents must be finite real numbers")
        result.append(float(exponent))
    return tuple(result)


def _looks_like_vector(value: Any) -> bool:
    if isinstance(value, (Mapping, str, bytes)):
        return False
    try:
        return len(value) == len(DIMENSION_BASIS) and all(
            _is_scalar(item) for item in value
        )
    except (TypeError, ValueError):
        return False


def normalize_input_dimensions(
    value: Any,
    n_features: int,
    *,
    name: str = "X_dimensions",
) -> list[DimensionVector] | None:
    """Normalize one dimension specification per input feature."""

    if value is None:
        return None
    if isinstance(value, Mapping) or _looks_like_vector(value):
        if n_features != 1:
            raise ValueError(
                f"{name} must contain one dimension specification per feature"
            )
        return [normalize_dimension(value, name=f"{name}[0]")]
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of dimension specifications")
    try:
        specifications = list(value)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a sequence of dimension specifications"
        ) from error
    if len(specifications) != n_features:
        raise ValueError(
            f"{name} must contain {n_features} dimension specifications, "
            f"got {len(specifications)}"
        )
    return [
        normalize_dimension(item, name=f"{name}[{index}]")
        for index, item in enumerate(specifications)
    ]


def normalize_output_dimensions(
    value: Any,
    n_outputs: int,
    *,
    name: str = "y_dimensions",
) -> DimensionVector | list[DimensionVector] | None:
    """Normalize one output dimension or one dimension per output."""

    if value is None:
        return None
    if isinstance(value, Mapping) or _looks_like_vector(value):
        if n_outputs != 1:
            raise ValueError(
                f"{name} must contain one dimension specification per output"
            )
        return normalize_dimension(value, name=name)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a dimension mapping or a sequence")
    try:
        specifications = list(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a dimension mapping or a sequence") from error
    if len(specifications) != n_outputs:
        raise ValueError(
            f"{name} must contain {n_outputs} dimension specifications, "
            f"got {len(specifications)}"
        )
    return [
        normalize_dimension(item, name=f"{name}[{index}]")
        for index, item in enumerate(specifications)
    ]


def dimension_is_zero(value: DimensionVector) -> bool:
    return all(abs(exponent) <= DIMENSION_TOLERANCE for exponent in value)


def dimension_equal(left: DimensionVector, right: DimensionVector) -> bool:
    """Compare dimension exponents with the tolerance used for propagation."""

    return len(left) == len(right) and bool(
        np.allclose(left, right, rtol=0.0, atol=DIMENSION_TOLERANCE)
    )


def dimension_add(left: DimensionVector, right: DimensionVector) -> DimensionVector:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def dimension_sub(left: DimensionVector, right: DimensionVector) -> DimensionVector:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def dimension_scale(value: DimensionVector, factor: float) -> DimensionVector:
    return tuple(factor * exponent for exponent in value)


def dimension_to_mapping(value: DimensionVector) -> dict[str, float]:
    return {
        basis: (
            round(exponent)
            if abs(exponent - round(exponent)) <= DIMENSION_TOLERANCE
            else exponent
        )
        for basis, exponent in zip(DIMENSION_BASIS, value, strict=True)
        if abs(exponent) > DIMENSION_TOLERANCE
    }


def dimension_to_string(value: DimensionVector) -> str:
    """Return a compact dimension label for reports."""

    mapping = dimension_to_mapping(value)
    if not mapping:
        return "1"
    parts = []
    for basis, exponent in mapping.items():
        if exponent == 1:
            parts.append(basis)
        else:
            parts.append(f"{basis}^{exponent}")
    return "*".join(parts)
