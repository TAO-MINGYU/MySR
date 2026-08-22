"""
nuclear_structure_features_individual.py

Exhaustive one-feature-one-function version of the nuclear-structure
features defined in features_1_1.py::make_nuclear_structure_features.

Scope
-----
Only nuclear-structure features depending on neutron number n and proton
number z are included. Nuclear-reaction features are intentionally excluded.

Usage example
-------------
from nuclear_structure_features_individual import A, symmetry_term

print(A(92, 62))
print(symmetry_term(92, 62))
"""

from __future__ import annotations

from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd


ArrayLike = Union[int, float, list, tuple, np.ndarray, pd.Series]

EPS = 1.0e-12

# Same magic-number list as the original features_1_1.py file.
# 184 is used there as a common superheavy-shell proxy.
MAGIC_NUMBERS = np.array([2, 8, 20, 28, 50, 82, 126, 184], dtype=float)


# ============================================================
# Internal utilities
# ============================================================

def _to_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _to_int_array(x: ArrayLike) -> np.ndarray:
    return np.rint(_to_float_array(x)).astype(int)


def _safe_divide(numerator: ArrayLike, denominator: ArrayLike, eps: float = EPS) -> np.ndarray:
    return _to_float_array(numerator) / (_to_float_array(denominator) + eps)


def _safe_log(x: ArrayLike, eps: float = EPS) -> np.ndarray:
    return np.log(np.maximum(_to_float_array(x), eps))


def _safe_sqrt(x: ArrayLike, eps: float = EPS) -> np.ndarray:
    return np.sqrt(np.maximum(_to_float_array(x), eps))


def _format_result(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr)
    return arr


def _broadcast_nz(n: ArrayLike, z: ArrayLike):
    return np.broadcast_arrays(_to_float_array(n), _to_float_array(z))


# ============================================================
# Shell / magic-number utilities
# ============================================================

def _magic_bounds(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS):
    x_arr = _to_float_array(x)

    lower = np.full(x_arr.shape, magic_numbers[0], dtype=float)
    upper = np.full(x_arr.shape, magic_numbers[-1], dtype=float)

    for magic in magic_numbers:
        lower = np.where(x_arr >= magic, magic, lower)

    for magic in magic_numbers[::-1]:
        upper = np.where(x_arr <= magic, magic, upper)

    return lower, upper


def _nearest_magic_distance(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS) -> np.ndarray:
    x_arr = _to_float_array(x)
    return np.min(np.abs(x_arr[..., None] - magic_numbers), axis=-1)


def _valence_number_to_shell_closure(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS) -> np.ndarray:
    x_arr = _to_float_array(x)
    lower, upper = _magic_bounds(x_arr, magic_numbers)
    return np.minimum(np.abs(x_arr - lower), np.abs(upper - x_arr))


def _mid_shell_fraction(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS) -> np.ndarray:
    x_arr = _to_float_array(x)
    lower, upper = _magic_bounds(x_arr, magic_numbers)
    width = np.maximum(upper - lower, EPS)
    value = 4.0 * (x_arr - lower) * (upper - x_arr) / (width**2 + EPS)
    return np.clip(value, 0.0, 1.0)


# ============================================================
# 1. Basic identity features
# ============================================================

def N(n: ArrayLike, z: ArrayLike):
    """Neutron number N."""
    n, z = _broadcast_nz(n, z)
    return _format_result(n)


def Z(n: ArrayLike, z: ArrayLike):
    """Proton number Z."""
    n, z = _broadcast_nz(n, z)
    return _format_result(z)


def A(n: ArrayLike, z: ArrayLike):
    """Mass number: A = N + Z."""
    n, z = _broadcast_nz(n, z)
    return _format_result(n + z)


def N_minus_Z(n: ArrayLike, z: ArrayLike):
    """Neutron excess: N - Z."""
    n, z = _broadcast_nz(n, z)
    return _format_result(n - z)


def Z_minus_N(n: ArrayLike, z: ArrayLike):
    """Proton excess: Z - N."""
    n, z = _broadcast_nz(n, z)
    return _format_result(z - n)


def abs_N_minus_Z(n: ArrayLike, z: ArrayLike):
    """Absolute neutron-proton imbalance: |N - Z|."""
    return _format_result(np.abs(N_minus_Z(n, z)))


# ============================================================
# 2. Ratios and isospin features
# ============================================================

def N_over_Z(n: ArrayLike, z: ArrayLike):
    """Neutron-to-proton ratio: N/Z."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_safe_divide(n, z))


def Z_over_N(n: ArrayLike, z: ArrayLike):
    """Proton-to-neutron ratio: Z/N."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_safe_divide(z, n))


def N_over_A(n: ArrayLike, z: ArrayLike):
    """Neutron fraction: N/A."""
    n, z = _broadcast_nz(n, z)
    a = n + z
    return _format_result(_safe_divide(n, a))


def Z_over_A(n: ArrayLike, z: ArrayLike):
    """Proton fraction: Z/A."""
    n, z = _broadcast_nz(n, z)
    a = n + z
    return _format_result(_safe_divide(z, a))


def isospin_asymmetry(n: ArrayLike, z: ArrayLike):
    """Isospin asymmetry: I = (N - Z)/A."""
    n, z = _broadcast_nz(n, z)
    a = n + z
    return _format_result(_safe_divide(n - z, a))


def abs_isospin_asymmetry(n: ArrayLike, z: ArrayLike):
    """Absolute isospin asymmetry: |I|."""
    return _format_result(np.abs(isospin_asymmetry(n, z)))


def isospin_asymmetry_squared(n: ArrayLike, z: ArrayLike):
    """Squared isospin asymmetry: I^2."""
    i = isospin_asymmetry(n, z)
    return _format_result(np.asarray(i) ** 2)


def Tz(n: ArrayLike, z: ArrayLike):
    """Third component of isospin proxy: Tz = (N - Z)/2."""
    return _format_result(0.5 * np.asarray(N_minus_Z(n, z)))


# ============================================================
# 3. Mass-number power features
# ============================================================

def A_1_3(n: ArrayLike, z: ArrayLike):
    """A^(1/3), common nuclear-radius scale."""
    a = np.asarray(A(n, z))
    return _format_result(np.cbrt(np.maximum(a, EPS)))


def A_2_3(n: ArrayLike, z: ArrayLike):
    """A^(2/3), common nuclear-surface scale."""
    return _format_result(np.asarray(A_1_3(n, z)) ** 2)


def sqrt_A(n: ArrayLike, z: ArrayLike):
    """sqrt(A)."""
    return _format_result(_safe_sqrt(A(n, z)))


def inv_A(n: ArrayLike, z: ArrayLike):
    """1/A."""
    return _format_result(_safe_divide(1.0, A(n, z)))


def inv_A_1_3(n: ArrayLike, z: ArrayLike):
    """1/A^(1/3)."""
    return _format_result(_safe_divide(1.0, A_1_3(n, z)))


def inv_A_2_3(n: ArrayLike, z: ArrayLike):
    """1/A^(2/3)."""
    return _format_result(_safe_divide(1.0, A_2_3(n, z)))


def inv_sqrt_A(n: ArrayLike, z: ArrayLike):
    """1/sqrt(A)."""
    return _format_result(_safe_divide(1.0, sqrt_A(n, z)))


def log_A(n: ArrayLike, z: ArrayLike):
    """log(A)."""
    return _format_result(_safe_log(A(n, z)))


# ============================================================
# 4. Liquid-drop / SEMF-inspired features
# ============================================================

def volume_term(n: ArrayLike, z: ArrayLike):
    """Liquid-drop volume proxy: A."""
    return A(n, z)


def surface_term(n: ArrayLike, z: ArrayLike):
    """Liquid-drop surface proxy: A^(2/3)."""
    return A_2_3(n, z)


def radius_term(n: ArrayLike, z: ArrayLike):
    """Nuclear radius proxy: A^(1/3)."""
    return A_1_3(n, z)


def coulomb_ZZ_over_A13(n: ArrayLike, z: ArrayLike):
    """Coulomb proxy: Z(Z - 1)/A^(1/3)."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_safe_divide(z * (z - 1.0), A_1_3(n, z)))


def coulomb_Z2_over_A13(n: ArrayLike, z: ArrayLike):
    """Alternative Coulomb proxy: Z^2/A^(1/3)."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_safe_divide(z**2, A_1_3(n, z)))


def symmetry_term(n: ArrayLike, z: ArrayLike):
    """Symmetry-energy proxy: (N - Z)^2/A."""
    n, z = _broadcast_nz(n, z)
    a = n + z
    return _format_result(_safe_divide((n - z) ** 2, a))


def symmetry_term_normalized(n: ArrayLike, z: ArrayLike):
    """Normalized symmetry proxy: I^2."""
    return isospin_asymmetry_squared(n, z)


def surface_symmetry_proxy(n: ArrayLike, z: ArrayLike):
    """Surface-symmetry proxy: A^(2/3) I^2."""
    return _format_result(np.asarray(A_2_3(n, z)) * np.asarray(isospin_asymmetry_squared(n, z)))


def fissility_Z2_over_A(n: ArrayLike, z: ArrayLike):
    """Fissility-like proxy: Z^2/A."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_safe_divide(z**2, n + z))


def charge_density_proxy(n: ArrayLike, z: ArrayLike):
    """Charge fraction proxy: Z/A."""
    return Z_over_A(n, z)


# ============================================================
# 5. Parity / pairing features
# ============================================================

def is_even_N(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if N is even, else 0."""
    n, z = _broadcast_nz(n, z)
    return _format_result((_to_int_array(n) % 2 == 0).astype(float))


def is_even_Z(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if Z is even, else 0."""
    n, z = _broadcast_nz(n, z)
    return _format_result((_to_int_array(z) % 2 == 0).astype(float))


def is_odd_N(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if N is odd, else 0."""
    return _format_result(1.0 - np.asarray(is_even_N(n, z)))


def is_odd_Z(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if Z is odd, else 0."""
    return _format_result(1.0 - np.asarray(is_even_Z(n, z)))


def is_even_even(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 for even-even nuclei."""
    return _format_result(np.asarray(is_even_N(n, z)) * np.asarray(is_even_Z(n, z)))


def is_odd_odd(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 for odd-odd nuclei."""
    return _format_result(np.asarray(is_odd_N(n, z)) * np.asarray(is_odd_Z(n, z)))


def is_odd_A(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if A is odd."""
    return _format_result((_to_int_array(A(n, z)) % 2 != 0).astype(float))


def pairing_sign(n: ArrayLike, z: ArrayLike):
    """
    Pairing sign:
    +1 for even-even nuclei;
     0 for odd-A nuclei;
    -1 for odd-odd nuclei.
    """
    ee = np.asarray(is_even_even(n, z))
    oo = np.asarray(is_odd_odd(n, z))
    return _format_result(np.where(ee == 1.0, 1.0, np.where(oo == 1.0, -1.0, 0.0)))


def pairing_A_minus_1_2(n: ArrayLike, z: ArrayLike):
    """Pairing proxy: pairing_sign / A^(1/2)."""
    return _format_result(np.asarray(pairing_sign(n, z)) * np.asarray(inv_sqrt_A(n, z)))


def pairing_A_minus_3_4(n: ArrayLike, z: ArrayLike):
    """Pairing proxy: pairing_sign / A^(3/4)."""
    return _format_result(np.asarray(pairing_sign(n, z)) * _safe_divide(1.0, np.maximum(A(n, z), EPS) ** 0.75))


def pairing_A_minus_1(n: ArrayLike, z: ArrayLike):
    """Pairing proxy: pairing_sign / A."""
    return _format_result(np.asarray(pairing_sign(n, z)) * np.asarray(inv_A(n, z)))


# ============================================================
# 6. Magic-number / shell features
# ============================================================

def lower_magic_N(n: ArrayLike, z: ArrayLike):
    """Lower magic number surrounding N."""
    n, z = _broadcast_nz(n, z)
    lower, upper = _magic_bounds(n)
    return _format_result(lower)


def upper_magic_N(n: ArrayLike, z: ArrayLike):
    """Upper magic number surrounding N."""
    n, z = _broadcast_nz(n, z)
    lower, upper = _magic_bounds(n)
    return _format_result(upper)


def lower_magic_Z(n: ArrayLike, z: ArrayLike):
    """Lower magic number surrounding Z."""
    n, z = _broadcast_nz(n, z)
    lower, upper = _magic_bounds(z)
    return _format_result(lower)


def upper_magic_Z(n: ArrayLike, z: ArrayLike):
    """Upper magic number surrounding Z."""
    n, z = _broadcast_nz(n, z)
    lower, upper = _magic_bounds(z)
    return _format_result(upper)


def shell_width_N(n: ArrayLike, z: ArrayLike):
    """Width of the N shell interval between surrounding magic numbers."""
    lower = np.asarray(lower_magic_N(n, z))
    upper = np.asarray(upper_magic_N(n, z))
    return _format_result(np.maximum(upper - lower, EPS))


def shell_width_Z(n: ArrayLike, z: ArrayLike):
    """Width of the Z shell interval between surrounding magic numbers."""
    lower = np.asarray(lower_magic_Z(n, z))
    upper = np.asarray(upper_magic_Z(n, z))
    return _format_result(np.maximum(upper - lower, EPS))


def distance_to_magic_N(n: ArrayLike, z: ArrayLike):
    """Distance from N to the nearest magic number."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_nearest_magic_distance(n))


def distance_to_magic_Z(n: ArrayLike, z: ArrayLike):
    """Distance from Z to the nearest magic number."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_nearest_magic_distance(z))


def distance_to_magic_sum(n: ArrayLike, z: ArrayLike):
    """Sum of neutron and proton distances to nearest magic numbers."""
    return _format_result(np.asarray(distance_to_magic_N(n, z)) + np.asarray(distance_to_magic_Z(n, z)))


def distance_to_magic_product(n: ArrayLike, z: ArrayLike):
    """Product of neutron and proton distances to nearest magic numbers."""
    return _format_result(np.asarray(distance_to_magic_N(n, z)) * np.asarray(distance_to_magic_Z(n, z)))


def valence_neutron_number(n: ArrayLike, z: ArrayLike):
    """Valence neutron number relative to nearest shell closure."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_valence_number_to_shell_closure(n))


def valence_proton_number(n: ArrayLike, z: ArrayLike):
    """Valence proton number relative to nearest shell closure."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_valence_number_to_shell_closure(z))


def valence_neutron_fraction(n: ArrayLike, z: ArrayLike):
    """Normalized valence-neutron fraction: clip(2*Nv/shell_width_N, 0, 1)."""
    return _format_result(np.clip(2.0 * np.asarray(valence_neutron_number(n, z)) / np.asarray(shell_width_N(n, z)), 0.0, 1.0))


def valence_proton_fraction(n: ArrayLike, z: ArrayLike):
    """Normalized valence-proton fraction: clip(2*Zv/shell_width_Z, 0, 1)."""
    return _format_result(np.clip(2.0 * np.asarray(valence_proton_number(n, z)) / np.asarray(shell_width_Z(n, z)), 0.0, 1.0))


def valence_product_NpNn(n: ArrayLike, z: ArrayLike):
    """Product of valence proton and valence neutron numbers: Np*Nn proxy."""
    return _format_result(np.asarray(valence_neutron_number(n, z)) * np.asarray(valence_proton_number(n, z)))


def casten_P_factor(n: ArrayLike, z: ArrayLike):
    """Casten P-factor proxy: P = Np*Nn/(Np + Nn)."""
    vn = np.asarray(valence_neutron_number(n, z))
    vp = np.asarray(valence_proton_number(n, z))
    return _format_result(_safe_divide(vn * vp, vn + vp))


# ============================================================
# 7. Mid-shell / collectivity proxy features
# ============================================================

def mid_shell_fraction_N(n: ArrayLike, z: ArrayLike):
    """Mid-shell fraction for N: 0 near shell closure, 1 near mid-shell."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_mid_shell_fraction(n))


def mid_shell_fraction_Z(n: ArrayLike, z: ArrayLike):
    """Mid-shell fraction for Z: 0 near shell closure, 1 near mid-shell."""
    n, z = _broadcast_nz(n, z)
    return _format_result(_mid_shell_fraction(z))


def mid_shell_fraction_sum(n: ArrayLike, z: ArrayLike):
    """Sum of neutron and proton mid-shell fractions."""
    return _format_result(np.asarray(mid_shell_fraction_N(n, z)) + np.asarray(mid_shell_fraction_Z(n, z)))


def mid_shell_fraction_product(n: ArrayLike, z: ArrayLike):
    """Product of neutron and proton mid-shell fractions."""
    return _format_result(np.asarray(mid_shell_fraction_N(n, z)) * np.asarray(mid_shell_fraction_Z(n, z)))


# ============================================================
# 8. Simple region indicators
# ============================================================

def is_neutron_rich(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if N > Z."""
    n, z = _broadcast_nz(n, z)
    return _format_result((n > z).astype(float))


def is_proton_rich(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if Z > N."""
    n, z = _broadcast_nz(n, z)
    return _format_result((z > n).astype(float))


def is_N_equal_Z(n: ArrayLike, z: ArrayLike):
    """Indicator: 1 if rounded N equals rounded Z."""
    n, z = _broadcast_nz(n, z)
    return _format_result((_to_int_array(n) == _to_int_array(z)).astype(float))


# ============================================================
# Convenience registry and table builder
# ============================================================

FEATURE_NAMES: List[str] = [
    "N",
    "Z",
    "A",
    "N_minus_Z",
    "Z_minus_N",
    "abs_N_minus_Z",
    "N_over_Z",
    "Z_over_N",
    "N_over_A",
    "Z_over_A",
    "isospin_asymmetry",
    "abs_isospin_asymmetry",
    "isospin_asymmetry_squared",
    "Tz",
    "A_1_3",
    "A_2_3",
    "sqrt_A",
    "inv_A",
    "inv_A_1_3",
    "inv_A_2_3",
    "inv_sqrt_A",
    "log_A",
    "volume_term",
    "surface_term",
    "radius_term",
    "coulomb_ZZ_over_A13",
    "coulomb_Z2_over_A13",
    "symmetry_term",
    "symmetry_term_normalized",
    "surface_symmetry_proxy",
    "fissility_Z2_over_A",
    "charge_density_proxy",
    "is_even_N",
    "is_even_Z",
    "is_odd_N",
    "is_odd_Z",
    "is_even_even",
    "is_odd_odd",
    "is_odd_A",
    "pairing_sign",
    "pairing_A_minus_1_2",
    "pairing_A_minus_3_4",
    "pairing_A_minus_1",
    "lower_magic_N",
    "upper_magic_N",
    "lower_magic_Z",
    "upper_magic_Z",
    "shell_width_N",
    "shell_width_Z",
    "distance_to_magic_N",
    "distance_to_magic_Z",
    "distance_to_magic_sum",
    "distance_to_magic_product",
    "valence_neutron_number",
    "valence_proton_number",
    "valence_neutron_fraction",
    "valence_proton_fraction",
    "valence_product_NpNn",
    "casten_P_factor",
    "mid_shell_fraction_N",
    "mid_shell_fraction_Z",
    "mid_shell_fraction_sum",
    "mid_shell_fraction_product",
    "is_neutron_rich",
    "is_proton_rich",
    "is_N_equal_Z",
]

FEATURE_FUNCTIONS: Dict[str, Callable[[ArrayLike, ArrayLike], object]] = {
    name: globals()[name] for name in FEATURE_NAMES
}


def make_nuclear_structure_feature_table(
    n: ArrayLike,
    z: ArrayLike,
    selected_features: List[str] | None = None,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from the individual feature functions.

    Parameters
    ----------
    n, z:
        Neutron and proton numbers. Scalars, arrays, lists, or pandas Series.
    selected_features:
        If None, build all 66 features. Otherwise build only the requested names.

    Returns
    -------
    pandas.DataFrame
        One column per feature.
    """
    names = FEATURE_NAMES if selected_features is None else selected_features

    unknown = [name for name in names if name not in FEATURE_FUNCTIONS]
    if unknown:
        raise KeyError(f"Unknown feature name(s): {unknown}")

    return pd.DataFrame({
        name: np.ravel(FEATURE_FUNCTIONS[name](n, z))
        for name in names
    })
