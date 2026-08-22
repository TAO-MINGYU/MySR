"""
nuclear_feature_builders.py

Common nuclear-structure and low-energy nuclear-reaction feature builders.

Conventions
-----------
N, Z:
    Neutron number and proton number of a nucleus.

Projectile:
    n_p, z_p

Target:
    n_t, z_t

Energy:
    energy is assumed to be in MeV.
    If energy_frame="lab", energy is projectile laboratory energy.
    If energy_frame="cm", energy is center-of-mass energy.

Notes
-----
These features are candidate features for symbolic regression.
Do not use all of them blindly in one run.
Select a physically motivated subset for each task.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd


ArrayLike = Union[int, float, list, tuple, np.ndarray, pd.Series]


EPS = 1.0e-12

# Common nuclear magic numbers.
# 184 is included as a common superheavy-shell proxy.
MAGIC_NUMBERS = np.array([2, 8, 20, 28, 50, 82, 126, 184], dtype=float)

# Constants for approximate reaction features.
R0_FM = 1.20
COULOMB_CONST_MEV_FM = 1.439964


# ============================================================
# Basic utilities
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


def _format_output(features: Dict[str, np.ndarray], as_dataframe: bool = False):
    if as_dataframe:
        return pd.DataFrame({key: np.ravel(value) for key, value in features.items()})

    formatted = {}
    for key, value in features.items():
        arr = np.asarray(value)
        if arr.shape == ():
            formatted[key] = float(arr)
        else:
            formatted[key] = arr

    return formatted


def _prefix_features(features: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    return {f"{prefix}{key}": value for key, value in features.items()}


# ============================================================
# Parity / pairing utilities
# ============================================================

def even_indicator(x: ArrayLike) -> np.ndarray:
    x_int = _to_int_array(x)
    return (x_int % 2 == 0).astype(float)


def odd_indicator(x: ArrayLike) -> np.ndarray:
    x_int = _to_int_array(x)
    return (x_int % 2 != 0).astype(float)


def pairing_sign(n: ArrayLike, z: ArrayLike) -> np.ndarray:
    """
    Pairing sign convention:
    +1 for even-even nuclei
     0 for odd-A nuclei
    -1 for odd-odd nuclei
    """
    n_even = even_indicator(n)
    z_even = even_indicator(z)

    n_odd = 1.0 - n_even
    z_odd = 1.0 - z_even

    return np.where(
        (n_even == 1.0) & (z_even == 1.0),
        1.0,
        np.where((n_odd == 1.0) & (z_odd == 1.0), -1.0, 0.0),
    )


# ============================================================
# Shell / magic-number utilities
# ============================================================

def nearest_magic_distance(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS) -> np.ndarray:
    x_arr = _to_float_array(x)
    return np.min(np.abs(x_arr[..., None] - magic_numbers), axis=-1)


def magic_bounds(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS):
    """
    Return lower and upper magic numbers surrounding x.

    If x is outside the provided magic-number range, the nearest boundary
    is used. This is a proxy, not a microscopic shell-model calculation.
    """
    x_arr = _to_float_array(x)

    lower = np.full(x_arr.shape, magic_numbers[0], dtype=float)
    upper = np.full(x_arr.shape, magic_numbers[-1], dtype=float)

    for magic in magic_numbers:
        lower = np.where(x_arr >= magic, magic, lower)

    for magic in magic_numbers[::-1]:
        upper = np.where(x_arr <= magic, magic, upper)

    return lower, upper


def valence_number_to_shell_closure(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS) -> np.ndarray:
    """
    Valence nucleon number relative to the nearest shell closure.

    This is often used as a proxy for shell evolution and collectivity.
    """
    x_arr = _to_float_array(x)
    lower, upper = magic_bounds(x_arr, magic_numbers)
    return np.minimum(np.abs(x_arr - lower), np.abs(upper - x_arr))


def mid_shell_fraction(x: ArrayLike, magic_numbers: np.ndarray = MAGIC_NUMBERS) -> np.ndarray:
    """
    Mid-shell proxy.

    Approximately:
    - 0 near shell closures;
    - 1 near the middle of a shell.
    """
    x_arr = _to_float_array(x)
    lower, upper = magic_bounds(x_arr, magic_numbers)

    width = np.maximum(upper - lower, EPS)
    value = 4.0 * (x_arr - lower) * (upper - x_arr) / (width**2 + EPS)

    return np.clip(value, 0.0, 1.0)


# ============================================================
# 1. Nuclear structure features from N and Z
# ============================================================

def make_nuclear_structure_features(
    n: ArrayLike,
    z: ArrayLike,
    *,
    as_dataframe: bool = False,
    prefix: str = "",
):
    """
    Construct common nuclear-structure features from neutron number N
    and proton number Z.

    Parameters
    ----------
    n:
        Neutron number.
    z:
        Proton number.
    as_dataframe:
        If True, return a pandas DataFrame.
        If False, return a dictionary.
    prefix:
        Optional prefix for feature names.

    Returns
    -------
    dict or pandas.DataFrame
        Nuclear-structure feature set.
    """

    n, z = np.broadcast_arrays(_to_float_array(n), _to_float_array(z))

    a = n + z
    a13 = np.cbrt(np.maximum(a, EPS))
    a23 = a13**2

    neutron_excess = n - z
    isospin_asymmetry = _safe_divide(neutron_excess, a)
    isospin_asymmetry_abs = np.abs(isospin_asymmetry)

    n_even = even_indicator(n)
    z_even = even_indicator(z)
    n_odd = 1.0 - n_even
    z_odd = 1.0 - z_even

    is_even_even = n_even * z_even
    is_odd_odd = n_odd * z_odd
    is_odd_a = np.where((_to_int_array(a) % 2) != 0, 1.0, 0.0)

    pair_sign = pairing_sign(n, z)

    lower_magic_n, upper_magic_n = magic_bounds(n)
    lower_magic_z, upper_magic_z = magic_bounds(z)

    shell_width_n = np.maximum(upper_magic_n - lower_magic_n, EPS)
    shell_width_z = np.maximum(upper_magic_z - lower_magic_z, EPS)

    d_n_magic = nearest_magic_distance(n)
    d_z_magic = nearest_magic_distance(z)

    valence_n = valence_number_to_shell_closure(n)
    valence_p = valence_number_to_shell_closure(z)

    valence_n_norm = np.clip(2.0 * valence_n / shell_width_n, 0.0, 1.0)
    valence_p_norm = np.clip(2.0 * valence_p / shell_width_z, 0.0, 1.0)

    valence_product = valence_n * valence_p

    # Casten P-factor:
    # P = NpNn / (Np + Nn)
    # Here valence_p and valence_n are valence proton/neutron numbers.
    casten_p_factor = _safe_divide(valence_product, valence_n + valence_p)

    mid_n = mid_shell_fraction(n)
    mid_z = mid_shell_fraction(z)

    features = {
        # ----------------------------
        # Basic identity features
        # ----------------------------
        "N": n,
        "Z": z,
        "A": a,
        "N_minus_Z": neutron_excess,
        "Z_minus_N": z - n,
        "abs_N_minus_Z": np.abs(neutron_excess),

        # ----------------------------
        # Ratios and isospin features
        # ----------------------------
        "N_over_Z": _safe_divide(n, z),
        "Z_over_N": _safe_divide(z, n),
        "N_over_A": _safe_divide(n, a),
        "Z_over_A": _safe_divide(z, a),
        "isospin_asymmetry": isospin_asymmetry,
        "abs_isospin_asymmetry": isospin_asymmetry_abs,
        "isospin_asymmetry_squared": isospin_asymmetry**2,
        "Tz": 0.5 * neutron_excess,

        # ----------------------------
        # Mass-number power features
        # ----------------------------
        "A_1_3": a13,
        "A_2_3": a23,
        "sqrt_A": _safe_sqrt(a),
        "inv_A": _safe_divide(1.0, a),
        "inv_A_1_3": _safe_divide(1.0, a13),
        "inv_A_2_3": _safe_divide(1.0, a23),
        "inv_sqrt_A": _safe_divide(1.0, _safe_sqrt(a)),
        "log_A": _safe_log(a),

        # ----------------------------
        # Liquid-drop / SEMF-inspired features
        # ----------------------------
        "volume_term": a,
        "surface_term": a23,
        "radius_term": a13,
        "coulomb_ZZ_over_A13": _safe_divide(z * (z - 1.0), a13),
        "coulomb_Z2_over_A13": _safe_divide(z**2, a13),
        "symmetry_term": _safe_divide((n - z) ** 2, a),
        "symmetry_term_normalized": isospin_asymmetry**2,
        "surface_symmetry_proxy": a23 * isospin_asymmetry**2,
        "fissility_Z2_over_A": _safe_divide(z**2, a),
        "charge_density_proxy": _safe_divide(z, a),

        # ----------------------------
        # Parity / pairing features
        # ----------------------------
        "is_even_N": n_even,
        "is_even_Z": z_even,
        "is_odd_N": n_odd,
        "is_odd_Z": z_odd,
        "is_even_even": is_even_even,
        "is_odd_odd": is_odd_odd,
        "is_odd_A": is_odd_a,
        "pairing_sign": pair_sign,
        "pairing_A_minus_1_2": pair_sign * _safe_divide(1.0, _safe_sqrt(a)),
        "pairing_A_minus_3_4": pair_sign * _safe_divide(1.0, np.maximum(a, EPS) ** 0.75),
        "pairing_A_minus_1": pair_sign * _safe_divide(1.0, a),

        # ----------------------------
        # Magic-number / shell features
        # ----------------------------
        "lower_magic_N": lower_magic_n,
        "upper_magic_N": upper_magic_n,
        "lower_magic_Z": lower_magic_z,
        "upper_magic_Z": upper_magic_z,
        "shell_width_N": shell_width_n,
        "shell_width_Z": shell_width_z,
        "distance_to_magic_N": d_n_magic,
        "distance_to_magic_Z": d_z_magic,
        "distance_to_magic_sum": d_n_magic + d_z_magic,
        "distance_to_magic_product": d_n_magic * d_z_magic,
        "valence_neutron_number": valence_n,
        "valence_proton_number": valence_p,
        "valence_neutron_fraction": valence_n_norm,
        "valence_proton_fraction": valence_p_norm,
        "valence_product_NpNn": valence_product,
        "casten_P_factor": casten_p_factor,

        # ----------------------------
        # Mid-shell / collectivity proxy features
        # ----------------------------
        "mid_shell_fraction_N": mid_n,
        "mid_shell_fraction_Z": mid_z,
        "mid_shell_fraction_sum": mid_n + mid_z,
        "mid_shell_fraction_product": mid_n * mid_z,

        # ----------------------------
        # Simple region indicators
        # ----------------------------
        "is_neutron_rich": (n > z).astype(float),
        "is_proton_rich": (z > n).astype(float),
        "is_N_equal_Z": (np.rint(n).astype(int) == np.rint(z).astype(int)).astype(float),
    }

    if prefix:
        features = _prefix_features(features, prefix)

    return _format_output(features, as_dataframe=as_dataframe)


# ============================================================
# 2. Nuclear reaction features from projectile, target, and E
# ============================================================

def make_nuclear_reaction_features(
    n_p: ArrayLike,
    z_p: ArrayLike,
    energy: ArrayLike,
    n_t: ArrayLike,
    z_t: ArrayLike,
    *,
    energy_frame: str = "lab",
    energy_is_per_nucleon: bool = False,
    include_structure_subfeatures: bool = True,
    as_dataframe: bool = False,
):
    """
    Construct common low-energy nuclear-reaction features.

    Parameters
    ----------
    n_p, z_p:
        Projectile neutron number and proton number.
    energy:
        Incident energy in MeV.
        If energy_is_per_nucleon=True, energy is interpreted as MeV/u.
    n_t, z_t:
        Target neutron number and proton number.
    energy_frame:
        "lab" or "cm".
        If "lab", energy is projectile laboratory energy.
        If "cm", energy is center-of-mass energy.
    energy_is_per_nucleon:
        If True, input energy is interpreted as energy per projectile nucleon.
    include_structure_subfeatures:
        If True, include structure features for projectile, target, and compound nucleus.
    as_dataframe:
        If True, return a pandas DataFrame.

    Returns
    -------
    dict or pandas.DataFrame
        Nuclear-reaction feature set.

    Notes
    -----
    Q-value, separation energies, optical-model parameters, resonance energies,
    and widths are not included because they cannot be determined accurately
    from N, Z, and E alone without external nuclear-data tables.
    """

    n_p, z_p, energy, n_t, z_t = np.broadcast_arrays(
        _to_float_array(n_p),
        _to_float_array(z_p),
        _to_float_array(energy),
        _to_float_array(n_t),
        _to_float_array(z_t),
    )

    a_p = n_p + z_p
    a_t = n_t + z_t

    n_c = n_p + n_t
    z_c = z_p + z_t
    a_c = a_p + a_t

    # Convert input energy.
    energy_total_input = energy * a_p if energy_is_per_nucleon else energy

    frame = energy_frame.lower().strip()
    if frame in {"lab", "laboratory"}:
        energy_lab = energy_total_input
        energy_cm = energy_lab * _safe_divide(a_t, a_p + a_t)
    elif frame in {"cm", "c.m.", "center_of_mass", "center-of-mass"}:
        energy_cm = energy_total_input
        energy_lab = energy_cm * _safe_divide(a_p + a_t, a_t)
    else:
        raise ValueError("energy_frame must be 'lab' or 'cm'.")

    a_p13 = np.cbrt(np.maximum(a_p, EPS))
    a_t13 = np.cbrt(np.maximum(a_t, EPS))
    a_c13 = np.cbrt(np.maximum(a_c, EPS))

    a_p23 = a_p13**2
    a_t23 = a_t13**2
    a_c23 = a_c13**2

    # Reduced mass in atomic-mass-number units.
    reduced_mass_A = _safe_divide(a_p * a_t, a_p + a_t)

    # Geometric radius and cross-section proxies.
    radius_sum_13 = a_p13 + a_t13
    interaction_radius_fm = R0_FM * radius_sum_13
    geometric_cross_section_fm2 = np.pi * interaction_radius_fm**2

    # 1 fm^2 = 10 mb.
    geometric_cross_section_mb = 10.0 * geometric_cross_section_fm2

    charge_product = z_p * z_t

    # Coulomb barrier proxy:
    # V_C ≈ 1.44 Zp Zt / [r0 (Ap^(1/3) + At^(1/3))]
    coulomb_barrier_MeV = COULOMB_CONST_MEV_FM * _safe_divide(
        charge_product,
        interaction_radius_fm,
    )

    has_coulomb_barrier = (charge_product > 0.0).astype(float)

    # Energy relative to Coulomb barrier.
    energy_over_barrier = _safe_divide(energy_cm, coulomb_barrier_MeV)
    energy_minus_barrier = energy_cm - coulomb_barrier_MeV
    barrier_minus_energy = coulomb_barrier_MeV - energy_cm
    above_barrier_indicator = (energy_cm > coulomb_barrier_MeV).astype(float)

    # Smooth classical above-barrier proxy.
    classical_barrier_factor = np.maximum(
        0.0,
        1.0 - _safe_divide(coulomb_barrier_MeV, energy_cm),
    )

    # Dimensionless reduced energy proxy:
    # useful for charged-particle reactions.
    reduced_energy_coulomb = _safe_divide(
        energy_cm * radius_sum_13,
        charge_product,
    )

    # Sommerfeld/Gamow-like proxies.
    # These omit constants and should be treated as symbolic-regression proxies.
    eta_proxy = charge_product * _safe_sqrt(_safe_divide(reduced_mass_A, energy_cm))
    gamow_factor_proxy = np.exp(-np.clip(2.0 * np.pi * eta_proxy, 0.0, 700.0))

    # Momentum / wavelength / angular-momentum proxies.
    momentum_proxy = _safe_sqrt(reduced_mass_A * energy_cm)
    inv_wavelength_proxy = momentum_proxy
    grazing_angular_momentum_proxy = interaction_radius_fm * momentum_proxy

    # Isospin asymmetries.
    i_p = _safe_divide(n_p - z_p, a_p)
    i_t = _safe_divide(n_t - z_t, a_t)
    i_c = _safe_divide(n_c - z_c, a_c)

    features = {
        # ----------------------------
        # Basic projectile / target / compound features
        # ----------------------------
        "N_projectile": n_p,
        "Z_projectile": z_p,
        "A_projectile": a_p,
        "N_target": n_t,
        "Z_target": z_t,
        "A_target": a_t,
        "N_compound": n_c,
        "Z_compound": z_c,
        "A_compound": a_c,

        # ----------------------------
        # Energy features
        # ----------------------------
        "E_input": energy,
        "E_lab": energy_lab,
        "E_cm": energy_cm,
        "E_lab_per_projectile_nucleon": _safe_divide(energy_lab, a_p),
        "E_cm_per_projectile_nucleon": _safe_divide(energy_cm, a_p),
        "E_cm_per_total_nucleon": _safe_divide(energy_cm, a_p + a_t),
        "sqrt_E_cm": _safe_sqrt(energy_cm),
        "inv_sqrt_E_cm": _safe_divide(1.0, _safe_sqrt(energy_cm)),
        "log_E_cm": _safe_log(energy_cm),
        "E_cm_squared": energy_cm**2,

        # ----------------------------
        # Mass / charge combinations
        # ----------------------------
        "reduced_mass_A": reduced_mass_A,
        "mass_asymmetry": _safe_divide(a_t - a_p, a_t + a_p),
        "charge_asymmetry": _safe_divide(z_t - z_p, z_t + z_p),
        "neutron_asymmetry": _safe_divide(n_t - n_p, n_t + n_p),
        "charge_product_ZpZt": charge_product,
        "mass_product_ApAt": a_p * a_t,

        # ----------------------------
        # Isospin combinations
        # ----------------------------
        "isospin_asymmetry_projectile": i_p,
        "isospin_asymmetry_target": i_t,
        "isospin_asymmetry_compound": i_c,
        "isospin_mismatch_target_minus_projectile": i_t - i_p,
        "abs_isospin_mismatch": np.abs(i_t - i_p),

        # ----------------------------
        # Radius / geometry features
        # ----------------------------
        "A_projectile_1_3": a_p13,
        "A_target_1_3": a_t13,
        "A_compound_1_3": a_c13,
        "A_projectile_2_3": a_p23,
        "A_target_2_3": a_t23,
        "A_compound_2_3": a_c23,
        "radius_sum_13": radius_sum_13,
        "interaction_radius_fm": interaction_radius_fm,
        "geometric_cross_section_core": radius_sum_13**2,
        "geometric_cross_section_fm2": geometric_cross_section_fm2,
        "geometric_cross_section_mb": geometric_cross_section_mb,

        # ----------------------------
        # Coulomb-barrier features
        # ----------------------------
        "has_coulomb_barrier": has_coulomb_barrier,
        "coulomb_barrier_MeV": coulomb_barrier_MeV,
        "E_over_coulomb_barrier": energy_over_barrier,
        "E_minus_coulomb_barrier": energy_minus_barrier,
        "coulomb_barrier_minus_E": barrier_minus_energy,
        "above_barrier_indicator": above_barrier_indicator,
        "classical_barrier_factor": classical_barrier_factor,
        "reduced_energy_coulomb": reduced_energy_coulomb,

        # ----------------------------
        # Penetrability / Gamow-like proxies
        # ----------------------------
        "eta_proxy": eta_proxy,
        "gamow_factor_proxy": gamow_factor_proxy,

        # ----------------------------
        # Momentum / angular-momentum proxies
        # ----------------------------
        "momentum_proxy": momentum_proxy,
        "inv_wavelength_proxy": inv_wavelength_proxy,
        "grazing_angular_momentum_proxy": grazing_angular_momentum_proxy,

        # ----------------------------
        # Compound-nucleus liquid-drop proxies
        # ----------------------------
        "compound_coulomb_ZZ_over_A13": _safe_divide(z_c * (z_c - 1.0), a_c13),
        "compound_coulomb_Z2_over_A13": _safe_divide(z_c**2, a_c13),
        "compound_symmetry_term": _safe_divide((n_c - z_c) ** 2, a_c),
        "compound_fissility_Z2_over_A": _safe_divide(z_c**2, a_c),

        # ----------------------------
        # Level-density / Fermi-gas-inspired proxies
        # ----------------------------
        "sqrt_A_compound_E_cm": _safe_sqrt(a_c * energy_cm),
        "sqrt_reduced_mass_E_cm": _safe_sqrt(reduced_mass_A * energy_cm),
        "fermi_gas_exp_argument_proxy": 2.0 * _safe_sqrt(a_c * energy_cm),
    }

    if include_structure_subfeatures:
        projectile_structure = make_nuclear_structure_features(
            n_p,
            z_p,
            as_dataframe=False,
            prefix="projectile_",
        )
        target_structure = make_nuclear_structure_features(
            n_t,
            z_t,
            as_dataframe=False,
            prefix="target_",
        )
        compound_structure = make_nuclear_structure_features(
            n_c,
            z_c,
            as_dataframe=False,
            prefix="compound_",
        )

        features.update(projectile_structure)
        features.update(target_structure)
        features.update(compound_structure)

    return _format_output(features, as_dataframe=as_dataframe)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example 1:
    # Nuclear structure features for 154Sm: Z=62, N=92
    structure_features = make_nuclear_structure_features(
        n=92,
        z=62,
        as_dataframe=True,
    )

    print("Nuclear structure features:")
    print(structure_features.T.head(40))

    # Example 2:
    # alpha + 154Sm reaction
    # alpha: Np=2, Zp=2
    # target 154Sm: Nt=92, Zt=62
    # incident energy: 20 MeV in lab frame
    reaction_features = make_nuclear_reaction_features(
        n_p=2,
        z_p=2,
        energy=20.0,
        n_t=92,
        z_t=62,
        energy_frame="lab",
        energy_is_per_nucleon=False,
        include_structure_subfeatures=True,
        as_dataframe=True,
    )

    print("\nNuclear reaction features:")
    print(reaction_features.T.head(60))