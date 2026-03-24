"""
ICS2 Week 6 - Galvanic Coupling Module

Solves for the mixed (galvanic) potential of the multi-metal system.
At steady state, total anodic current = total cathodic current:

    sum_k( A_k * j_anodic,k(E_couple) ) = sum_k( A_k * |j_cathodic,k(E_couple)| )

This is solved using a root-finding algorithm (Brent's method on the
net current function).

Also computes area-ratio effects: when A_cathode >> A_anode, the
anodic dissolution rate accelerates (classic galvanic corrosion driver).
"""

import numpy as np
from scipy.optimize import brentq
from .config import METALS, T_DEFAULT, F_CONST, R_GAS
from .butler_volmer import butler_volmer_current, effective_j0


def net_galvanic_current(E_couple, areas, R_oxides=None, T=T_DEFAULT,
                         j0_overrides=None):
    """
    Compute net current at a candidate coupling potential.

    Parameters
    ----------
    E_couple : float
        Candidate mixed potential [V vs SHE].
    areas : dict
        Exposed area of each metal {metal_key: area_m2}.
    R_oxides : dict
        Oxide film resistances.
    T : float
        Temperature [K].
    j0_overrides : dict
        Override j0 values for inverse estimation.

    Returns
    -------
    I_net : float
        Net current [A]. Zero at the mixed potential.
    """
    if R_oxides is None:
        R_oxides = {k: METALS[k]["R_oxide_0"] for k in METALS}
    if j0_overrides is None:
        j0_overrides = {}

    I_net = 0.0
    for mk in METALS:
        if mk not in areas or areas[mk] <= 0:
            continue
        metal = METALS[mk]
        j0 = j0_overrides.get(mk, metal["j0"])
        j0_eff = effective_j0(j0, R_oxides.get(mk, metal["R_oxide_0"]), T)
        eta = E_couple - metal["E_eq"]
        j = butler_volmer_current(eta, j0_eff, metal["alpha_a"], metal["alpha_c"], T)
        I_net += areas[mk] * j

    return I_net


def solve_mixed_potential(areas, R_oxides=None, T=T_DEFAULT,
                          j0_overrides=None, E_range=(-1.0, 2.0)):
    """
    Find the mixed (galvanic coupling) potential using Brent's root finding.

    Parameters
    ----------
    areas : dict
        {metal_key: exposed_area_m2}
    R_oxides : dict
        Oxide resistances per metal.
    T : float
        Temperature [K].
    j0_overrides : dict
        Override j0 for inverse problems.
    E_range : tuple
        Search range for mixed potential [V].

    Returns
    -------
    E_mix : float
        Mixed potential [V vs SHE].
    I_corr : float
        Corrosion current at mixed potential [A].
    metal_currents : dict
        {metal_key: {'j_net': float, 'j_anodic': float, 'role': str}}
    """
    if R_oxides is None:
        R_oxides = {k: METALS[k]["R_oxide_0"] for k in METALS}
    if j0_overrides is None:
        j0_overrides = {}

    def f(E):
        return net_galvanic_current(E, areas, R_oxides, T, j0_overrides)

    # Verify bracket
    fa = f(E_range[0])
    fb = f(E_range[1])

    if fa * fb > 0:
        # Bracket does not contain root; search adaptively
        E_test = np.linspace(E_range[0], E_range[1], 1000)
        f_test = np.array([f(e) for e in E_test])
        sign_changes = np.where(np.diff(np.sign(f_test)))[0]
        if len(sign_changes) == 0:
            # Return potential at minimum |I_net| as fallback
            idx_min = np.argmin(np.abs(f_test))
            E_mix = E_test[idx_min]
            I_corr = abs(f_test[idx_min])
        else:
            idx = sign_changes[0]
            E_mix = brentq(f, E_test[idx], E_test[idx + 1], xtol=1e-10)
            I_corr = 0.0
    else:
        E_mix = brentq(f, E_range[0], E_range[1], xtol=1e-10)
        I_corr = 0.0

    # Compute individual metal contributions at mixed potential
    metal_currents = {}
    total_anodic = 0.0
    for mk in METALS:
        if mk not in areas or areas[mk] <= 0:
            continue
        metal = METALS[mk]
        j0 = j0_overrides.get(mk, metal["j0"])
        j0_eff = effective_j0(j0, R_oxides.get(mk, metal["R_oxide_0"]), T)
        eta = E_mix - metal["E_eq"]
        j = butler_volmer_current(eta, j0_eff, metal["alpha_a"], metal["alpha_c"], T)

        role = "anode" if j > 0 else "cathode"
        metal_currents[mk] = {
            "j_net": j,
            "j_anodic": max(j, 0.0),
            "I_total": areas[mk] * j,
            "role": role,
            "eta": eta,
        }
        if j > 0:
            total_anodic += areas[mk] * j

    I_corr = total_anodic  # Total anodic current = corrosion current

    return E_mix, I_corr, metal_currents


def compute_area_ratio_sensitivity(base_areas, metal_to_vary="SS304",
                                   ratio_range=(0.1, 20.0), n_points=50,
                                   T=T_DEFAULT):
    """
    Sweep cathode-to-anode area ratio and compute corrosion rate sensitivity.

    This demonstrates the classic galvanic corrosion area ratio effect:
    larger cathode area relative to anode area accelerates anode dissolution.

    Parameters
    ----------
    base_areas : dict
        Base exposed areas.
    metal_to_vary : str
        Metal whose area is varied.
    ratio_range : tuple
        Range of area multipliers.
    n_points : int
        Number of points in sweep.

    Returns
    -------
    ratios : ndarray
        Area multiplier values.
    E_mix_arr : ndarray
        Mixed potential at each ratio.
    I_corr_arr : ndarray
        Corrosion current at each ratio.
    """
    ratios = np.linspace(ratio_range[0], ratio_range[1], n_points)
    E_mix_arr = np.zeros(n_points)
    I_corr_arr = np.zeros(n_points)

    base_area = base_areas[metal_to_vary]

    for i, r in enumerate(ratios):
        areas_mod = dict(base_areas)
        areas_mod[metal_to_vary] = base_area * r
        E_mix, I_corr, _ = solve_mixed_potential(areas_mod, T=T)
        E_mix_arr[i] = E_mix
        I_corr_arr[i] = I_corr

    return ratios, E_mix_arr, I_corr_arr


def corrosion_rate_from_current(j_anodic, metal_key):
    """
    Convert anodic current density to corrosion rate using Faraday's law.

    CR [mm/year] = (j * M) / (n * F * rho) * (3.156e7 s/year) * 1000 mm/m

    Parameters
    ----------
    j_anodic : float
        Anodic current density [A/m^2].
    metal_key : str
        Metal identifier.

    Returns
    -------
    cr_mm_per_year : float
        Corrosion rate [mm/year].
    """
    metal = METALS[metal_key]
    M = metal["M"]
    n = metal["n_electrons"]
    rho = metal["rho"]

    cr_m_per_s = (j_anodic * M) / (n * F_CONST * rho)
    cr_mm_per_year = cr_m_per_s * 3.156e7 * 1e3

    return cr_mm_per_year
