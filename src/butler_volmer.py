"""
ICS2 Week 6 - Butler-Volmer Electrochemical Kinetics Module

Implements Butler-Volmer equation for each metal interface in the
USB-C charging port galvanic system.

Governing equation:
    j = j_0 * { exp[alpha_a * F * eta / (R*T)] - exp[-alpha_c * F * eta / (R*T)] }

where eta = phi_metal - phi_electrolyte - E_eq is the overpotential.
"""

import numpy as np
from .config import F_CONST, R_GAS, T_DEFAULT, METALS


def butler_volmer_current(eta, j0, alpha_a, alpha_c, T=T_DEFAULT):
    """
    Compute current density from Butler-Volmer equation.

    Parameters
    ----------
    eta : float or ndarray
        Overpotential [V]. Positive eta drives anodic (dissolution) current.
    j0 : float
        Exchange current density [A/m^2].
    alpha_a : float
        Anodic transfer coefficient (dimensionless).
    alpha_c : float
        Cathodic transfer coefficient (dimensionless).
    T : float
        Temperature [K].

    Returns
    -------
    j : float or ndarray
        Net current density [A/m^2]. Positive = anodic (dissolution).
    """
    f = F_CONST / (R_GAS * T)
    # Clamp exponent arguments to prevent overflow
    arg_a = np.clip(alpha_a * f * eta, -500, 500)
    arg_c = np.clip(-alpha_c * f * eta, -500, 500)
    j = j0 * (np.exp(arg_a) - np.exp(arg_c))
    return j


def butler_volmer_derivative(eta, j0, alpha_a, alpha_c, T=T_DEFAULT):
    """
    Derivative dj/d(eta) for Newton iteration in the coupled solver.

    dj/d(eta) = j_0 * [ alpha_a*f*exp(alpha_a*f*eta) + alpha_c*f*exp(-alpha_c*f*eta) ]
    """
    f = F_CONST / (R_GAS * T)
    dj = j0 * (alpha_a * f * np.exp(alpha_a * f * eta)
                + alpha_c * f * np.exp(-alpha_c * f * eta))
    return dj


def tafel_anodic(eta, j0, alpha_a, T=T_DEFAULT):
    """
    Anodic Tafel approximation (valid for eta >> RT/F, approx > 70 mV).
    j_anodic = j_0 * exp(alpha_a * F * eta / (R*T))
    """
    f = F_CONST / (R_GAS * T)
    return j0 * np.exp(alpha_a * f * eta)


def tafel_cathodic(eta, j0, alpha_c, T=T_DEFAULT):
    """
    Cathodic Tafel approximation (valid for eta << -RT/F).
    j_cathodic = -j_0 * exp(-alpha_c * F * eta / (R*T))
    """
    f = F_CONST / (R_GAS * T)
    return -j0 * np.exp(-alpha_c * f * eta)


def effective_j0(j0_base, R_oxide, T=T_DEFAULT):
    """
    Effective exchange current density accounting for oxide film resistance.

    The oxide film adds a series resistance that reduces the effective
    exchange current density:

        j0_eff = j0_base / (1 + j0_base * R_oxide * F / (R*T))

    This is a linearized correction valid when j0 * R_oxide is small
    compared to RT/F.

    Parameters
    ----------
    j0_base : float
        Bare metal exchange current density [A/m^2].
    R_oxide : float
        Oxide film specific resistance [Ohm*m^2].
    T : float
        Temperature [K].

    Returns
    -------
    j0_eff : float
        Effective exchange current density [A/m^2].
    """
    thermal_voltage = R_GAS * T / F_CONST  # approx 0.0265 V at 35C
    denominator = 1.0 + j0_base * R_oxide / thermal_voltage
    return j0_base / max(denominator, 1e-30)


def compute_metal_current(metal_key, phi_metal, phi_electrolyte,
                          R_oxide=None, T=T_DEFAULT, j0_override=None):
    """
    Compute current density at a specific metal interface.

    Parameters
    ----------
    metal_key : str
        Key into METALS dict ('Au', 'Ni', 'SS304', 'Cu').
    phi_metal : float
        Metal potential [V].
    phi_electrolyte : float
        Electrolyte potential at the interface [V].
    R_oxide : float, optional
        Current oxide film resistance [Ohm*m^2]. If None, uses initial value.
    T : float
        Temperature [K].
    j0_override : float, optional
        Override exchange current density (for inverse estimation).

    Returns
    -------
    j_net : float
        Net current density [A/m^2].
    j_anodic : float
        Anodic component [A/m^2].
    j_cathodic : float
        Cathodic component [A/m^2].
    """
    metal = METALS[metal_key]
    E_eq = metal["E_eq"]
    j0 = j0_override if j0_override is not None else metal["j0"]
    alpha_a = metal["alpha_a"]
    alpha_c = metal["alpha_c"]

    if R_oxide is None:
        R_oxide = metal["R_oxide_0"]

    j0_eff = effective_j0(j0, R_oxide, T)

    eta = phi_metal - phi_electrolyte - E_eq

    j_net = butler_volmer_current(eta, j0_eff, alpha_a, alpha_c, T)
    j_anod = tafel_anodic(eta, j0_eff, alpha_a, T) if eta > 0.01 else max(j_net, 0.0)
    j_cath = tafel_cathodic(eta, j0_eff, alpha_c, T) if eta < -0.01 else min(j_net, 0.0)

    return j_net, j_anod, j_cath


def polarization_curve(metal_key, eta_range=(-0.5, 0.5), n_points=500, T=T_DEFAULT):
    """
    Generate a polarization curve (Evans diagram data) for a given metal.

    Returns
    -------
    eta_arr : ndarray
        Overpotential values [V].
    j_arr : ndarray
        Current density values [A/m^2].
    """
    metal = METALS[metal_key]
    eta_arr = np.linspace(eta_range[0], eta_range[1], n_points)
    j_arr = butler_volmer_current(
        eta_arr, metal["j0"], metal["alpha_a"], metal["alpha_c"], T
    )
    return eta_arr, j_arr
