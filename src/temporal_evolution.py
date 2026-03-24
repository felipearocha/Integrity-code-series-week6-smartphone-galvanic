"""
ICS2 Week 6 - Temporal Evolution Module

Couples Faradaic dissolution mass loss with oxide film growth
over multiple charge/discharge cycles.

Governing ODEs:
    dh_k/dt = -(j_anodic,k * M_k) / (n_k * F * rho_k)     (thickness loss)
    dR_oxide,k/dt = beta_k * j_anodic,k                      (oxide growth)

The oxide growth feeds back into effective j0, creating a nonlinear
coupling where initial corrosion is fast, then self-limiting as
protective oxide builds up (passivation effect).
"""

import numpy as np
from scipy.integrate import solve_ivp
from .config import METALS, F_CONST, T_DEFAULT, SIM_PARAMS
from .galvanic_coupling import solve_mixed_potential
from .butler_volmer import effective_j0, butler_volmer_current


def temporal_state_vector(metals_order=None):
    """
    Define the state vector for the ODE system.
    For each metal: [thickness_remaining, R_oxide, cumulative_charge]
    Total states = 3 * n_metals.
    """
    if metals_order is None:
        metals_order = ["Au", "Ni", "SS304", "Cu"]
    return metals_order


def pack_state(thicknesses, R_oxides, charges, metals_order):
    """Pack into a single state vector."""
    y = []
    for mk in metals_order:
        y.extend([thicknesses[mk], R_oxides[mk], charges[mk]])
    return np.array(y)


def unpack_state(y, metals_order):
    """Unpack state vector into dictionaries."""
    thicknesses = {}
    R_oxides = {}
    charges = {}
    for i, mk in enumerate(metals_order):
        thicknesses[mk] = y[3 * i]
        R_oxides[mk] = y[3 * i + 1]
        charges[mk] = y[3 * i + 2]
    return thicknesses, R_oxides, charges


def rhs_temporal(t, y, metals_order, areas, T=T_DEFAULT, charging_active=True):
    """
    Right-hand side of the temporal ODE system.

    Parameters
    ----------
    t : float
        Time [s].
    y : ndarray
        State vector.
    metals_order : list of str
        Metal keys in order.
    areas : dict
        Exposed areas per metal.
    T : float
        Temperature [K].
    charging_active : bool
        Whether charging bias is applied.

    Returns
    -------
    dydt : ndarray
        Time derivatives of state vector.
    """
    thicknesses, R_oxides, charges = unpack_state(y, metals_order)

    # Check for fully dissolved metals (thickness <= 0)
    for mk in metals_order:
        if thicknesses[mk] <= 0:
            thicknesses[mk] = 0.0

    # Solve for mixed potential at current oxide state
    E_mix, I_corr, metal_currents = solve_mixed_potential(
        areas, R_oxides=R_oxides, T=T
    )

    dydt = np.zeros_like(y)

    for i, mk in enumerate(metals_order):
        metal = METALS[mk]

        if thicknesses[mk] <= 0:
            # Metal fully dissolved, no further corrosion
            dydt[3 * i] = 0.0
            dydt[3 * i + 1] = 0.0
            dydt[3 * i + 2] = 0.0
            continue

        if mk in metal_currents:
            j_anodic = metal_currents[mk]["j_anodic"]
        else:
            j_anodic = 0.0

        # Thickness loss rate [m/s]
        dh_dt = -(j_anodic * metal["M"]) / (metal["n_electrons"] * F_CONST * metal["rho"])

        # Oxide growth rate [Ohm*m^2/s]
        dR_oxide_dt = metal["beta_oxide"] * j_anodic

        # Cumulative charge [C/m^2/s]
        dQ_dt = j_anodic

        dydt[3 * i] = dh_dt
        dydt[3 * i + 1] = dR_oxide_dt
        dydt[3 * i + 2] = dQ_dt

    return dydt


def simulate_charge_cycles(areas, n_cycles=None, cycle_duration=None,
                           dt=None, T=T_DEFAULT):
    """
    Simulate the temporal evolution over multiple charge cycles.

    Parameters
    ----------
    areas : dict
        Exposed metal areas.
    n_cycles : int
        Number of charge cycles.
    cycle_duration : float
        Duration per cycle [s].
    dt : float
        Output time step [s].
    T : float
        Temperature [K].

    Returns
    -------
    results : dict
        Time history of all state variables and derived quantities.
    """
    if n_cycles is None:
        n_cycles = SIM_PARAMS["n_charge_cycles"]
    if cycle_duration is None:
        cycle_duration = SIM_PARAMS["cycle_duration"]
    if dt is None:
        dt = SIM_PARAMS["dt_temporal"]

    metals_order = temporal_state_vector()

    # Initial state
    thicknesses_0 = {mk: METALS[mk]["thickness"] for mk in metals_order}
    R_oxides_0 = {mk: METALS[mk]["R_oxide_0"] for mk in metals_order}
    charges_0 = {mk: 0.0 for mk in metals_order}

    y0 = pack_state(thicknesses_0, R_oxides_0, charges_0, metals_order)

    t_total = n_cycles * cycle_duration
    t_eval = np.arange(0, t_total, dt)

    # Solve ODE
    sol = solve_ivp(
        rhs_temporal,
        [0, t_total],
        y0,
        method="RK45",
        t_eval=t_eval,
        args=(metals_order, areas, T),
        rtol=1e-8,
        atol=1e-12,
        max_step=dt,
    )

    if not sol.success:
        print(f"Warning: ODE solver did not converge. Message: {sol.message}")

    # Unpack results
    results = {
        "t": sol.t,
        "t_cycles": sol.t / cycle_duration,
        "metals_order": metals_order,
    }

    for i, mk in enumerate(metals_order):
        results[f"{mk}_thickness"] = sol.y[3 * i]
        results[f"{mk}_R_oxide"] = sol.y[3 * i + 1]
        results[f"{mk}_charge"] = sol.y[3 * i + 2]

        # Compute thickness loss as percentage
        h0 = METALS[mk]["thickness"]
        results[f"{mk}_thickness_loss_pct"] = (
            100.0 * (h0 - sol.y[3 * i]) / h0
        )

    # Compute corrosion rates at each time step
    for mk in metals_order:
        cr_arr = np.zeros(len(sol.t))
        for idx in range(len(sol.t)):
            thick, R_ox, charge = {}, {}, {}
            for j, mk2 in enumerate(metals_order):
                thick[mk2] = sol.y[3 * j, idx]
                R_ox[mk2] = sol.y[3 * j + 1, idx]
                charge[mk2] = sol.y[3 * j + 2, idx]

            _, _, mc = solve_mixed_potential(areas, R_oxides=R_ox, T=T)
            if mk in mc:
                j_a = mc[mk]["j_anodic"]
                metal = METALS[mk]
                cr_m_per_s = (j_a * metal["M"]) / (metal["n_electrons"] * F_CONST * metal["rho"])
                cr_arr[idx] = cr_m_per_s * 3.156e7 * 1e3  # mm/year
            else:
                cr_arr[idx] = 0.0
        results[f"{mk}_corrosion_rate_mmpy"] = cr_arr

    return results


def estimate_port_lifetime(results, critical_thickness_fraction=0.8):
    """
    Estimate charging port lifetime based on when the most vulnerable
    metal (thinnest plating) loses a critical fraction of its thickness.

    Parameters
    ----------
    results : dict
        Output from simulate_charge_cycles.
    critical_thickness_fraction : float
        Fraction of thickness loss triggering failure (0.8 = 80% loss).

    Returns
    -------
    lifetime_cycles : float
        Estimated cycles to failure.
    lifetime_years : float
        Estimated years to failure (assuming 1 cycle/day).
    critical_metal : str
        Metal that fails first.
    """
    metals_order = results["metals_order"]
    min_lifetime = np.inf
    critical_metal = None

    for mk in metals_order:
        loss_pct = results[f"{mk}_thickness_loss_pct"]
        threshold = critical_thickness_fraction * 100.0

        indices = np.where(loss_pct >= threshold)[0]
        if len(indices) > 0:
            cycle_at_failure = results["t_cycles"][indices[0]]
            if cycle_at_failure < min_lifetime:
                min_lifetime = cycle_at_failure
                critical_metal = mk

    if critical_metal is None:
        min_lifetime = results["t_cycles"][-1]
        critical_metal = "none_failed"

    lifetime_years = min_lifetime / 365.25  # 1 cycle per day

    return min_lifetime, lifetime_years, critical_metal
