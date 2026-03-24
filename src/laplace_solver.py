"""
ICS2 Week 6 - 2D Laplace Equation Solver for Thin Film Electrolyte

Solves:  div(kappa * grad(phi)) = 0  in 2D
with nonlinear Butler-Volmer boundary conditions at metal surfaces
and zero-flux conditions at insulator boundaries.

Numerical method: Finite difference with iterative Gauss-Seidel
and linearized BV boundary update (Picard iteration).

Domain: USB-C receptacle cross-section (8.34 mm x 2.56 mm).
"""

import numpy as np
from .config import (NX, NY, DX, DY, METALS, ELECTROLYTE,
                     SIM_PARAMS, T_DEFAULT, F_CONST, R_GAS)
from .butler_volmer import butler_volmer_current, butler_volmer_derivative, effective_j0
from .config import build_metal_zone_map


def solve_laplace_2d(zone_map=None, kappa=None, phi_metals=None,
                     R_oxides=None, T=T_DEFAULT, j0_overrides=None,
                     charging_bias=0.0, tol=None, max_iter=None):
    """
    Solve 2D Laplace equation in the electrolyte domain with
    Butler-Volmer boundary conditions.

    Parameters
    ----------
    zone_map : ndarray of str, shape (NY, NX)
        Material zone labels. If None, builds default.
    kappa : float
        Electrolyte conductivity [S/m]. If None, uses default.
    phi_metals : dict
        Fixed metal potentials {metal_key: potential_V}. If None, uses E_eq.
    R_oxides : dict
        Oxide film resistances {metal_key: R_Ohm_m2}. If None, uses initial.
    T : float
        Temperature [K].
    j0_overrides : dict
        Override exchange current densities {metal_key: j0_value}.
    charging_bias : float
        Additional voltage applied to Au contact pins during charging [V].
    tol : float
        Convergence tolerance for potential field.
    max_iter : int
        Maximum Picard/Gauss-Seidel iterations.

    Returns
    -------
    phi : ndarray, shape (NY, NX)
        Electrolyte potential field [V].
    j_field : ndarray, shape (NY, NX)
        Current density magnitude at each point [A/m^2].
    converged : bool
        Whether the solver converged.
    n_iter : int
        Number of iterations to convergence.
    """
    if zone_map is None:
        zone_map = build_metal_zone_map()
    if kappa is None:
        kappa = ELECTROLYTE["conductivity"]
    if phi_metals is None:
        phi_metals = {k: METALS[k]["E_eq"] for k in METALS}
    if R_oxides is None:
        R_oxides = {k: METALS[k]["R_oxide_0"] for k in METALS}
    if j0_overrides is None:
        j0_overrides = {}
    if tol is None:
        tol = SIM_PARAMS["laplace_tol"]
    if max_iter is None:
        max_iter = SIM_PARAMS["laplace_max_iter"]

    # Apply charging bias to Au pins
    if charging_bias != 0.0:
        phi_metals = dict(phi_metals)
        phi_metals["Au"] = phi_metals.get("Au", METALS["Au"]["E_eq"]) + charging_bias

    # Initialize potential field
    phi = np.zeros((NY, NX))

    # Set initial guess: average of metal potentials weighted by area
    metal_pots = [v for v in phi_metals.values()]
    phi[:, :] = np.mean(metal_pots)

    # Identify electrolyte cells
    is_electrolyte = (zone_map == "electrolyte")

    # Precompute metal boundary info
    metal_cells = {}
    for mk in METALS:
        mask = (zone_map == mk)
        if np.any(mask):
            metal_cells[mk] = mask

    # Set metal boundary potentials as fixed (Dirichlet-like via BV coupling)
    for mk, mask in metal_cells.items():
        phi[mask] = phi_metals.get(mk, METALS[mk]["E_eq"])

    # Iterative solver
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        phi_old = phi.copy()

        # Gauss-Seidel update for interior electrolyte cells
        for j in range(1, NY - 1):
            for i in range(1, NX - 1):
                if not is_electrolyte[j, i]:
                    continue

                # Standard 5-point stencil for Laplace equation
                # (phi_i+1 + phi_i-1)/dx^2 + (phi_j+1 + phi_j-1)/dy^2 = 0
                # => phi_ij = weighted average
                coeff_x = 1.0 / (DX * DX)
                coeff_y = 1.0 / (DY * DY)
                denom = 2.0 * (coeff_x + coeff_y)

                phi[j, i] = (coeff_x * (phi[j, i + 1] + phi[j, i - 1])
                              + coeff_y * (phi[j + 1, i] + phi[j - 1, i])) / denom

        # Update metal boundary cells using Butler-Volmer flux matching
        for mk, mask in metal_cells.items():
            metal = METALS[mk]
            j0 = j0_overrides.get(mk, metal["j0"])
            j0_eff = effective_j0(j0, R_oxides.get(mk, metal["R_oxide_0"]), T)
            phi_m = phi_metals.get(mk, metal["E_eq"])

            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                # Find adjacent electrolyte cell
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < NY and 0 <= nc < NX:
                        if is_electrolyte[nr, nc]:
                            neighbors.append(phi[nr, nc])

                if len(neighbors) > 0:
                    phi_elec_local = np.mean(neighbors)
                    eta = phi_m - phi_elec_local - metal["E_eq"]

                    # BV current at this point
                    j_bv = butler_volmer_current(
                        eta, j0_eff, metal["alpha_a"], metal["alpha_c"], T
                    )

                    # Update boundary phi to enforce flux continuity:
                    # kappa * (phi_elec - phi_boundary) / dn = j_bv
                    # This is a simplified linearized update
                    dn = min(DX, DY)
                    phi[r, c] = phi_elec_local - j_bv * dn / kappa

        # Check convergence
        max_change = np.max(np.abs(phi - phi_old))
        n_iter = iteration + 1

        if max_change < tol:
            converged = True
            break

    # Compute current density field from potential gradient
    j_field = np.zeros((NY, NX))
    for j_idx in range(1, NY - 1):
        for i_idx in range(1, NX - 1):
            if is_electrolyte[j_idx, i_idx]:
                dphidx = (phi[j_idx, i_idx + 1] - phi[j_idx, i_idx - 1]) / (2.0 * DX)
                dphidy = (phi[j_idx + 1, i_idx] - phi[j_idx - 1, i_idx]) / (2.0 * DY)
                # Clamp to prevent overflow
                dphidx = np.clip(dphidx, -1e10, 1e10)
                dphidy = np.clip(dphidy, -1e10, 1e10)
                j_field[j_idx, i_idx] = kappa * np.sqrt(dphidx**2 + dphidy**2)

    return phi, j_field, converged, n_iter


def extract_boundary_currents(phi, zone_map, kappa=None, R_oxides=None,
                              T=T_DEFAULT, j0_overrides=None):
    """
    Extract current density at each metal boundary from the solved potential field.

    Returns
    -------
    boundary_currents : dict
        {metal_key: {'total_current': float, 'avg_j': float, 'max_j': float,
                     'area': float, 'j_values': list}}
    """
    if kappa is None:
        kappa = ELECTROLYTE["conductivity"]
    if zone_map is None:
        zone_map = build_metal_zone_map()
    if R_oxides is None:
        R_oxides = {k: METALS[k]["R_oxide_0"] for k in METALS}
    if j0_overrides is None:
        j0_overrides = {}

    is_electrolyte = (zone_map == "electrolyte")
    boundary_currents = {}
    cell_area = DX * DY

    for mk in METALS:
        mask = (zone_map == mk)
        if not np.any(mask):
            boundary_currents[mk] = {
                "total_current": 0.0, "avg_j": 0.0, "max_j": 0.0,
                "area": 0.0, "j_values": []
            }
            continue

        metal = METALS[mk]
        j0 = j0_overrides.get(mk, metal["j0"])
        j0_eff = effective_j0(j0, R_oxides.get(mk, metal["R_oxide_0"]), T)

        rows, cols = np.where(mask)
        j_values = []

        for r, c in zip(rows, cols):
            # Normal flux from adjacent electrolyte cell
            neighbors_phi = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < NY and 0 <= nc < NX:
                    if is_electrolyte[nr, nc]:
                        neighbors_phi.append(phi[nr, nc])

            if len(neighbors_phi) > 0:
                phi_elec = np.mean(neighbors_phi)
                # Current from flux: j = -kappa * grad(phi) dot n
                dn = min(DX, DY)
                j_local = kappa * (phi_elec - phi[r, c]) / dn
                j_values.append(j_local)

        j_values = np.array(j_values) if len(j_values) > 0 else np.array([0.0])
        total_area = len(rows) * cell_area

        boundary_currents[mk] = {
            "total_current": np.sum(j_values) * cell_area,
            "avg_j": np.mean(j_values),
            "max_j": np.max(np.abs(j_values)),
            "area": total_area,
            "j_values": j_values.tolist(),
        }

    return boundary_currents
