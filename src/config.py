"""
ICS2 Week 6 - Configuration Module
Material properties, geometry, and simulation parameters for
multi-scale galvanic corrosion in USB-C smartphone charging ports.

All values are in SI units unless noted otherwise.
Constitutive parameters flagged with [ASSUMED] require experimental calibration.
"""

import numpy as np

# ==============================================================================
# Physical Constants
# ==============================================================================
F_CONST = 96485.0       # Faraday constant [C/mol]
R_GAS = 8.314           # Universal gas constant [J/(mol*K)]
T_DEFAULT = 308.15      # Default temperature: 35 C body contact [K]

# ==============================================================================
# USB-C Receptacle Geometry (2D cross-section)
# USB-IF specification: 8.34 mm x 2.56 mm opening
# ==============================================================================
PORT_WIDTH = 8.34e-3     # [m]
PORT_HEIGHT = 2.56e-3    # [m]

# Grid resolution for 2D Laplace solver
NX = 84                  # nodes along width (approx 0.1 mm resolution)
NY = 26                  # nodes along height
DX = PORT_WIDTH / (NX - 1)
DY = PORT_HEIGHT / (NY - 1)

# ==============================================================================
# Metal Properties
# Each metal is defined as a dictionary with electrochemical and physical data.
# ==============================================================================

METALS = {
    "Au": {
        "name": "Gold (contact plating)",
        "E_eq": 1.50,           # Equilibrium potential vs SHE [V]
        "j0": 1e-7,             # Exchange current density [A/m^2] [ASSUMED]
        "alpha_a": 0.5,         # Anodic transfer coefficient
        "alpha_c": 0.5,         # Cathodic transfer coefficient
        "n_electrons": 3,       # Electrons per dissolution reaction
        "M": 196.97e-3,         # Molar mass [kg/mol]
        "rho": 19300.0,         # Density [kg/m^3]
        "thickness": 0.5e-6,    # Plating thickness [m] (0.5 um typical)
        "R_oxide_0": 1e4,       # Initial oxide film resistance [Ohm*m^2] [ASSUMED]
        "beta_oxide": 1e-2,     # Oxide growth rate per charge [Ohm*m^2/(C/m^2)] [ASSUMED]
    },
    "Ni": {
        "name": "Nickel (underlayer)",
        "E_eq": -0.257,
        "j0": 1e-6,             # [ASSUMED]
        "alpha_a": 0.5,
        "alpha_c": 0.5,
        "n_electrons": 2,
        "M": 58.69e-3,
        "rho": 8908.0,
        "thickness": 2.0e-6,    # 2 um typical
        "R_oxide_0": 5e3,       # [ASSUMED]
        "beta_oxide": 5e-3,     # [ASSUMED]
    },
    "SS304": {
        "name": "Stainless Steel 304 (shell)",
        "E_eq": -0.10,
        "j0": 1e-5,             # [ASSUMED]
        "alpha_a": 0.5,
        "alpha_c": 0.5,
        "n_electrons": 2,       # Simplified: Fe dissolution dominant
        "M": 55.85e-3,          # Fe molar mass as proxy
        "rho": 7930.0,
        "thickness": 200e-6,    # Shell wall thickness
        "R_oxide_0": 1e3,       # Cr2O3 passive film [ASSUMED]
        "beta_oxide": 2e-3,     # [ASSUMED]
    },
    "Cu": {
        "name": "Copper alloy (PCB trace)",
        "E_eq": 0.340,
        "j0": 1e-5,             # [ASSUMED]
        "alpha_a": 0.5,
        "alpha_c": 0.5,
        "n_electrons": 2,
        "M": 63.55e-3,
        "rho": 8920.0,
        "thickness": 35e-6,     # Standard 1 oz Cu PCB
        "R_oxide_0": 2e3,       # [ASSUMED]
        "beta_oxide": 3e-3,     # [ASSUMED]
    },
}

# ==============================================================================
# Metal zone map on USB-C cross-section
# Defines which grid cells belong to which metal boundary.
# Layout: top/bottom rows = contact pins (Au over Ni over Cu on PCB)
#         left/right columns = SS shell
#         center = electrolyte (thin film)
# ==============================================================================

def build_metal_zone_map(nx=NX, ny=NY):
    """
    Returns a 2D array (ny, nx) where each cell is labeled:
    'Au', 'Ni', 'SS304', 'Cu', 'insulator', or 'electrolyte'.

    Simplified layout based on USB-C receptacle cross-section:
    - Shell (SS304): left 2 cols, right 2 cols, bottom 2 rows
    - Contact pins (Au surface, Ni under, Cu trace below):
      12 pins on top row, 12 pins on bottom row
      Each pin is approximately 0.3 mm wide, spaced 0.5 mm apart
    - Insulator (LCP plastic): gaps between pins and housing
    - Electrolyte: all interior cells not occupied by metal/insulator
    """
    zone_map = np.full((ny, nx), "electrolyte", dtype=object)

    # SS304 shell boundaries
    zone_map[:, :2] = "SS304"          # left wall
    zone_map[:, -2:] = "SS304"         # right wall
    zone_map[:2, :] = "SS304"          # bottom wall

    # Contact pins on top row (y = ny-1 and ny-2)
    pin_width_cells = max(1, int(0.3e-3 / DX))
    pin_spacing_cells = max(1, int(0.5e-3 / DX))
    n_pins = 12

    start_x = 4  # offset from shell wall
    for pin_idx in range(n_pins):
        x_start = start_x + pin_idx * (pin_width_cells + pin_spacing_cells)
        x_end = min(x_start + pin_width_cells, nx - 2)
        if x_end > nx - 3:
            break
        zone_map[ny - 1, x_start:x_end] = "Au"
        zone_map[ny - 2, x_start:x_end] = "Ni"

    # Contact pins on bottom inner row (y = 2 and 3, above SS shell)
    for pin_idx in range(n_pins):
        x_start = start_x + pin_idx * (pin_width_cells + pin_spacing_cells)
        x_end = min(x_start + pin_width_cells, nx - 2)
        if x_end > nx - 3:
            break
        zone_map[3, x_start:x_end] = "Au"
        zone_map[2, x_start:x_end] = "Ni"

    # Cu traces underneath Ni (one row deeper)
    for pin_idx in range(n_pins):
        x_start = start_x + pin_idx * (pin_width_cells + pin_spacing_cells)
        x_end = min(x_start + pin_width_cells, nx - 2)
        if x_end > nx - 3:
            break
        # Cu only where there is space (not in SS304 zone)
        if ny - 3 >= 0 and zone_map[ny - 3, x_start] != "SS304":
            zone_map[ny - 3, x_start:x_end] = "Cu"

    # Mark non-metal, non-electrolyte cells as insulator
    # (gaps between pins on the boundary rows)
    for row in [ny - 1, ny - 2, 2, 3]:
        for col in range(2, nx - 2):
            if zone_map[row, col] == "electrolyte":
                zone_map[row, col] = "insulator"

    return zone_map


# ==============================================================================
# Electrolyte Properties
# ==============================================================================

ELECTROLYTE = {
    "name": "Human sweat proxy (0.5% NaCl)",
    "conductivity": 1.5,        # [S/m]
    "pH": 5.5,
    "film_thickness": 100e-6,   # Default thin film thickness [m]
    "NaCl_concentration": 0.085,  # [mol/L] approx 0.5 wt%
}

# ==============================================================================
# Simulation Parameters
# ==============================================================================

SIM_PARAMS = {
    "n_charge_cycles": 500,       # Number of charge/discharge cycles to simulate
    "cycle_duration": 3600.0,     # Duration of one charge cycle [s] (1 hour)
    "dt_temporal": 60.0,          # Time step for temporal evolution [s]
    "laplace_tol": 1e-6,          # Convergence tolerance for Laplace solver
    "laplace_max_iter": 10000,    # Max iterations for Laplace solver
    "charging_voltage": 5.0,      # USB charging voltage [V]
    "n_lhs_samples": 2000,        # Latin Hypercube samples for parametric sweep
    "n_bayesian_samples": 5000,   # MCMC samples for inverse estimation
    "random_seed": 42,
}

# ==============================================================================
# Inverse Problem Configuration
# ==============================================================================

INVERSE_CONFIG = {
    # Prior ranges for exchange current densities [A/m^2] (log10 scale)
    "j0_Au_prior": (-9, -5),     # log10(j0) range
    "j0_Ni_prior": (-8, -4),
    "j0_SS304_prior": (-7, -3),
    "j0_Cu_prior": (-7, -3),
    # Synthetic observation noise
    "obs_noise_std": 0.05,       # 5% relative noise on synthetic impedance data
    "n_obs_points": 20,          # Number of synthetic observation locations
}

# ==============================================================================
# Cybersecurity Configuration
# ==============================================================================

CYBER_CONFIG = {
    "enable_audit_log": True,
    "hash_algorithm": "sha256",
    "sensor_spoofing_threshold": 3.0,  # standard deviations
    "max_parameter_drift_rate": 0.1,   # max allowable rate of change per cycle
}
