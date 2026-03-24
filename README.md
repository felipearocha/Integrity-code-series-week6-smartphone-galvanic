# INTEGRITY CODE SERIES - WEEK 6
# Multi-Scale Galvanic Corrosion in Smartphone Charging Ports
# Dissimilar Metal Electrochemistry with Inverse Parameter Estimation

## Executive Summary

Every year, millions of smartphones fail due to corrosion at the USB-C charging port.
The port contains at least four dissimilar metals (Au plating on contacts, Ni underlayer,
stainless steel shell, Cu alloy traces on PCB) in a geometry where sweat, humidity, and
pocket lint create thin electrolyte films. Under charging bias (5V-20V USB PD), the
system becomes an unintentional electrochemical cell with spatially varying current
density, potential distribution, and material dissolution rates.

This project builds a multi-scale galvanic corrosion simulator that couples:
1. Macro-scale: Laplace equation for potential distribution in the thin electrolyte film
2. Micro-scale: Butler-Volmer kinetics at each metal interface
3. Temporal: Faradaic dissolution and oxide film growth over charge cycles
4. Inverse: Bayesian estimation of exchange current densities from sparse impedance data

## Industrial Decision Improved

Consumer electronics reliability engineering. This model enables:
- Quantitative prediction of port lifetime under humid tropical conditions
- Design optimization of plating stack thickness (Au/Ni/Cu) vs. cost
- Warranty cost estimation based on regional humidity and user behavior
- Go/no-go decisions on connector material substitution

## Failure Consequence Mitigated

Charging port corrosion causes intermittent charging, data loss during backup,
fire risk from short circuits through corroded contacts, and warranty claims.
Apple iPhone 15 USB-C port corrosion complaints are well documented in public forums.

## Economic Exposure

At 1.2 billion smartphones shipped annually with an estimated 2-5% experiencing
charging port corrosion within 2 years, and repair costs of 80-150 USD per device,
the global annual exposure is on the order of 2-9 billion USD.

## Escalation from Week 5

Week 5: 1D spatial PDE (chainage), single-phase corrosion kinetics, forward LHS sweep.
Week 6: 2D spatial PDE (Laplace in thin film), multi-metal galvanic coupling,
inverse Bayesian parameter estimation, micro-to-macro scale bridging.

New dimensions: Multi-scale coupling + Inverse parameter estimation.

## Governing Physics

### Scale 1: Macro (Electrolyte Domain) - Laplace Equation

In the thin electrolyte film, assuming electroneutrality and negligible convection:

    div(kappa * grad(phi)) = 0

where:
- phi: electric potential in electrolyte [V]
- kappa: electrolyte conductivity [S/m]

For a thin film of thickness delta_e, this reduces to a 2D problem on the port surface.

### Scale 2: Micro (Metal Interfaces) - Butler-Volmer Kinetics

At each metal surface (boundary condition for the Laplace equation):

    j_k = j_0k * { exp[alpha_a,k * F * eta_k / (R*T)] - exp[-alpha_c,k * F * eta_k / (R*T)] }

where:
- j_k: current density at metal k [A/m^2]
- j_0k: exchange current density of metal k [A/m^2]
- alpha_a,k, alpha_c,k: anodic/cathodic transfer coefficients
- eta_k = phi_metal,k - phi_electrolyte - E_eq,k: overpotential [V]
- F = 96485 C/mol, R = 8.314 J/(mol*K), T: temperature [K]

### Scale 3: Temporal Evolution - Faradaic Mass Loss

    dm_k/dt = (M_k / (n_k * F)) * j_anodic,k

where:
- m_k: mass loss per unit area [kg/m^2]
- M_k: molar mass of metal k [kg/mol]
- n_k: electrons transferred per atom dissolved

Thickness loss:

    dh_k/dt = dm_k/dt / rho_k

### Oxide Film Growth (Micro-Scale Feedback)

Protective oxide film resistance increases with time:

    R_oxide,k(t) = R_oxide,0,k + beta_k * integral(j_anodic,k dt)

This feeds back into the effective exchange current density, creating a nonlinear
coupling between scales.

### Galvanic Coupling Constraint

At the mixed potential, total anodic current equals total cathodic current:

    sum_k( A_k * j_anodic,k(phi_couple) ) = sum_k( A_k * j_cathodic,k(phi_couple) )

where A_k is the exposed area of metal k.

## Materials in the System

| Metal       | Role            | E_eq vs SHE [V] | j_0 [A/m^2] | n  | M [g/mol] | rho [kg/m^3] |
|-------------|-----------------|------------------|--------------|----|-----------|--------------|
| Au          | Contact plating | +1.50            | 1e-7         | 3  | 196.97    | 19300        |
| Ni          | Underlayer      | -0.257           | 1e-6         | 2  | 58.69     | 8908         |
| SS 304      | Shell           | -0.10            | 1e-5         | 2  | 55.85     | 7930         |
| Cu (C52100) | PCB trace       | +0.340           | 1e-5         | 2  | 63.55     | 8920         |

Note: j_0 values are assumed order-of-magnitude estimates for thin-film NaCl electrolyte.
These are the parameters targeted by the inverse estimation module.

## Electrolyte Properties

Human sweat proxy: 0.5% NaCl, pH 5.5, conductivity approx 1.5 S/m.
Thin film thickness: 10-500 micrometers (depending on humidity/lint trapping).

## Boundary Conditions

- Laplace domain: 2D rectangular (8.34 mm x 2.56 mm USB-C receptacle cross section)
- Metal boundaries: Butler-Volmer flux condition (nonlinear Neumann)
- Insulator boundaries (plastic housing): zero-flux Neumann (dPhi/dn = 0)
- Under charging bias: additional 5V offset on contact pins

## Assumptions (Explicit)

1. Electrolyte is dilute enough for Laplace approximation (no concentration gradients)
2. Thin film is uniform thickness (no meniscus effects)
3. Temperature is uniform at 35 C (body contact temperature)
4. No convective transport in thin film
5. Oxide films are compact and follow linear growth with charge passed
6. Au dissolution is negligible under normal conditions (included for completeness)
7. beta_k (oxide growth rate) is an assumed constitutive parameter requiring calibration

## Repository Structure

    integrity_code_series_week6_smartphone_galvanic/
    |-- README.md
    |-- requirements.txt
    |-- src/
    |   |-- __init__.py
    |   |-- config.py              # Material properties, geometry, simulation parameters
    |   |-- laplace_solver.py      # 2D Laplace equation FD solver with BV boundary conditions
    |   |-- butler_volmer.py       # Butler-Volmer kinetics for each metal
    |   |-- galvanic_coupling.py   # Mixed potential solver and area-ratio effects
    |   |-- temporal_evolution.py  # Faradaic dissolution and oxide growth ODE
    |   |-- inverse_bayesian.py   # Bayesian inference for j_0 estimation
    |   |-- multi_scale_engine.py # Orchestrator coupling all scales
    |   |-- cybersecurity.py      # STRIDE threat model, audit logging, sensor validation
    |-- tests/
    |   |-- test_butler_volmer.py
    |   |-- test_laplace.py
    |   |-- test_galvanic.py
    |   |-- test_inverse.py
    |   |-- test_temporal.py
    |   |-- test_cybersecurity.py
    |   |-- test_integration.py
    |-- viz/
    |   |-- plot_potential_field.py
    |   |-- plot_current_density.py
    |   |-- plot_dissolution_map.py
    |   |-- plot_inverse_posterior.py
    |   |-- plot_parameter_sensitivity.py
    |   |-- generate_gif.py
    |-- assets/
    |-- notebooks/
    |   |-- week6_walkthrough.ipynb
    |-- linkedin/
    |   |-- post_draft.txt
    |   |-- video_script.txt
    |-- docs/
    |   |-- technical_report.md

## Execution Order

    pip install -r requirements.txt
    python -m pytest tests/ -v
    python -m src.multi_scale_engine
    python -m viz.plot_potential_field
    python -m viz.plot_current_density
    python -m viz.plot_dissolution_map
    python -m viz.plot_inverse_posterior
    python -m viz.plot_parameter_sensitivity
    python -m viz.generate_gif

## Intended ZIP

    integrity_code_series_week6_smartphone_galvanic.zip
