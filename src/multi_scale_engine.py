"""
ICS2 Week 6 - Multi-Scale Engine Orchestrator

Couples all simulation components:
1. Macro: 2D Laplace solver for potential distribution
2. Micro: Butler-Volmer kinetics at metal interfaces
3. Temporal: Faradaic dissolution + oxide growth ODE
4. Inverse: Bayesian j0 estimation from synthetic observations

Also runs LHS parametric sweep for sensitivity analysis.
"""

import numpy as np
import json
import os
from scipy.stats import qmc

from .config import (METALS, ELECTROLYTE, SIM_PARAMS, T_DEFAULT,
                     NX, NY, DX, DY, build_metal_zone_map)
from .laplace_solver import solve_laplace_2d, extract_boundary_currents
from .galvanic_coupling import (solve_mixed_potential, compute_area_ratio_sensitivity,
                                 corrosion_rate_from_current)
from .temporal_evolution import simulate_charge_cycles, estimate_port_lifetime
from .inverse_bayesian import (generate_synthetic_observations, run_mcmc,
                                 compute_posterior_statistics)
from .cybersecurity import AuditLogger, SensorValidator, STRIDE_THREATS


def compute_metal_areas(zone_map=None):
    """Compute exposed area of each metal from the zone map."""
    if zone_map is None:
        zone_map = build_metal_zone_map()

    cell_area = DX * DY
    areas = {}
    for mk in METALS:
        n_cells = np.sum(zone_map == mk)
        areas[mk] = n_cells * cell_area

    return areas


def run_forward_simulation(verbose=True):
    """
    Execute the full forward simulation pipeline.

    Returns
    -------
    results : dict
        All simulation outputs.
    """
    results = {}

    if verbose:
        print("=" * 70)
        print("ICS2 WEEK 6: Multi-Scale Galvanic Corrosion Simulation")
        print("System: USB-C Smartphone Charging Port")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Build geometry and compute areas
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/6] Building USB-C port geometry...")

    zone_map = build_metal_zone_map()
    areas = compute_metal_areas(zone_map)

    results["zone_map"] = zone_map
    results["areas"] = areas

    if verbose:
        for mk, a in areas.items():
            print(f"  {mk}: {a*1e6:.4f} mm^2")

    # ------------------------------------------------------------------
    # Step 2: Solve 2D Laplace equation
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2/6] Solving 2D Laplace equation (thin film electrolyte)...")

    phi, j_field, converged, n_iter = solve_laplace_2d(
        zone_map=zone_map,
        charging_bias=SIM_PARAMS["charging_voltage"]
    )

    results["phi_field"] = phi
    results["j_field"] = j_field
    results["laplace_converged"] = converged
    results["laplace_iterations"] = n_iter

    if verbose:
        print(f"  Converged: {converged} in {n_iter} iterations")
        print(f"  Potential range: [{phi.min():.4f}, {phi.max():.4f}] V")
        print(f"  Max current density: {j_field.max():.4e} A/m^2")

    # ------------------------------------------------------------------
    # Step 3: Solve mixed potential (galvanic coupling)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[3/6] Solving mixed potential (galvanic coupling)...")

    E_mix, I_corr, metal_currents = solve_mixed_potential(areas)

    results["E_mix"] = E_mix
    results["I_corr"] = I_corr
    results["metal_currents"] = metal_currents

    if verbose:
        print(f"  Mixed potential: {E_mix:.4f} V vs SHE")
        print(f"  Corrosion current: {I_corr:.4e} A")
        for mk, mc in metal_currents.items():
            cr = corrosion_rate_from_current(mc["j_anodic"], mk)
            print(f"  {mk}: role={mc['role']}, j_net={mc['j_net']:.4e} A/m^2, "
                  f"CR={cr:.4f} mm/yr")

    # ------------------------------------------------------------------
    # Step 4: Temporal evolution (charge cycles)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[4/6] Simulating temporal evolution over charge cycles...")
        print(f"  Cycles: {SIM_PARAMS['n_charge_cycles']}, "
              f"Duration: {SIM_PARAMS['cycle_duration']}s each")

    temporal_results = simulate_charge_cycles(
        areas,
        n_cycles=min(SIM_PARAMS["n_charge_cycles"], 100),  # Limit for speed
        cycle_duration=SIM_PARAMS["cycle_duration"],
    )

    results["temporal"] = temporal_results

    lifetime_cycles, lifetime_years, critical_metal = estimate_port_lifetime(
        temporal_results
    )

    results["lifetime_cycles"] = lifetime_cycles
    results["lifetime_years"] = lifetime_years
    results["critical_metal"] = critical_metal

    if verbose:
        print(f"  Estimated port lifetime: {lifetime_cycles:.0f} cycles "
              f"({lifetime_years:.1f} years)")
        print(f"  Critical metal (fails first): {critical_metal}")

    # ------------------------------------------------------------------
    # Step 5: Area ratio sensitivity
    # ------------------------------------------------------------------
    if verbose:
        print("\n[5/6] Computing area ratio sensitivity (SS304 shell)...")

    ratios, E_mix_sweep, I_corr_sweep = compute_area_ratio_sensitivity(
        areas, metal_to_vary="SS304", n_points=30
    )

    results["area_ratio_sweep"] = {
        "ratios": ratios,
        "E_mix": E_mix_sweep,
        "I_corr": I_corr_sweep,
    }

    if verbose:
        print(f"  Sweep range: {ratios[0]:.1f}x to {ratios[-1]:.1f}x SS304 area")
        print(f"  I_corr range: [{I_corr_sweep.min():.4e}, {I_corr_sweep.max():.4e}] A")

    # ------------------------------------------------------------------
    # Step 6: LHS parametric sweep
    # ------------------------------------------------------------------
    if verbose:
        print("\n[6/6] Running LHS parametric sweep...")

    lhs_results = run_lhs_sweep(areas, n_samples=min(SIM_PARAMS["n_lhs_samples"], 500))
    results["lhs"] = lhs_results

    if verbose:
        print(f"  Samples: {len(lhs_results['E_mix'])}")
        print(f"  E_mix range: [{min(lhs_results['E_mix']):.4f}, "
              f"{max(lhs_results['E_mix']):.4f}] V")

    return results


def run_lhs_sweep(areas, n_samples=500):
    """
    Latin Hypercube Sampling sweep over j0 parameters.

    Varies log10(j0) for all four metals simultaneously.
    """
    metals_order = ["Au", "Ni", "SS304", "Cu"]

    # Define parameter bounds (log10 scale)
    bounds = {
        "Au": (-9, -5),
        "Ni": (-8, -4),
        "SS304": (-7, -3),
        "Cu": (-7, -3),
    }

    sampler = qmc.LatinHypercube(d=4, seed=SIM_PARAMS["random_seed"])
    samples = sampler.random(n=n_samples)

    # Scale to parameter bounds
    l_bounds = [bounds[mk][0] for mk in metals_order]
    u_bounds = [bounds[mk][1] for mk in metals_order]
    scaled = qmc.scale(samples, l_bounds, u_bounds)

    E_mix_arr = np.zeros(n_samples)
    I_corr_arr = np.zeros(n_samples)
    cr_Ni_arr = np.zeros(n_samples)

    for i in range(n_samples):
        j0_override = {mk: 10.0 ** scaled[i, j] for j, mk in enumerate(metals_order)}

        try:
            E_mix, I_corr, mc = solve_mixed_potential(
                areas, j0_overrides=j0_override
            )
            E_mix_arr[i] = E_mix
            I_corr_arr[i] = I_corr

            if "Ni" in mc:
                cr_Ni_arr[i] = corrosion_rate_from_current(mc["Ni"]["j_anodic"], "Ni")
        except Exception:
            E_mix_arr[i] = np.nan
            I_corr_arr[i] = np.nan
            cr_Ni_arr[i] = np.nan

    return {
        "samples_log10": scaled,
        "metals_order": metals_order,
        "E_mix": E_mix_arr,
        "I_corr": I_corr_arr,
        "CR_Ni": cr_Ni_arr,
    }


def run_inverse_estimation(areas, verbose=True):
    """
    Run the Bayesian inverse parameter estimation pipeline.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("INVERSE ESTIMATION: Bayesian j0 Recovery")
        print("=" * 70)

    # True parameters (what we try to recover)
    true_j0s = {
        "Au": 5e-8,
        "Ni": 3e-6,
        "SS304": 2e-5,
        "Cu": 8e-6,
    }

    if verbose:
        print("\nTrue j0 values:")
        for mk, val in true_j0s.items():
            print(f"  {mk}: {val:.2e} A/m^2 (log10 = {np.log10(val):.2f})")

    # Generate synthetic observations
    if verbose:
        print("\nGenerating synthetic observations...")

    obs_data = generate_synthetic_observations(true_j0s, areas, n_obs=20)

    if verbose:
        print(f"  Generated {len(obs_data)} observation points")

    # Run MCMC
    if verbose:
        print("\nRunning MCMC (this may take a moment)...")

    chain, acceptance_rate, log_lls = run_mcmc(
        obs_data, n_samples=2000, proposal_std=0.15
    )

    if verbose:
        print(f"\n  Acceptance rate: {acceptance_rate:.3f}")
        print(f"  Chain length: {len(chain)}")

    # Compute posterior statistics
    stats = compute_posterior_statistics(chain, burn_in_fraction=0.3)

    if verbose:
        print("\nPosterior estimates vs true values:")
        for mk in ["Au", "Ni", "SS304", "Cu"]:
            true_log10 = np.log10(true_j0s[mk])
            est = stats[mk]
            print(f"  {mk}: true={true_log10:.2f}, "
                  f"mean={est['mean_log10']:.2f}, "
                  f"95% CI=[{est['ci_95_log10'][0]:.2f}, {est['ci_95_log10'][1]:.2f}]")

    return {
        "true_j0s": true_j0s,
        "obs_data": obs_data,
        "chain": chain,
        "acceptance_rate": acceptance_rate,
        "log_likelihoods": log_lls,
        "posterior_stats": stats,
    }


def run_cybersecurity_demo(results, verbose=True):
    """
    Demonstrate cybersecurity architecture components.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CYBERSECURITY ARCHITECTURE DEMO")
        print("=" * 70)

    # Initialize audit logger
    logger = AuditLogger()

    # Log simulation events
    logger.log_event(
        "simulation_start", "multi_scale_engine",
        {"n_metals": 4, "grid_size": f"{NX}x{NY}"},
    )

    logger.log_event(
        "parameter_load", "config_module",
        {"j0_Au": METALS["Au"]["j0"], "j0_Ni": METALS["Ni"]["j0"]},
        {"source": "config.py", "calibration_date": "2026-03-23"},
    )

    logger.log_event(
        "prediction_output", "multi_scale_engine",
        {"E_mix": results.get("E_mix", 0), "I_corr": results.get("I_corr", 0)},
    )

    # Verify chain
    valid, broken_at = logger.verify_chain()
    if verbose:
        print(f"\nAudit chain valid: {valid}")
        print(f"Chain entries: {len(logger.chain)}")

    # Sensor validation demo
    validator = SensorValidator(spoofing_threshold=3.0, max_drift_rate=0.1)

    # Normal readings
    for i in range(10):
        val = 1e-5 + np.random.normal(0, 1e-7)
        result = validator.validate_reading("ALT_station_1", val, (1e-8, 1e-2))

    # Anomalous reading (potential spoofing)
    spoofed = validator.validate_reading("ALT_station_1", 0.5, (1e-8, 1e-2))

    if verbose:
        print(f"\nSensor validation demo:")
        print(f"  Spoofed reading flags: {spoofed['flags']}")
        print(f"  Valid: {spoofed['valid']}")

    # Print STRIDE summary
    if verbose:
        print(f"\nSTRIDE Threat Model Summary:")
        for threat_name, threat in STRIDE_THREATS.items():
            print(f"  {threat_name}: {threat['threat'][:80]}...")

    return {
        "audit_chain": logger.get_chain_summary(),
        "stride_threats": list(STRIDE_THREATS.keys()),
        "sensor_validation_demo": spoofed,
    }


# ==============================================================================
# Main execution
# ==============================================================================

if __name__ == "__main__":
    print("Starting ICS2 Week 6 Full Simulation Pipeline...\n")

    # Forward simulation
    results = run_forward_simulation(verbose=True)

    # Inverse estimation
    inverse_results = run_inverse_estimation(
        results["areas"], verbose=True
    )
    results["inverse"] = inverse_results

    # Cybersecurity demo
    cyber_results = run_cybersecurity_demo(results, verbose=True)
    results["cybersecurity"] = cyber_results

    # Save key results
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "E_mix_V": float(results["E_mix"]),
        "I_corr_A": float(results["I_corr"]),
        "lifetime_cycles": float(results["lifetime_cycles"]),
        "lifetime_years": float(results["lifetime_years"]),
        "critical_metal": results["critical_metal"],
        "laplace_converged": results["laplace_converged"],
        "inverse_acceptance_rate": inverse_results["acceptance_rate"],
        "areas_mm2": {k: float(v * 1e6) for k, v in results["areas"].items()},
    }

    with open(os.path.join(output_dir, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"Results saved to {output_dir}/simulation_summary.json")
    print("=" * 70)
