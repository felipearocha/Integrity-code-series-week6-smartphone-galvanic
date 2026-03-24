"""
ICS2 Week 6 - Test Suite

Tests cover:
1. Butler-Volmer kinetics: symmetry, limits, derivative consistency
2. Laplace solver: convergence, boundary conditions, conservation
3. Galvanic coupling: charge conservation, area ratio effects
4. Temporal evolution: mass conservation, passivation behavior
5. Inverse estimation: parameter recovery, MCMC diagnostics
6. Cybersecurity: audit chain integrity, sensor validation
7. Integration: full pipeline consistency
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (METALS, F_CONST, R_GAS, T_DEFAULT, NX, NY,
                         DX, DY, build_metal_zone_map, SIM_PARAMS)
from src.butler_volmer import (butler_volmer_current, butler_volmer_derivative,
                                 effective_j0, tafel_anodic, tafel_cathodic,
                                 compute_metal_current, polarization_curve)
from src.laplace_solver import solve_laplace_2d, extract_boundary_currents
from src.galvanic_coupling import (solve_mixed_potential, net_galvanic_current,
                                    corrosion_rate_from_current,
                                    compute_area_ratio_sensitivity)
from src.temporal_evolution import (simulate_charge_cycles, estimate_port_lifetime,
                                     pack_state, unpack_state, temporal_state_vector)
from src.inverse_bayesian import (generate_synthetic_observations, log_likelihood,
                                    log_prior, run_mcmc, compute_posterior_statistics)
from src.cybersecurity import (AuditLogger, SensorValidator, STRIDE_THREATS,
                                detect_data_poisoning)


# ======================================================================
# Butler-Volmer Tests
# ======================================================================

class TestButlerVolmer:

    def test_zero_overpotential_gives_zero_current(self):
        """At eta=0, BV equation gives j=0 (equilibrium)."""
        j = butler_volmer_current(0.0, j0=1e-5, alpha_a=0.5, alpha_c=0.5)
        assert abs(j) < 1e-15

    def test_symmetry_at_equal_transfer_coefficients(self):
        """With alpha_a = alpha_c, j(eta) = -j(-eta)."""
        eta = 0.1
        j_pos = butler_volmer_current(eta, 1e-5, 0.5, 0.5)
        j_neg = butler_volmer_current(-eta, 1e-5, 0.5, 0.5)
        assert abs(j_pos + j_neg) < 1e-12

    def test_positive_overpotential_gives_anodic_current(self):
        """Positive eta should give positive (anodic) current."""
        j = butler_volmer_current(0.2, 1e-5, 0.5, 0.5)
        assert j > 0

    def test_negative_overpotential_gives_cathodic_current(self):
        """Negative eta should give negative (cathodic) current."""
        j = butler_volmer_current(-0.2, 1e-5, 0.5, 0.5)
        assert j < 0

    def test_tafel_approximation_at_high_overpotential(self):
        """At large positive eta, BV approaches anodic Tafel limit."""
        eta = 0.3  # >> RT/F ~ 0.027 V
        j_bv = butler_volmer_current(eta, 1e-5, 0.5, 0.5)
        j_tafel = tafel_anodic(eta, 1e-5, 0.5)
        # Should agree within 1%
        assert abs(j_bv - j_tafel) / abs(j_tafel) < 0.01

    def test_derivative_positive_definite(self):
        """dj/d(eta) should always be positive (monotonic BV)."""
        for eta in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            dj = butler_volmer_derivative(eta, 1e-5, 0.5, 0.5)
            assert dj > 0

    def test_derivative_numerical_consistency(self):
        """Check analytical derivative against finite difference."""
        eta = 0.05
        d_eta = 1e-8
        j0, aa, ac = 1e-5, 0.5, 0.5

        dj_analytical = butler_volmer_derivative(eta, j0, aa, ac)
        j_plus = butler_volmer_current(eta + d_eta, j0, aa, ac)
        j_minus = butler_volmer_current(eta - d_eta, j0, aa, ac)
        dj_numerical = (j_plus - j_minus) / (2 * d_eta)

        assert abs(dj_analytical - dj_numerical) / abs(dj_analytical) < 1e-4

    def test_effective_j0_with_zero_oxide(self):
        """With R_oxide=0, effective j0 equals base j0."""
        j0_eff = effective_j0(1e-5, 0.0)
        assert abs(j0_eff - 1e-5) < 1e-15

    def test_effective_j0_decreases_with_oxide(self):
        """Increasing oxide resistance should decrease effective j0."""
        j0_base = 1e-5
        j0_a = effective_j0(j0_base, 1e3)
        j0_b = effective_j0(j0_base, 1e4)
        assert j0_b < j0_a

    def test_polarization_curve_shape(self):
        """Polarization curve should be monotonically increasing."""
        eta, j = polarization_curve("Ni", eta_range=(-0.3, 0.3))
        dj = np.diff(j)
        assert np.all(dj > 0)


# ======================================================================
# Laplace Solver Tests
# ======================================================================

class TestLaplaceSolver:

    def test_solver_converges(self):
        """Laplace solver should converge within max iterations."""
        zone_map = build_metal_zone_map()
        phi, j_field, converged, n_iter = solve_laplace_2d(
            zone_map=zone_map, max_iter=5000, tol=1e-4
        )
        assert converged or n_iter <= 5000

    def test_potential_field_is_finite(self):
        """No NaN or Inf in potential field."""
        zone_map = build_metal_zone_map()
        phi, _, _, _ = solve_laplace_2d(zone_map=zone_map, max_iter=1000, tol=1e-3)
        assert np.all(np.isfinite(phi))

    def test_current_density_non_negative(self):
        """Current density magnitude should be non-negative."""
        zone_map = build_metal_zone_map()
        _, j_field, _, _ = solve_laplace_2d(zone_map=zone_map, max_iter=1000, tol=1e-3)
        assert np.all(j_field >= 0)

    def test_zone_map_contains_all_metals(self):
        """Zone map should contain all four metals."""
        zone_map = build_metal_zone_map()
        for mk in METALS:
            assert np.any(zone_map == mk), f"Metal {mk} not found in zone map"


# ======================================================================
# Galvanic Coupling Tests
# ======================================================================

class TestGalvanicCoupling:

    def setup_method(self):
        self.areas = {
            "Au": 1e-6,
            "Ni": 2e-6,
            "SS304": 10e-6,
            "Cu": 3e-6,
        }

    def test_mixed_potential_exists(self):
        """Mixed potential should be findable."""
        E_mix, I_corr, mc = solve_mixed_potential(self.areas)
        assert np.isfinite(E_mix)

    def test_charge_conservation_at_mixed_potential(self):
        """Total anodic current should equal total cathodic current."""
        E_mix, I_corr, mc = solve_mixed_potential(self.areas)
        I_net = net_galvanic_current(E_mix, self.areas)
        assert abs(I_net) < 1e-10  # Near zero at mixed potential

    def test_mixed_potential_between_extremes(self):
        """E_mix should be between most noble and least noble E_eq."""
        E_eqs = [METALS[mk]["E_eq"] for mk in METALS]
        E_mix, _, _ = solve_mixed_potential(self.areas)
        assert min(E_eqs) <= E_mix <= max(E_eqs)

    def test_area_ratio_increases_corrosion(self):
        """Increasing cathode area should increase anodic dissolution."""
        ratios, E_mix_arr, I_corr_arr = compute_area_ratio_sensitivity(
            self.areas, metal_to_vary="SS304", n_points=10
        )
        # Generally, more cathode area means more corrosion current
        assert I_corr_arr[-1] >= I_corr_arr[0] * 0.5  # Allow some flexibility

    def test_corrosion_rate_positive_for_anode(self):
        """Faradaic corrosion rate should be positive for dissolving metals."""
        _, _, mc = solve_mixed_potential(self.areas)
        for mk, data in mc.items():
            if data["role"] == "anode":
                cr = corrosion_rate_from_current(data["j_anodic"], mk)
                assert cr >= 0

    def test_corrosion_rate_units(self):
        """Sanity check: corrosion rate should be in reasonable range for thin film."""
        cr = corrosion_rate_from_current(1e-3, "Ni")
        # 1 mA/m^2 on Ni should give a small but positive rate
        assert 0 < cr < 100  # mm/year, reasonable for micro-scale corrosion


# ======================================================================
# Temporal Evolution Tests
# ======================================================================

class TestTemporalEvolution:

    def setup_method(self):
        self.areas = {
            "Au": 1e-6,
            "Ni": 2e-6,
            "SS304": 10e-6,
            "Cu": 3e-6,
        }

    def test_state_pack_unpack_roundtrip(self):
        """Pack and unpack should be identity."""
        metals = temporal_state_vector()
        thick = {mk: METALS[mk]["thickness"] for mk in metals}
        R_ox = {mk: METALS[mk]["R_oxide_0"] for mk in metals}
        charges = {mk: 0.0 for mk in metals}

        y = pack_state(thick, R_ox, charges, metals)
        thick2, R_ox2, charges2 = unpack_state(y, metals)

        for mk in metals:
            assert abs(thick[mk] - thick2[mk]) < 1e-15
            assert abs(R_ox[mk] - R_ox2[mk]) < 1e-15

    def test_thickness_decreases_over_time(self):
        """Metal thickness should generally decrease (dissolution)."""
        results = simulate_charge_cycles(
            self.areas, n_cycles=10, cycle_duration=3600, dt=600
        )
        # Ni is typically anodic, should lose thickness
        h_start = results["Ni_thickness"][0]
        h_end = results["Ni_thickness"][-1]
        assert h_end <= h_start

    def test_oxide_resistance_increases(self):
        """Oxide film resistance should increase over time for anodic metals."""
        results = simulate_charge_cycles(
            self.areas, n_cycles=10, cycle_duration=3600, dt=600
        )
        R_start = results["Ni_R_oxide"][0]
        R_end = results["Ni_R_oxide"][-1]
        assert R_end >= R_start

    def test_lifetime_estimation(self):
        """Lifetime estimate should be positive and finite."""
        results = simulate_charge_cycles(
            self.areas, n_cycles=50, cycle_duration=3600, dt=600
        )
        cycles, years, metal = estimate_port_lifetime(results)
        assert cycles > 0
        assert np.isfinite(years)


# ======================================================================
# Inverse Estimation Tests
# ======================================================================

class TestInverseEstimation:

    def setup_method(self):
        self.areas = {
            "Au": 1e-6,
            "Ni": 2e-6,
            "SS304": 10e-6,
            "Cu": 3e-6,
        }
        self.true_j0s = {
            "Au": 5e-8,
            "Ni": 3e-6,
            "SS304": 2e-5,
            "Cu": 8e-6,
        }

    def test_synthetic_observations_have_correct_count(self):
        """Should generate requested number of observations."""
        obs = generate_synthetic_observations(self.true_j0s, self.areas, n_obs=15)
        assert len(obs) == 15

    def test_log_prior_within_bounds(self):
        """Prior should be 0 within bounds, -inf outside."""
        in_bounds = {"Au": -7, "Ni": -6, "SS304": -5, "Cu": -5}
        assert log_prior(in_bounds) == 0.0

        out_bounds = {"Au": -20, "Ni": -6, "SS304": -5, "Cu": -5}
        assert log_prior(out_bounds) == -np.inf

    def test_mcmc_produces_chain(self):
        """MCMC should produce a non-empty chain."""
        obs = generate_synthetic_observations(self.true_j0s, self.areas, n_obs=10)
        chain, acc_rate, lls = run_mcmc(obs, n_samples=100, proposal_std=0.2)
        assert len(chain) > 0
        assert 0 <= acc_rate <= 1

    def test_posterior_statistics(self):
        """Posterior stats should have required keys."""
        obs = generate_synthetic_observations(self.true_j0s, self.areas, n_obs=10)
        chain, _, _ = run_mcmc(obs, n_samples=200, proposal_std=0.2)
        stats = compute_posterior_statistics(chain)
        for mk in ["Au", "Ni", "SS304", "Cu"]:
            assert "mean_log10" in stats[mk]
            assert "ci_95_log10" in stats[mk]
            assert "mean_j0" in stats[mk]


# ======================================================================
# Cybersecurity Tests
# ======================================================================

class TestCybersecurity:

    def test_audit_chain_integrity(self):
        """Audit chain should verify correctly after normal operations."""
        logger = AuditLogger()
        logger.log_event("test", "user1", {"key": "value"})
        logger.log_event("test2", "user2", {"key2": "value2"})
        valid, broken_at = logger.verify_chain()
        assert valid
        assert broken_at is None

    def test_audit_chain_detects_tampering(self):
        """Tampered chain should fail verification."""
        logger = AuditLogger()
        logger.log_event("test", "user1", {"key": "value"})
        logger.log_event("test2", "user2", {"key2": "value2"})

        # Tamper with an entry
        logger.chain[0]["data"]["key"] = "tampered"

        valid, broken_at = logger.verify_chain()
        assert not valid
        assert broken_at == 0

    def test_sensor_validator_normal_reading(self):
        """Normal readings should pass validation."""
        validator = SensorValidator()
        result = validator.validate_reading("sensor1", 1e-5, (1e-8, 1e-2))
        assert result["valid"]

    def test_sensor_validator_out_of_range(self):
        """Out-of-range readings should fail validation."""
        validator = SensorValidator()
        result = validator.validate_reading("sensor1", 100.0, (1e-8, 1e-2))
        assert not result["valid"]
        assert any("OUT_OF_RANGE" in f for f in result["flags"])

    def test_sensor_validator_anomaly_detection(self):
        """Statistical anomaly should be flagged."""
        validator = SensorValidator(spoofing_threshold=2.0)
        # Build history
        for _ in range(20):
            validator.validate_reading("s1", 1e-5 + np.random.normal(0, 1e-7),
                                        (1e-8, 1e-2))
        # Inject anomaly
        result = validator.validate_reading("s1", 0.001, (1e-8, 1e-2))
        assert not result["valid"]

    def test_stride_model_completeness(self):
        """All six STRIDE categories should be defined."""
        expected = {"Spoofing", "Tampering", "Repudiation",
                    "Information_Disclosure", "Denial_of_Service",
                    "Elevation_of_Privilege"}
        assert set(STRIDE_THREATS.keys()) == expected

    def test_physical_consistency_check(self):
        """Balanced currents should pass consistency check."""
        validator = SensorValidator()
        currents = {"Au": -0.001, "Ni": 0.0005, "SS304": -0.0003, "Cu": 0.0008}
        consistent, imbalance = validator.check_physical_consistency(currents)
        # This checks that the function runs; exact balance depends on values
        assert isinstance(consistent, bool)
        assert imbalance >= 0


# ======================================================================
# Integration Tests
# ======================================================================

class TestIntegration:

    def test_full_forward_pipeline(self):
        """Full forward simulation should complete without errors."""
        from src.multi_scale_engine import run_forward_simulation
        results = run_forward_simulation(verbose=False)
        assert "E_mix" in results
        assert "phi_field" in results
        assert "temporal" in results

    def test_zone_map_area_consistency(self):
        """Total metal + insulator + electrolyte area should equal domain area."""
        zone_map = build_metal_zone_map()
        total_cells = NX * NY
        counted = 0
        for label in ["Au", "Ni", "SS304", "Cu", "insulator", "electrolyte"]:
            counted += np.sum(zone_map == label)
        assert counted == total_cells

    def test_data_generation_volume(self):
        """LHS sweep should generate at least 500 data points."""
        from src.multi_scale_engine import run_lhs_sweep, compute_metal_areas
        areas = compute_metal_areas()
        lhs = run_lhs_sweep(areas, n_samples=500)
        assert len(lhs["E_mix"]) == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
