"""
ICS2 Week 6 - Bayesian Inverse Parameter Estimation

Estimates exchange current densities (j_0) for each metal from
sparse synthetic impedance/current observations using Metropolis-Hastings
MCMC.

The inverse problem:
    Given: noisy observations of galvanic current at known area ratios
    Find:  j_0 for Au, Ni, SS304, Cu

This is justified because:
1. j_0 values in thin-film NaCl are poorly characterized for multi-metal stacks
2. The system is underdetermined with forward-only approaches
3. Bayesian estimation naturally quantifies parameter uncertainty
4. Classical grid search is computationally prohibitive in 4D parameter space

Prior: Uniform in log10(j_0) space within physically plausible ranges.
Likelihood: Gaussian, assuming observation noise is normally distributed.
"""

import numpy as np
from .config import METALS, INVERSE_CONFIG, SIM_PARAMS, T_DEFAULT
from .galvanic_coupling import solve_mixed_potential, net_galvanic_current


def generate_synthetic_observations(true_j0s, areas, n_obs=None,
                                     noise_std=None, T=T_DEFAULT,
                                     seed=None):
    """
    Generate synthetic "experimental" data by running the forward model
    with known true parameters and adding Gaussian noise.

    The observations are galvanic current measurements at varying
    area ratios (simulating different stages of coating wear).

    Parameters
    ----------
    true_j0s : dict
        True exchange current densities {metal_key: j0_value}.
    areas : dict
        Base exposed areas.
    n_obs : int
        Number of observation points.
    noise_std : float
        Relative noise standard deviation.
    T : float
        Temperature [K].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    obs_data : list of dict
        Each entry: {'areas': dict, 'I_measured': float, 'sigma': float}
    """
    if n_obs is None:
        n_obs = INVERSE_CONFIG["n_obs_points"]
    if noise_std is None:
        noise_std = INVERSE_CONFIG["obs_noise_std"]
    if seed is None:
        seed = SIM_PARAMS["random_seed"]

    rng = np.random.RandomState(seed)

    obs_data = []
    for i in range(n_obs):
        # Vary area ratios to create diverse observations
        areas_mod = {}
        for mk in areas:
            # Random area perturbation factor [0.5, 2.0]
            factor = 0.5 + 1.5 * rng.random()
            areas_mod[mk] = areas[mk] * factor

        # Forward solve with true parameters
        E_mix, I_corr, _ = solve_mixed_potential(
            areas_mod, T=T, j0_overrides=true_j0s
        )

        # Add noise
        sigma = max(abs(I_corr) * noise_std, 1e-15)
        I_measured = I_corr + rng.normal(0, sigma)

        obs_data.append({
            "areas": dict(areas_mod),
            "I_measured": I_measured,
            "sigma": sigma,
        })

    return obs_data


def log_likelihood(j0_candidates, obs_data, T=T_DEFAULT):
    """
    Compute log-likelihood of candidate j0 values given observations.

    L = -0.5 * sum_i [ (I_pred_i - I_obs_i)^2 / sigma_i^2 ]
    """
    ll = 0.0
    for obs in obs_data:
        E_mix, I_pred, _ = solve_mixed_potential(
            obs["areas"], T=T, j0_overrides=j0_candidates
        )
        residual = I_pred - obs["I_measured"]
        ll -= 0.5 * (residual / obs["sigma"]) ** 2

    return ll


def log_prior(log10_j0s):
    """
    Uniform prior in log10(j0) space.
    Returns 0 if all within bounds, -inf otherwise.
    """
    bounds = {
        "Au": INVERSE_CONFIG["j0_Au_prior"],
        "Ni": INVERSE_CONFIG["j0_Ni_prior"],
        "SS304": INVERSE_CONFIG["j0_SS304_prior"],
        "Cu": INVERSE_CONFIG["j0_Cu_prior"],
    }

    for mk, val in log10_j0s.items():
        lo, hi = bounds[mk]
        if val < lo or val > hi:
            return -np.inf

    return 0.0


def run_mcmc(obs_data, n_samples=None, T=T_DEFAULT, seed=None,
             proposal_std=0.1):
    """
    Metropolis-Hastings MCMC for Bayesian estimation of j0 parameters.

    Parameters
    ----------
    obs_data : list of dict
        Synthetic observations.
    n_samples : int
        Total MCMC samples.
    T : float
        Temperature [K].
    seed : int
        Random seed.
    proposal_std : float
        Standard deviation of Gaussian proposal in log10 space.

    Returns
    -------
    chain : ndarray, shape (n_accepted, 4)
        MCMC chain in log10(j0) space [Au, Ni, SS304, Cu].
    acceptance_rate : float
        Fraction of proposals accepted.
    log_likelihoods : list
        Log-likelihood at each accepted sample.
    """
    if n_samples is None:
        n_samples = INVERSE_CONFIG.get("n_bayesian_samples",
                                        SIM_PARAMS["n_bayesian_samples"])
    if seed is None:
        seed = SIM_PARAMS["random_seed"]

    rng = np.random.RandomState(seed + 100)
    metals_order = ["Au", "Ni", "SS304", "Cu"]

    # Initialize at prior midpoints
    current_log10 = {}
    for mk in metals_order:
        lo, hi = INVERSE_CONFIG[f"j0_{mk}_prior"]
        current_log10[mk] = 0.5 * (lo + hi)

    current_j0s = {mk: 10.0 ** current_log10[mk] for mk in metals_order}
    current_ll = log_likelihood(current_j0s, obs_data, T)
    current_lp = log_prior(current_log10)

    chain = []
    log_lls = []
    n_accepted = 0

    for step in range(n_samples):
        # Propose new parameters
        proposed_log10 = {}
        for mk in metals_order:
            proposed_log10[mk] = current_log10[mk] + rng.normal(0, proposal_std)

        proposed_lp = log_prior(proposed_log10)
        if proposed_lp == -np.inf:
            continue

        proposed_j0s = {mk: 10.0 ** proposed_log10[mk] for mk in metals_order}
        proposed_ll = log_likelihood(proposed_j0s, obs_data, T)

        # Metropolis criterion
        log_alpha = (proposed_ll + proposed_lp) - (current_ll + current_lp)

        if np.log(rng.random()) < log_alpha:
            current_log10 = proposed_log10
            current_j0s = proposed_j0s
            current_ll = proposed_ll
            current_lp = proposed_lp
            n_accepted += 1

        chain.append([current_log10[mk] for mk in metals_order])
        log_lls.append(current_ll)

        if (step + 1) % 500 == 0:
            rate = n_accepted / (step + 1)
            print(f"  MCMC step {step + 1}/{n_samples}, acceptance rate: {rate:.3f}")

    chain = np.array(chain)
    acceptance_rate = n_accepted / n_samples

    return chain, acceptance_rate, log_lls


def compute_posterior_statistics(chain, burn_in_fraction=0.3):
    """
    Compute posterior mean, median, and credible intervals.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, 4)
        MCMC chain in log10(j0) space.
    burn_in_fraction : float
        Fraction of samples to discard as burn-in.

    Returns
    -------
    stats : dict
        {metal_key: {'mean': float, 'median': float,
                     'ci_95': (float, float), 'std': float}}
        All in log10(j0) space.
    """
    metals_order = ["Au", "Ni", "SS304", "Cu"]
    n_burn = int(len(chain) * burn_in_fraction)
    post_chain = chain[n_burn:]

    stats = {}
    for i, mk in enumerate(metals_order):
        samples = post_chain[:, i]
        stats[mk] = {
            "mean_log10": np.mean(samples),
            "median_log10": np.median(samples),
            "std_log10": np.std(samples),
            "ci_95_log10": (np.percentile(samples, 2.5),
                            np.percentile(samples, 97.5)),
            "mean_j0": 10.0 ** np.mean(samples),
            "ci_95_j0": (10.0 ** np.percentile(samples, 2.5),
                         10.0 ** np.percentile(samples, 97.5)),
        }

    return stats
