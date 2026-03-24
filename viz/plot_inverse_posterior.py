"""
ICS2 Week 6 - Inverse Estimation Posterior Visualization

Generates:
- Corner plot of posterior distributions for j0 parameters
- MCMC trace plots for convergence diagnostics
- True vs estimated parameter comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.multi_scale_engine import compute_metal_areas, run_inverse_estimation


def plot_inverse_posterior(save_path="assets/inverse_posterior.png", dpi=200):
    """Generate Bayesian inverse estimation diagnostic plots."""

    print("Generating inverse estimation posterior plots...")

    areas = compute_metal_areas()
    inv = run_inverse_estimation(areas, verbose=True)

    chain = inv["chain"]
    true_j0s = inv["true_j0s"]
    stats = inv["posterior_stats"]
    log_lls = inv["log_likelihoods"]

    metals_order = ["Au", "Ni", "SS304", "Cu"]
    true_log10 = [np.log10(true_j0s[mk]) for mk in metals_order]

    burn_in = int(0.3 * len(chain))
    post_chain = chain[burn_in:]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    fig.suptitle("ICS2 Week 6: Bayesian Inverse Estimation of Exchange Current Densities\n"
                 "Metropolis-Hastings MCMC | 4-Parameter Recovery from Synthetic Galvanic Data",
                 fontsize=12, fontweight="bold")

    colors = ["#FFD700", "#A0A0A0", "#707070", "#B87333"]

    # Row 1: MCMC Trace Plots
    for i, mk in enumerate(metals_order):
        ax = axes[0, i]
        ax.plot(chain[:, i], color=colors[i], alpha=0.5, linewidth=0.3)
        ax.axhline(y=true_log10[i], color="red", linewidth=1.5,
                    linestyle="--", label="True value")
        ax.axvline(x=burn_in, color="black", linewidth=0.8,
                    linestyle=":", label="Burn-in")
        ax.set_ylabel(f"log10(j0_{mk})")
        ax.set_xlabel("MCMC Step")
        ax.set_title(f"{mk} Trace")
        ax.legend(fontsize=7)

    # Row 2: Marginal Posterior Histograms
    for i, mk in enumerate(metals_order):
        ax = axes[1, i]
        ax.hist(post_chain[:, i], bins=40, color=colors[i], alpha=0.7,
                density=True, edgecolor="black", linewidth=0.3)
        ax.axvline(x=true_log10[i], color="red", linewidth=2,
                    linestyle="--", label=f"True: {true_log10[i]:.2f}")
        mean_val = stats[mk]["mean_log10"]
        ax.axvline(x=mean_val, color="blue", linewidth=1.5,
                    linestyle="-", label=f"Mean: {mean_val:.2f}")

        ci = stats[mk]["ci_95_log10"]
        ax.axvspan(ci[0], ci[1], alpha=0.15, color="blue", label="95% CI")

        ax.set_xlabel(f"log10(j0) [{mk}]")
        ax.set_ylabel("Posterior Density")
        ax.set_title(f"{mk} Posterior")
        ax.legend(fontsize=7)

    # Row 3: Pairwise correlations (2 pairs) + convergence + summary
    # Pair 1: Ni vs SS304
    ax_pair1 = axes[2, 0]
    ax_pair1.scatter(post_chain[:, 1], post_chain[:, 2], s=1, alpha=0.3,
                      color="steelblue")
    ax_pair1.axvline(x=true_log10[1], color="red", linewidth=1, linestyle="--")
    ax_pair1.axhline(y=true_log10[2], color="red", linewidth=1, linestyle="--")
    ax_pair1.set_xlabel("log10(j0_Ni)")
    ax_pair1.set_ylabel("log10(j0_SS304)")
    ax_pair1.set_title("Ni vs SS304 Correlation")

    # Pair 2: Au vs Cu
    ax_pair2 = axes[2, 1]
    ax_pair2.scatter(post_chain[:, 0], post_chain[:, 3], s=1, alpha=0.3,
                      color="darkorange")
    ax_pair2.axvline(x=true_log10[0], color="red", linewidth=1, linestyle="--")
    ax_pair2.axhline(y=true_log10[3], color="red", linewidth=1, linestyle="--")
    ax_pair2.set_xlabel("log10(j0_Au)")
    ax_pair2.set_ylabel("log10(j0_Cu)")
    ax_pair2.set_title("Au vs Cu Correlation")

    # Log-likelihood convergence
    ax_conv = axes[2, 2]
    ax_conv.plot(log_lls, color="darkgreen", linewidth=0.5, alpha=0.7)
    ax_conv.axvline(x=burn_in, color="black", linewidth=0.8, linestyle=":")
    ax_conv.set_xlabel("MCMC Step")
    ax_conv.set_ylabel("Log-Likelihood")
    ax_conv.set_title("MCMC Convergence")

    # Summary table
    ax_table = axes[2, 3]
    ax_table.axis("off")
    table_data = []
    for mk in metals_order:
        s = stats[mk]
        table_data.append([
            mk,
            f"{np.log10(true_j0s[mk]):.2f}",
            f"{s['mean_log10']:.2f}",
            f"[{s['ci_95_log10'][0]:.2f}, {s['ci_95_log10'][1]:.2f}]",
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Metal", "True log10(j0)", "Est. Mean", "95% CI"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax_table.set_title("Parameter Recovery Summary", fontsize=10)

    # Acceptance rate annotation
    fig.text(0.5, -0.02,
             f"Acceptance Rate: {inv['acceptance_rate']:.3f} | "
             f"Chain Length: {len(chain)} | "
             f"Burn-in: {burn_in} | "
             f"Post-burn-in samples: {len(post_chain)}",
             ha="center", fontsize=9, fontstyle="italic",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_inverse_posterior()
