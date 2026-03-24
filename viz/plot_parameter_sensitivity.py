"""
ICS2 Week 6 - LHS Parameter Sensitivity Visualization

Generates iso-risk maps and sensitivity scatter plots from
Latin Hypercube Sampling parametric sweep.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.multi_scale_engine import compute_metal_areas, run_lhs_sweep


def plot_parameter_sensitivity(save_path="assets/lhs_sensitivity.png", dpi=200):
    """Generate LHS sensitivity analysis plots."""

    print("Generating LHS parameter sensitivity plots...")

    areas = compute_metal_areas()
    lhs = run_lhs_sweep(areas, n_samples=800)

    samples = lhs["samples_log10"]
    E_mix = lhs["E_mix"]
    I_corr = lhs["I_corr"]
    CR_Ni = lhs["CR_Ni"]
    metals_order = lhs["metals_order"]

    # Filter out NaN
    valid = np.isfinite(E_mix) & np.isfinite(I_corr) & np.isfinite(CR_Ni)
    samples = samples[valid]
    E_mix = E_mix[valid]
    I_corr = I_corr[valid]
    CR_Ni = CR_Ni[valid]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=dpi)
    fig.suptitle("ICS2 Week 6: LHS Parametric Sensitivity Analysis\n"
                 f"4D Parameter Space | {len(E_mix)} Valid Samples",
                 fontsize=12, fontweight="bold")

    # Panel 1: j0_Ni vs j0_SS304 colored by I_corr
    ax = axes[0, 0]
    sc = ax.scatter(samples[:, 1], samples[:, 2], c=np.log10(np.abs(I_corr) + 1e-20),
                     s=3, alpha=0.6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log10(I_corr) [A]")
    ax.set_xlabel("log10(j0_Ni)")
    ax.set_ylabel("log10(j0_SS304)")
    ax.set_title("Ni vs SS304: Corrosion Current")

    # Panel 2: j0_Au vs j0_Cu colored by E_mix
    ax = axes[0, 1]
    sc = ax.scatter(samples[:, 0], samples[:, 3], c=E_mix,
                     s=3, alpha=0.6, cmap="coolwarm")
    fig.colorbar(sc, ax=ax, label="E_mix [V vs SHE]")
    ax.set_xlabel("log10(j0_Au)")
    ax.set_ylabel("log10(j0_Cu)")
    ax.set_title("Au vs Cu: Mixed Potential")

    # Panel 3: j0_Ni vs CR_Ni (direct relationship)
    ax = axes[0, 2]
    cr_valid = CR_Ni > 0
    if np.any(cr_valid):
        ax.scatter(samples[cr_valid, 1], np.log10(CR_Ni[cr_valid] + 1e-20),
                    s=3, alpha=0.5, color="crimson")
    ax.set_xlabel("log10(j0_Ni)")
    ax.set_ylabel("log10(CR_Ni) [mm/yr]")
    ax.set_title("Ni Exchange Current vs Corrosion Rate")
    ax.grid(True, alpha=0.3)

    # Panel 4: Histogram of mixed potential
    ax = axes[1, 0]
    ax.hist(E_mix, bins=50, color="steelblue", edgecolor="black",
            linewidth=0.3, alpha=0.7)
    ax.set_xlabel("Mixed Potential [V vs SHE]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mixed Potential")
    ax.axvline(x=np.median(E_mix), color="red", linewidth=1.5,
                linestyle="--", label=f"Median: {np.median(E_mix):.3f} V")
    ax.legend(fontsize=8)

    # Panel 5: Histogram of corrosion current
    ax = axes[1, 1]
    log_I = np.log10(np.abs(I_corr) + 1e-20)
    ax.hist(log_I, bins=50, color="darkorange", edgecolor="black",
            linewidth=0.3, alpha=0.7)
    ax.set_xlabel("log10(I_corr) [A]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Corrosion Current")
    ax.axvline(x=np.median(log_I), color="red", linewidth=1.5,
                linestyle="--", label=f"Median: {np.median(log_I):.2f}")
    ax.legend(fontsize=8)

    # Panel 6: Feature importance via rank correlation
    ax = axes[1, 2]
    from scipy.stats import spearmanr
    importances = []
    labels = []
    for i, mk in enumerate(metals_order):
        rho_corr, _ = spearmanr(samples[:, i], np.log10(np.abs(I_corr) + 1e-20))
        importances.append(abs(rho_corr))
        labels.append(f"j0_{mk}")

    bars = ax.barh(labels, importances, color=["#FFD700", "#A0A0A0", "#707070", "#B87333"])
    ax.set_xlabel("|Spearman Correlation| with log10(I_corr)")
    ax.set_title("Parameter Sensitivity Ranking")
    for bar, val in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_parameter_sensitivity()
