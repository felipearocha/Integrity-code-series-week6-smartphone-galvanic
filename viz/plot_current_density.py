"""
ICS2 Week 6 - Evans Diagram and Mixed Potential Visualization

Shows the classic galvanic corrosion diagram with polarization
curves for all four metals intersecting at the mixed potential.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import METALS, T_DEFAULT, F_CONST, R_GAS
from src.butler_volmer import butler_volmer_current
from src.galvanic_coupling import solve_mixed_potential
from src.multi_scale_engine import compute_metal_areas


def plot_current_density(save_path="assets/evans_diagram.png", dpi=200):
    """Generate Evans diagram showing all four metals and mixed potential."""

    print("Generating Evans diagram (polarization curves)...")

    areas = compute_metal_areas()
    E_mix, I_corr, mc = solve_mixed_potential(areas)

    colors = {"Au": "#FFD700", "Ni": "#A0A0A0", "SS304": "#505050", "Cu": "#B87333"}

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=dpi)

    E_range = np.linspace(-0.8, 1.8, 2000)

    for mk in METALS:
        metal = METALS[mk]
        j_arr = np.array([
            butler_volmer_current(E - metal["E_eq"], metal["j0"],
                                   metal["alpha_a"], metal["alpha_c"])
            for E in E_range
        ])

        # Plot anodic branch (positive current)
        j_anodic = np.where(j_arr > 0, j_arr, np.nan)
        ax.semilogy(E_range, j_anodic, color=colors[mk], linewidth=2,
                     label=f"{mk} anodic (E_eq={metal['E_eq']:.3f} V)")

        # Plot cathodic branch (absolute value of negative current)
        j_cathodic = np.where(j_arr < 0, -j_arr, np.nan)
        ax.semilogy(E_range, j_cathodic, color=colors[mk], linewidth=2,
                     linestyle="--")

        # Mark equilibrium potential
        ax.axvline(x=metal["E_eq"], color=colors[mk], linewidth=0.5,
                     linestyle=":", alpha=0.5)

    # Mark mixed potential
    ax.axvline(x=E_mix, color="red", linewidth=2, linestyle="-",
                label=f"E_mix = {E_mix:.4f} V")
    ax.axhline(y=abs(I_corr / sum(areas.values())), color="red",
                linewidth=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Potential [V vs SHE]", fontsize=12)
    ax.set_ylabel("Current Density [A/m^2]", fontsize=12)
    ax.set_title("ICS2 Week 6: Evans Diagram\n"
                 "Galvanic Coupling of Au/Ni/SS304/Cu in 0.5% NaCl Thin Film",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(1e-10, 1e2)
    ax.set_xlim(-0.8, 1.8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # Annotate anode/cathode roles
    for mk, data in mc.items():
        metal = METALS[mk]
        role_text = "ANODE" if data["role"] == "anode" else "CATHODE"
        y_pos = max(abs(data["j_net"]), 1e-9)
        ax.annotate(f"{mk}: {role_text}", xy=(E_mix, y_pos),
                     xytext=(E_mix + 0.15, y_pos * 5),
                     fontsize=8, color=colors[mk],
                     arrowprops=dict(arrowstyle="->", color=colors[mk], lw=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_current_density()
