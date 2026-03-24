"""
ICS2 Week 6 - Dissolution Map and Temporal Evolution Plots

Generates:
- Thickness loss over charge cycles for all four metals
- Corrosion rate evolution showing passivation effect
- Area ratio sensitivity plot
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import METALS, SIM_PARAMS
from src.multi_scale_engine import compute_metal_areas
from src.temporal_evolution import simulate_charge_cycles, estimate_port_lifetime
from src.galvanic_coupling import compute_area_ratio_sensitivity


def plot_dissolution_map(save_path="assets/dissolution_temporal.png", dpi=200):
    """Generate temporal evolution and dissolution plots."""

    print("Generating dissolution and temporal evolution plots...")

    areas = compute_metal_areas()

    results = simulate_charge_cycles(
        areas, n_cycles=80, cycle_duration=3600, dt=300
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=dpi)
    fig.suptitle("ICS2 Week 6: Temporal Evolution of Galvanic Corrosion\n"
                 "USB-C Charging Port Multi-Metal System",
                 fontsize=12, fontweight="bold")

    colors = {"Au": "#FFD700", "Ni": "#A0A0A0", "SS304": "#707070", "Cu": "#B87333"}
    metals_order = results["metals_order"]

    # Panel 1: Thickness remaining vs cycles
    ax1 = axes[0, 0]
    for mk in metals_order:
        h = results[f"{mk}_thickness"] * 1e6  # Convert to micrometers
        ax1.plot(results["t_cycles"], h, color=colors[mk],
                 linewidth=2, label=f"{mk} ({METALS[mk]['thickness']*1e6:.1f} um initial)")
    ax1.set_xlabel("Charge Cycles")
    ax1.set_ylabel("Remaining Thickness [um]")
    ax1.set_title("Metal Thickness vs Charge Cycles")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Panel 2: Thickness loss percentage
    ax2 = axes[0, 1]
    for mk in metals_order:
        loss = results[f"{mk}_thickness_loss_pct"]
        ax2.plot(results["t_cycles"], loss, color=colors[mk],
                 linewidth=2, label=mk)
    ax2.axhline(y=80, color="red", linestyle="--", linewidth=1, label="80% failure threshold")
    ax2.set_xlabel("Charge Cycles")
    ax2.set_ylabel("Thickness Loss [%]")
    ax2.set_title("Cumulative Thickness Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Oxide resistance growth (passivation feedback)
    ax3 = axes[1, 0]
    for mk in metals_order:
        R_ox = results[f"{mk}_R_oxide"]
        R_ox_normalized = R_ox / METALS[mk]["R_oxide_0"]
        ax3.plot(results["t_cycles"], R_ox_normalized, color=colors[mk],
                 linewidth=2, label=mk)
    ax3.set_xlabel("Charge Cycles")
    ax3.set_ylabel("Normalized Oxide Resistance (R/R_0)")
    ax3.set_title("Oxide Film Growth (Passivation Feedback)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Area ratio sensitivity
    ax4 = axes[1, 1]
    ratios, E_mix_arr, I_corr_arr = compute_area_ratio_sensitivity(
        areas, metal_to_vary="SS304", n_points=40
    )
    ax4_twin = ax4.twinx()

    l1, = ax4.plot(ratios, I_corr_arr * 1e6, color="crimson", linewidth=2,
                    label="Corrosion Current")
    l2, = ax4_twin.plot(ratios, E_mix_arr, color="navy", linewidth=2,
                         linestyle="--", label="Mixed Potential")

    ax4.set_xlabel("SS304 Shell Area Multiplier")
    ax4.set_ylabel("Galvanic Corrosion Current [uA]", color="crimson")
    ax4_twin.set_ylabel("Mixed Potential [V vs SHE]", color="navy")
    ax4.set_title("Area Ratio Sensitivity (SS304 Shell)")
    ax4.legend(handles=[l1, l2], fontsize=8, loc="center right")
    ax4.grid(True, alpha=0.3)

    # Lifetime annotation
    lifetime_cycles, lifetime_years, critical_metal = estimate_port_lifetime(results)
    fig.text(0.5, -0.02,
             f"Estimated Port Lifetime: {lifetime_cycles:.0f} cycles "
             f"({lifetime_years:.1f} years) | Critical Metal: {critical_metal}",
             ha="center", fontsize=10, fontstyle="italic",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_dissolution_map()
