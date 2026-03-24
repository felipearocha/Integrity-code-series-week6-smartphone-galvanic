"""
ICS2 Week 6 - Hero Visual: 2D Potential Field and Current Density Map

Generates the primary visualization showing:
- Left panel: Electrolyte potential distribution phi(x,y)
- Right panel: Current density magnitude |j|(x,y)
with USB-C port geometry overlay and metal zone labels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from src.config import (NX, NY, DX, DY, PORT_WIDTH, PORT_HEIGHT,
                         METALS, build_metal_zone_map, SIM_PARAMS)
from src.laplace_solver import solve_laplace_2d


def plot_potential_field(save_path="assets/hero_potential_current.png", dpi=200):
    """Generate the hero 2D potential and current density visualization."""

    print("Generating hero visual: potential field + current density...")

    zone_map = build_metal_zone_map()

    phi, j_field, converged, n_iter = solve_laplace_2d(
        zone_map=zone_map,
        charging_bias=SIM_PARAMS["charging_voltage"],
        max_iter=5000,
        tol=1e-4,
    )

    x = np.linspace(0, PORT_WIDTH * 1e3, NX)  # mm
    y = np.linspace(0, PORT_HEIGHT * 1e3, NY)  # mm
    X, Y = np.meshgrid(x, y)

    # Build zone mask for overlay
    zone_numeric = np.zeros((NY, NX))
    zone_colors = {"Au": 1, "Ni": 2, "SS304": 3, "Cu": 4, "insulator": 5}
    for label, val in zone_colors.items():
        zone_numeric[zone_map == label] = val

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=dpi)
    fig.suptitle("ICS2 Week 6: USB-C Charging Port Galvanic Corrosion\n"
                 "2D Electrolyte Potential and Current Density (Thin Film Model)",
                 fontsize=12, fontweight="bold", y=1.02)

    # Left panel: Potential field
    ax1 = axes[0]
    phi_plot = np.ma.masked_where(zone_map != "electrolyte", phi)
    im1 = ax1.pcolormesh(X, Y, phi_plot, cmap="RdBu_r", shading="auto")
    cb1 = fig.colorbar(im1, ax=ax1, label="Potential [V vs SHE]", shrink=0.85)

    # Overlay metal zones
    for mk, color_val in zone_colors.items():
        if mk == "insulator":
            continue
        mask = (zone_map == mk)
        if np.any(mask):
            ax1.contour(X, Y, mask.astype(float), levels=[0.5],
                        colors="black", linewidths=0.5)

    ax1.set_xlabel("Width [mm]")
    ax1.set_ylabel("Height [mm]")
    ax1.set_title("Electrolyte Potential " + r"$\phi(x,y)$")
    ax1.set_aspect("equal")

    # Right panel: Current density
    ax2 = axes[1]
    j_plot = j_field.copy()
    j_plot = np.where(np.isfinite(j_plot), j_plot, 0.0)
    j_plot[j_plot < 1e-10] = 1e-10
    j_plot = np.ma.masked_where(zone_map != "electrolyte", j_plot)

    # Clamp to reasonable range for visualization
    j_plot = np.clip(j_plot, 1e-6, 1e6)

    im2 = ax2.pcolormesh(X, Y, j_plot, cmap="hot_r", shading="auto",
                          norm=LogNorm(vmin=1e-6, vmax=1e3))
    cb2 = fig.colorbar(im2, ax=ax2, label=r"$|j|$ [A/m$^2$]", shrink=0.85)

    for mk in zone_colors:
        if mk == "insulator":
            continue
        mask = (zone_map == mk)
        if np.any(mask):
            ax2.contour(X, Y, mask.astype(float), levels=[0.5],
                        colors="white", linewidths=0.5)

    ax2.set_xlabel("Width [mm]")
    ax2.set_ylabel("Height [mm]")
    ax2.set_title(r"Current Density $|j|(x,y)$")
    ax2.set_aspect("equal")

    # Legend for metal zones
    legend_patches = [
        mpatches.Patch(color="gold", label="Au (contacts)"),
        mpatches.Patch(color="silver", label="Ni (underlayer)"),
        mpatches.Patch(color="gray", label="SS304 (shell)"),
        mpatches.Patch(color="peru", label="Cu (PCB trace)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_potential_field()
