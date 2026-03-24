"""
ICS2 Week 6 - Animated GIF V9

CONCEPT: Wide lateral cross-section of a contact pin plating stack,
drawn like a geological stratigraphy diagram.

Three layers with EQUAL VISUAL HEIGHT (not proportional to real thickness):
- Au (top, gold) - surface erodes downward with jagged corrosion front
- Ni (middle, silver) - exposed as Au recedes, then erodes too
- Cu (bottom, brown) - substrate, always present

The corrosion front is a 1D profile (x-position vs height) that
evolves each frame: random pit nucleation, lateral growth, deepening.

Above the metal: electrolyte zone with animated particles:
- Red circles: metal cations (M2+) drifting upward from corroding surface
- Blue circles: anions (Cl-) drifting downward toward surface
- Green arrows: electron flow inside the metal

Oxide crust: dark band that thickens on exposed Ni surface.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
import tempfile
import shutil
from scipy.ndimage import gaussian_filter1d

from src.config import METALS


def generate_gif(save_path="assets/corrosion_evolution.gif", n_frames=90, fps=8):
    print("Generating animated GIF v9 (stratigraphy cross-section)...")

    # Visual layer heights (equal, not proportional to real thickness)
    W = 800  # width in pixels/points
    layer_h = 80  # visual height per layer
    elec_h = 120  # electrolyte zone height
    cu_base_y = 30
    ni_base_y = cu_base_y + layer_h
    au_base_y = ni_base_y + layer_h
    elec_top = au_base_y + layer_h + elec_h
    total_h = elec_top + 30

    # 1D corrosion front profiles (x-position array)
    nx = W
    x = np.linspace(0, 1, nx)

    # Au front: starts at top of Au layer, erodes downward
    au_front = np.full(nx, float(layer_h))  # height above au_base_y
    # Ni front: starts at top of Ni layer
    ni_front = np.full(nx, float(layer_h))  # height above ni_base_y
    # Oxide thickness on Ni
    oxide_thick = np.zeros(nx)

    rng = np.random.RandomState(42)

    # Pre-seed Au weak spots
    for _ in range(15):
        cx = rng.randint(20, nx - 20)
        w = rng.randint(5, 25)
        depth = rng.uniform(0.3, 0.7) * layer_h
        profile = depth * np.exp(-0.5 * ((np.arange(nx) - cx) / (w * 0.5)) ** 2)
        au_front -= profile
    au_front = np.clip(au_front, 0, layer_h)

    # Scratch defects
    for _ in range(4):
        cx = rng.randint(0, nx)
        w = rng.randint(2, 6)
        for i in range(max(0, cx - w), min(nx, cx + w)):
            au_front[i] *= rng.uniform(0.2, 0.5)

    # Dissolution rates calibrated for visual
    dh_au_frame = layer_h / 50.0
    dh_ni_frame = layer_h / 40.0

    tmpdir = tempfile.mkdtemp()
    frame_paths = []

    for fi in range(n_frames):
        cycle = fi * 22

        if fi > 0:
            # Au erosion
            au_frac = au_front / layer_h
            rate = 1.0 + 5.0 * (1.0 - np.clip(au_frac, 0, 1)) ** 1.5
            noise = np.clip(1.0 + 0.4 * rng.randn(nx), 0.2, 3.0)
            au_front -= dh_au_frame * rate * noise
            # Smooth slightly for natural look
            au_front = gaussian_filter1d(au_front, sigma=1.5)
            au_front = np.clip(au_front, 0, layer_h)

            # Lateral pit growth in Au
            au_gone = au_front <= 1.0
            if np.any(au_gone):
                gone_idx = np.where(au_gone)[0]
                for idx in gone_idx:
                    for d in [-3, -2, -1, 1, 2, 3]:
                        ni = idx + d
                        if 0 <= ni < nx and au_front[ni] > 1.0:
                            au_front[ni] -= dh_au_frame * 3.0 * rng.uniform(0.5, 2.0)
                au_front = np.clip(au_front, 0, layer_h)

            # Ni erosion where Au is gone
            ni_exposed_mask = au_front <= 1.0
            if np.any(ni_exposed_mask):
                ox_r = 1.0 / (1.0 + oxide_thick * 0.15)
                ni_frac = ni_front / layer_h
                ni_accel = 1.0 + 3.0 * (1.0 - np.clip(ni_frac, 0, 1)) ** 1.5
                ni_noise = np.clip(1.0 + 0.35 * rng.randn(nx), 0.3, 2.0)
                ni_front[ni_exposed_mask] -= (dh_ni_frame * ni_noise[ni_exposed_mask]
                                               * ox_r[ni_exposed_mask]
                                               * ni_accel[ni_exposed_mask])
                ni_front = gaussian_filter1d(ni_front, sigma=1.0)
                ni_front = np.clip(ni_front, 0, layer_h)
                oxide_thick[ni_exposed_mask] += 0.5

                # Lateral Ni pit growth
                ni_gone = ni_exposed_mask & (ni_front <= 1.0)
                if np.any(ni_gone):
                    gone_idx = np.where(ni_gone)[0]
                    for idx in gone_idx:
                        for d in [-2, -1, 1, 2]:
                            nni = idx + d
                            if 0 <= nni < nx and ni_front[nni] > 1.0 and ni_exposed_mask[nni]:
                                ni_front[nni] -= dh_ni_frame * 2.5 * rng.uniform(0.5, 1.5)
                    ni_front = np.clip(ni_front, 0, layer_h)

        # Stats
        au_pct = 100 * np.mean(au_front > 1.0)
        ni_exposed_pct = 100 * np.mean((au_front <= 1.0) & (ni_front > 1.0))
        cu_exposed_pct = 100 * np.mean((au_front <= 1.0) & (ni_front <= 1.0))

        # === BUILD FRAME ===
        fig, ax = plt.subplots(figsize=(14, 5.5), dpi=120)
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#060a10")

        x_plot = np.linspace(0, 14, nx)

        # Cu layer (always full)
        ax.fill_between(x_plot, cu_base_y / 10, ni_base_y / 10,
                          color="#CD7F32", alpha=0.9)
        ax.text(7, (cu_base_y + ni_base_y) / 20, "Cu SUBSTRATE (35 um)",
                 ha="center", va="center", fontsize=9, color="white",
                 fontweight="bold", fontfamily="monospace", alpha=0.7)

        # Ni layer (eroding from top)
        ni_top_plot = ni_base_y / 10 + ni_front / 10
        ax.fill_between(x_plot, ni_base_y / 10, ni_top_plot,
                          color="#A0A0A8", alpha=0.9)

        # Oxide crust on Ni (dark band at top of Ni where exposed)
        for i in range(nx - 1):
            if au_front[i] <= 1.0 and oxide_thick[i] > 0.5:
                ox_visual = min(oxide_thick[i] * 0.15, 3.0)
                ax.fill_between(x_plot[i:i+2],
                                 ni_top_plot[i:i+2] - ox_visual / 10,
                                 ni_top_plot[i:i+2],
                                 color="#2a3a2a", alpha=0.7)

        # Ni label (only if visible)
        if np.mean(ni_front) > 5:
            ni_mid = ni_base_y / 10 + np.mean(ni_front) / 20
            ax.text(7, ni_mid, "Ni BARRIER (2 um)",
                     ha="center", va="center", fontsize=9, color="white",
                     fontweight="bold", fontfamily="monospace", alpha=0.7)

        # Au layer (eroding from top)
        au_bottom_plot = ni_top_plot  # Au sits on top of whatever Ni remains
        au_top_plot = au_bottom_plot + au_front / 10
        # Only draw Au where it exists
        for i in range(nx - 1):
            if au_front[i] > 1.0:
                ax.fill_between(x_plot[i:i+2],
                                 au_bottom_plot[i:i+2],
                                 au_top_plot[i:i+2],
                                 color="#FFD700", alpha=0.95)

        # Au label
        if au_pct > 10:
            au_mid_y = np.mean(au_bottom_plot + au_front / 20)
            ax.text(7, au_mid_y, "Au PLATING (0.5 um)",
                     ha="center", va="center", fontsize=9, color="#333",
                     fontweight="bold", fontfamily="monospace")

        # Electrolyte zone above
        elec_bottom = np.maximum(au_top_plot, ni_top_plot)
        elec_top_y = max(au_base_y + layer_h + elec_h, 280) / 10

        ax.fill_between(x_plot, elec_bottom, elec_top_y,
                          color="#0a2545", alpha=0.3)

        # === ANIMATED PARTICLES ===
        n_particles = 40 + fi * 2  # More particles as corrosion progresses
        n_particles = min(n_particles, 120)

        rng_p = np.random.RandomState(42 + fi * 7)

        for _ in range(n_particles):
            px = rng_p.uniform(0.5, 13.5)
            py_min = float(np.interp(px, x_plot, elec_bottom)) + 0.2
            py = rng_p.uniform(py_min, elec_top_y - 0.5)

            is_cation = rng_p.random() > 0.45

            if is_cation:
                # Metal cations rising from surface
                color = "#ff5555"
                marker = "o"
                size = rng_p.uniform(15, 35)
                alpha = rng_p.uniform(0.4, 0.8)
                # Drift arrow upward
                ax.annotate("", xy=(px + rng_p.uniform(-0.1, 0.1), py + 0.4),
                             xytext=(px, py),
                             arrowprops=dict(arrowstyle="-|>", color="#ff555566",
                                              lw=0.6))
            else:
                # Anions (Cl-) drifting toward surface
                color = "#5588ff"
                marker = "o"
                size = rng_p.uniform(12, 28)
                alpha = rng_p.uniform(0.3, 0.7)
                ax.annotate("", xy=(px + rng_p.uniform(-0.1, 0.1), py - 0.3),
                             xytext=(px, py),
                             arrowprops=dict(arrowstyle="-|>", color="#5588ff44",
                                              lw=0.5))

            ax.scatter(px, py, s=size, c=color, alpha=alpha, zorder=10,
                        edgecolor="none")

        # Electron flow arrows inside metal
        if fi % 3 == 0:
            arrow_offset = 0
        elif fi % 3 == 1:
            arrow_offset = 0.3
        else:
            arrow_offset = 0.6

        for ex in np.arange(1 + arrow_offset, 13, 2.0):
            ey = (cu_base_y + ni_base_y) / 20
            ax.annotate("", xy=(ex + 0.8, ey), xytext=(ex, ey),
                         arrowprops=dict(arrowstyle="-|>", color="#44ff44",
                                          lw=1.2, alpha=0.25))

        # Electrolyte label
        ax.text(1.0, elec_top_y - 0.8,
                 "ELECTROLYTE (sweat film: 0.5% NaCl, pH 5.5)",
                 fontsize=8, color="#5599cc", fontfamily="monospace",
                 fontstyle="italic", alpha=0.8)

        # Ion legend
        ax.scatter([], [], s=25, c="#ff5555", label="M$^{2+}$ (dissolved metal)")
        ax.scatter([], [], s=20, c="#5588ff", label="Cl$^-$ (aggressive anion)")
        ax.legend(fontsize=7, loc="upper right", facecolor="#0d1117cc",
                   edgecolor="#333", labelcolor="white")

        # === GAUGE BARS at bottom ===
        gauge_y_base = 0.8
        bar_data = [
            ("Au", au_pct, "#FFD700"),
            ("Ni", ni_exposed_pct, "#A0A0A8"),
            ("Cu", cu_exposed_pct, "#CD7F32"),
        ]
        for i, (label, pct, color) in enumerate(bar_data):
            bx = 1.0 + i * 4.5
            # Background
            ax.add_patch(Rectangle((bx, gauge_y_base), 3.5, 0.35,
                                     facecolor="#1a1a1a", edgecolor="#333",
                                     linewidth=0.5, zorder=20))
            # Fill
            ax.add_patch(Rectangle((bx, gauge_y_base), 3.5 * pct / 100, 0.35,
                                     facecolor=color, alpha=0.85, zorder=21))
            ax.text(bx - 0.1, gauge_y_base + 0.17, label,
                     ha="right", va="center", fontsize=8, color=color,
                     fontweight="bold", fontfamily="monospace", zorder=22)
            ax.text(bx + 3.6, gauge_y_base + 0.17, f"{pct:.0f}%",
                     ha="left", va="center", fontsize=7, color="white",
                     fontfamily="monospace", zorder=22)

        # Title
        ax.text(7, elec_top_y + 1.0,
                 f"INTEGRITY CODE SERIES  |  Week 6: Galvanic Corrosion  |  Cycle {cycle}",
                 ha="center", fontsize=11, color="#00d4ff",
                 fontweight="bold", fontfamily="monospace")

        ax.set_xlim(0, 14)
        ax.set_ylim(0, elec_top_y + 2.0)
        ax.axis("off")

        fp = os.path.join(tmpdir, f"f_{fi:04d}.png")
        fig.savefig(fp, dpi=120, facecolor=fig.get_facecolor(), pad_inches=0.1)
        plt.close(fig)
        frame_paths.append(fp)

        if (fi + 1) % 10 == 0:
            print(f"  Frame {fi+1}/{n_frames} | Au:{au_pct:.0f}% "
                  f"Ni:{ni_exposed_pct:.0f}% Cu:{cu_exposed_pct:.0f}%")

    frames = [imageio.imread(fp) for fp in frame_paths]
    final = [frames[0]] * 5 + frames + [frames[-1]] * 7
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, final, fps=fps, loop=0)
    shutil.rmtree(tmpdir)
    print(f"  Saved: {save_path} ({len(final)} frames, {fps} fps)")
    return save_path


if __name__ == "__main__":
    generate_gif()
