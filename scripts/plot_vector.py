import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
from collect_data_vector import n_qubit, repeat, n_points, d_values
from scipy.optimize import curve_fit

matplotlib.rcParams.update({"font.size": 15})

# === Choose plot mode: "single", "dual", or "all" ===
plot_mode = "single"
# ================================================

ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

(
    min_overlap,
    d,
    overlap_estimate,
    actual_overlap,
    toffoli_ratio,
) = np.loadtxt(data_folder / f"ordering_alg_n_{n_qubit}_vector.npy", unpack=True)

# === Custom colors ===
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# === Setup figure based on mode ===
if plot_mode == "all":
    plt.figure(figsize=(13, 14))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
elif plot_mode == "dual":
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax3 = plt.subplot(1, 2, 2)
    ax2 = ax4 = None
else:  # single
    plt.figure(figsize=(7, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax2 = ax3 = ax4 = None


# === Plot for fixed d as a function of min allowed overlap ===
for i, fixed_d in enumerate(d_values):
    fixed_d = int(fixed_d)
    mask_d_fixed = np.isclose(d, fixed_d)
    x1 = min_overlap[mask_d_fixed]

    actual_overlap_y = actual_overlap[mask_d_fixed]
    overlap_estimate_y = overlap_estimate[mask_d_fixed]
    toffoli_ratio_y = toffoli_ratio[mask_d_fixed]

    unique_x1 = np.unique(x1)

    means_actual_overlap = [np.mean(actual_overlap_y[x1 == ux1]) for ux1 in unique_x1]
    stds_actual_overlap = [np.std(actual_overlap_y[x1 == ux1]) for ux1 in unique_x1]

    means_overlap_estimate = [
        np.mean(overlap_estimate_y[x1 == ux1]) for ux1 in unique_x1
    ]
    stds_overlap_estimate = [np.std(overlap_estimate_y[x1 == ux1]) for ux1 in unique_x1]

    means_toffoli_ratio = [np.mean(toffoli_ratio_y[x1 == ux1]) for ux1 in unique_x1]
    stds_toffoli_ratio = [np.std(toffoli_ratio_y[x1 == ux1]) for ux1 in unique_x1]

    color = line_colors[i % len(line_colors)]

    # Overlap plot
    ax1.errorbar(
        unique_x1,
        means_actual_overlap,
        yerr=stds_actual_overlap,
        label=f"d = {fixed_d}",
        color=color,
    )
    ax1.errorbar(
        unique_x1,
        means_overlap_estimate,
        yerr=stds_overlap_estimate,
        fmt="o",
        color=color,
        alpha=0.3,
    )

    # Toffoli ratio (if shown)
    if plot_mode in ["dual", "all"]:
        ax3.errorbar(
            unique_x1,
            means_toffoli_ratio,
            yerr=stds_toffoli_ratio,
            label=f"d = {fixed_d}",
            color=color,
        )

# === Style subplot 1 ===
ax1.set_xlabel("Min Overlap allowed")
ax1.set_ylabel("Overlap estimate")
ax1.set_facecolor("#F9F9FB")
ax1.legend()
ax1.grid()

# === Style subplot 3 (if shown) ===
if plot_mode in ["dual", "all"]:
    ax3.set_xlabel("Min Overlap allowed")
    ax3.set_ylabel("Toffoli Ratio")
    ax3.set_facecolor("#F9F9FB")
    ax3.legend()
    ax3.grid()


# === Second set of plots (only for "all") ===
if plot_mode == "all":
    min_overlap_values = np.linspace(1, n_points, 4, dtype=int) / n_points

    for j, fixed_min_overlap in enumerate(min_overlap_values):
        mask_x1_fixed = np.isclose(min_overlap, fixed_min_overlap)
        x2 = d[mask_x1_fixed]

        actual_overlap_y = actual_overlap[mask_x1_fixed]
        overlap_estimate_y = overlap_estimate[mask_x1_fixed]
        toffoli_ratio_y = toffoli_ratio[mask_x1_fixed]

        unique_x2 = np.unique(x2)

        color = line_colors[j % len(line_colors)]
        ax2.errorbar(
            unique_x2,
            [np.mean(actual_overlap_y[x2 == ux1]) for ux1 in unique_x2],
            yerr=[np.std(actual_overlap_y[x2 == ux1]) for ux1 in unique_x2],
            label=f"TD = {fixed_min_overlap}",
            color=color,
        )
        ax2.errorbar(
            unique_x2,
            [np.mean(overlap_estimate_y[x2 == ux1]) for ux1 in unique_x2],
            yerr=[np.std(overlap_estimate_y[x2 == ux1]) for ux1 in unique_x2],
            fmt="o",
            color=color,
            alpha=0.3,
        )
        ax4.errorbar(
            unique_x2,
            [np.mean(toffoli_ratio_y[x2 == ux1]) for ux1 in unique_x2],
            yerr=[np.std(toffoli_ratio_y[x2 == ux1]) for ux1 in unique_x2],
            label=f"TD = {fixed_min_overlap}",
            color=color,
        )

    for ax in [ax2, ax4]:
        ax.set_facecolor("#F9F9FB")
        ax.legend()
        ax.grid()

    ax2.set_xlabel("d")
    ax2.set_ylabel("Overlap estimate")
    ax4.set_xlabel("d")
    ax4.set_ylabel("Toffoli Ratio")


# === Final layout ===
plt.tight_layout()
filename_map = {
    "single": f"ordering_n_{n_qubit}_vector_single.pdf",
    "dual": f"ordering_n_{n_qubit}_vector_dual.pdf",
    "all": f"ordering_n_{n_qubit}_vector_all.pdf",
}
plt.savefig(plot_folder / filename_map[plot_mode])
plt.show()
