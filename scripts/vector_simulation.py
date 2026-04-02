"""
Approximate simulation with explicit overlap via GR_circuit.
"""

from pathlib import Path
import copy
import functools
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from approx_grover_rudolph import (
    generate_sparse_unit_vector,
    build_dictionary,
    optimize_full_dict,
    ordering_geometric_series,
    gate_count,
    GR_circuit,
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M = 10
repeat = 20
n_points = 10
vec_type = "real"
n_qubit = 12
d_values = [5, 10, 50, 100]
min_overlap_values = np.linspace(0.5, 1, num=n_points)

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 8
FILEPATH = data_folder / f"ordering_alg_n_{n_qubit}_vector.npy"


def compute_values(n_qubits: int, sparsity: int, min_overlap: float):
    """
    Return npy line:
        min_overlap  d  overlap_estimate  actual_overlap  total_merges  zero_merges
    """
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    angles_phases = build_dictionary(psi)
    angles_phases_copy = copy.deepcopy(angles_phases)

    angle_phases_zero = optimize_full_dict(angles_phases_copy)

    overlap_estimate, total_merges, zero_merges = ordering_geometric_series(
        angles_phases, min_overlap, M
    )

    psi_approx = GR_circuit(angles_phases)

    psi_dense = psi.toarray().flatten()[: len(psi_approx)]
    actual_overlap = abs(np.real(psi_dense.conj() @ psi_approx))

    return (
        f"{min_overlap}\t{sparsity}\t{overlap_estimate}\t"
        f"{actual_overlap}\t{total_merges}\t{zero_merges}\n"
    )


def _output_progress(results, total):
    n_completed, n_percent_old = 0, 0
    while n_completed < total:
        n_completed = sum(1 for r in results if r.ready())
        n_percent = int(n_completed / total * 100)
        if n_percent > n_percent_old:
            print(f"{n_percent}% finished")
        n_percent_old = n_percent
        time.sleep(1)


def _file_needs_update(filepath, expected_lines):
    if not filepath.exists():
        return True
    with open(filepath, "r") as f:
        existing = sum(1 for line in f if line.strip())
    return existing < expected_lines


def collect():
    expected = repeat * len(min_overlap_values) * len(d_values)
    if not _file_needs_update(FILEPATH, expected):
        print(f"Data file already complete: {FILEPATH}")
        return

    print("──────── Collecting: Overlap (vector) ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for min_ov in min_overlap_values:
                for d in d_values:
                    results.append(
                        pool.apply_async(compute_values, (n_qubit, d, min_ov))
                    )
        pool.close()
        _output_progress(results, expected)
        pool.join()

    print("Writing results …")
    with open(FILEPATH, "w") as f:
        for r in results:
            f.write(r.get())
    print()


def _load_data():
    return np.loadtxt(FILEPATH, unpack=True)


def plot():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    min_overlap, d, overlap_estimate, actual_overlap, _, _ = _load_data()

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        ao = actual_overlap[mask]
        oe = overlap_estimate[mask]

        unique_x = np.unique(x)
        means_ao = [np.mean(ao[x == ux]) for ux in unique_x]
        stds_ao = [np.std(ao[x == ux]) for ux in unique_x]
        means_oe = [np.mean(oe[x == ux]) for ux in unique_x]
        stds_oe = [np.std(oe[x == ux]) for ux in unique_x]

        color = line_colors[i % len(line_colors)]
        ax.errorbar(unique_x, means_ao, yerr=stds_ao,
                    label=f"d = {fixed_d} (actual)", color=color)
        ax.errorbar(unique_x, means_oe, yerr=stds_oe,
                    fmt="o", color=color, alpha=0.3,
                    label=f"d = {fixed_d} (estimate)")

    ax.set_xlabel("Min Overlap allowed")
    ax.set_ylabel("Overlap")
    ax.set_facecolor("#F9F9FB")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(plot_folder / f"ordering_n_{n_qubit}_vector_single.pdf")
    plt.show()
    plt.close()


def plot_overlap_difference():
    """
    Plot the difference (overlap_estimate - actual_overlap) vs min_overlap,
    one curve per d value, with error bars showing ±1 std over repeats.
    """
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    min_ov, d_arr, overlap_estimate, actual_overlap, _, _ = _load_data()

    diff = overlap_estimate - actual_overlap

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d_arr, fixed_d)
        x = min_ov[mask]
        df = diff[mask]

        unique_x = np.unique(x)
        means_diff = np.array([np.mean(df[x == ux]) for ux in unique_x])
        stds_diff = np.array([np.std(df[x == ux]) for ux in unique_x])

        color = line_colors[i % len(line_colors)]
        ax.errorbar(
            unique_x, means_diff, yerr=stds_diff,
            color=color, ls="-", marker="s", markersize=4,
            label=f"d = {fixed_d}",
        )

    ax.axhline(y=0, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("Min Overlap allowed")
    ax.set_ylabel("Overlap estimate $-$ Actual overlap")
    ax.set_facecolor("#F9F9FB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = plot_folder / f"overlap_difference_n_{n_qubit}.pdf"
    plt.savefig(outpath)
    plt.show()
    plt.close()
    print(f"Overlap difference plot saved: {outpath}")


def plot_cross_term():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    min_ov, d_arr, overlap_estimate, actual_overlap, _, _ = _load_data()

    sum_delta = 1.0 - overlap_estimate
    cross_term = overlap_estimate - actual_overlap
    ratio = cross_term / np.where(np.abs(sum_delta) > 1e-12, sum_delta, np.nan)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d_arr, fixed_d)
        if not np.any(mask):
            continue
        color = line_colors[i % len(line_colors)]
        ax.scatter(sum_delta[mask], ratio[mask],
                   color=color, alpha=0.4, s=20, label=f"$d = {fixed_d}$")

    ax.axhline(y=0.1, color="gray", ls="--", alpha=0.6, label="0.1 threshold")
    ax.axhline(y=-0.1, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel(r"$\sum_i \Delta_i$")
    ax.set_ylabel(r"$\mathrm{cross\;term}\;/\;\sum_i \Delta_i$")
    ax.set_facecolor("#F9F9FB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = plot_folder / f"cross_term_ratio_n_{n_qubit}.pdf"
    plt.savefig(outpath)
    plt.show()
    plt.close()
    print(f"Cross-term plot saved: {outpath}")


def plot_merge_counts():
    """
    Plot total number of mergings (solid) and mergings-with-zero (dashed)
    vs min_overlap, one curve per d value.
    """
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    min_ov, d_arr, _, _, total_merges, zero_merges = _load_data()

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d_arr, fixed_d)
        x = min_ov[mask]
        tm = total_merges[mask]
        zm = zero_merges[mask]

        unique_x = np.unique(x)
        means_tm = np.array([np.mean(tm[x == ux]) for ux in unique_x])
        stds_tm  = np.array([np.std(tm[x == ux])  for ux in unique_x])
        means_zm = np.array([np.mean(zm[x == ux]) for ux in unique_x])
        stds_zm  = np.array([np.std(zm[x == ux])  for ux in unique_x])

        color = line_colors[i % len(line_colors)]

        # total mergings – solid line
        ax.errorbar(
            unique_x, means_tm, yerr=stds_tm,
            color=color, ls="-", marker="s", markersize=4,
            label=f"d = {fixed_d} (total)",
        )
        # mergings with 0 – dashed line
        ax.errorbar(
            unique_x, means_zm, yerr=stds_zm,
            color=color, ls="--", marker="o", markersize=4, alpha=0.6,
            label=f"d = {fixed_d} (with 0)",
        )

    ax.set_xlabel("Min Overlap allowed")
    ax.set_ylabel("Number of mergings")
    ax.set_facecolor("#F9F9FB")
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = plot_folder / f"merge_counts_n_{n_qubit}.pdf"
    plt.savefig(outpath)
    plt.show()
    plt.close()
    print(f"Merge counts plot saved: {outpath}")


if __name__ == "__main__":
    collect()
    plot()
    plot_overlap_difference()
    plot_cross_term()
    plot_merge_counts()