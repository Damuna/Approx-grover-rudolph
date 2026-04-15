"""
Approximate simulation with explicit overlap via GR_circuit.
Includes overlap plots and gate count ratio plots.
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
    ordering_geometric_series,
    hybrid_CNOT_count,
    GR_circuit_sparse,
    optimize_full_dict_support_aware_exact,
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M = 20
repeat = 20
n_points = 10
vec_type = "real"
n_qubit = 10
D_values = [1e-4, 5e-4, 1e-3, 5e-3]
d_values = [int(D * 2**n_qubit) for D in D_values]
min_overlap_values = np.linspace(0.75, 1, num=n_points)

# Test Parameters
repeat = 10
n_qubit = 15
D_values = [1e-4, 5e-4, 1e-3, 5e-3]
d_values = [int(D * 2**n_qubit) for D in D_values]
force = False

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 2
FILEPATH = data_folder / f"ordering_alg_n_{n_qubit}_vector.npy"


def compute_values(min_overlap: float, n_qubits: int, sparsity: int):
    # Generate random vector
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    # Standard GR
    baseline_angles = build_dictionary(psi, n_qubit)

    # Exact merging
    exact_angles = optimize_full_dict_support_aware_exact(
        copy.deepcopy(baseline_angles)
    )
    num_gates_exact = hybrid_CNOT_count(exact_angles)

    # Approx mergings
    approx_angles = copy.deepcopy(exact_angles)
    overlap_estimate = ordering_geometric_series(
        approx_angles,
        min_overlap,
        M, baseline_gate_operations=baseline_angles,
    )
    num_gates_approx = hybrid_CNOT_count(approx_angles)

    # Compute overlap
    psi_approx = GR_circuit_sparse(approx_angles, return_sparse=True)
    psi_ref = psi[:, : psi_approx.shape[1]]
    actual_overlap = abs(psi_ref.conj().multiply(psi_approx).sum())

    num_gates_uniform = (2**n_qubits) - 1

    return (
        f"{min_overlap}\t{sparsity}\t{overlap_estimate}\t"
        f"{actual_overlap}\t{num_gates_approx}\t"
        f"{num_gates_uniform}\t{num_gates_exact}\n"
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


def collect(force_recollect=False):
    expected = repeat * len(min_overlap_values) * len(d_values)

    if not force_recollect and not _file_needs_update(FILEPATH, expected):
        print(f"Data file already complete: {FILEPATH}")
        print("Use force_recollect=True to override and recollect data.")
        return

    if force_recollect:
        print(f"Force recollecting data (overwriting {FILEPATH})")

    print("──────── Collecting: Overlap (vector) ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for min_ov in min_overlap_values:
                for d in d_values:
                    results.append(
                        pool.apply_async(compute_values, (min_ov, n_qubit, d))
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
    data = np.loadtxt(FILEPATH, unpack=True)

    if data.shape[0] != 7:
        raise ValueError(
            f"Expected 7 columns in {FILEPATH}, found {data.shape[0]}"
        )

    return data


def _set_plot_style():
    SMALL_SIZE = 19
    MEDIUM_SIZE = 21
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "font.size": SMALL_SIZE,
        "axes.titlesize": SMALL_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
    }
    plt.rcParams.update(params)


def plot():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        overlap_estimate,
        actual_overlap,
        num_gates_approx,
        num_gates_uniform,
        num_gates_exact,
    ) = _load_data()

    _set_plot_style()

    # --------------------------------------------------
    # Figure 1: actual overlap and estimated overlap
    # --------------------------------------------------
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
        D_formatted = "{:.0e}".format(D_values[i])

        ax.errorbar(
            unique_x,
            means_ao,
            yerr=stds_ao,
            label=f"D = {D_formatted}",
            color=color,
            fmt="--",
        )
        ax.errorbar(
            unique_x,
            means_oe,
            yerr=stds_oe,
            fmt="o",
            color=color,
            alpha=0.5,
        )

    ax.set_xlabel("minimum allowed overlap")
    ax.set_ylabel("overlap")
    ax.set_facecolor("#F9F9FB")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    ax.grid()
    plt.savefig(
        plot_folder / f"ordering_n_{n_qubit}_vector_single.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # --------------------------------------------------
    # Figure 2: ratio exact / uniform
    # --------------------------------------------------
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    ratio_uniform_exact = num_gates_exact / num_gates_uniform

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        y = ratio_uniform_exact[mask]

        unique_x = np.unique(x)
        means_y = np.array([np.mean(y[x == ux]) for ux in unique_x])
        stds_y = np.array([np.std(y[x == ux]) for ux in unique_x])

        color = line_colors[i % len(line_colors)]
        D_formatted = "{:.0e}".format(D_values[i])

        ax.errorbar(
            unique_x,
            means_y,
            yerr=stds_y,
            color=color,
            ls="-",
            marker="o",
            label=f"D = {D_formatted}",
        )

    ax.set_xlabel("minimum allowed overlap")
    ax.set_ylabel(r"$\mathrm{CNOT}_{\mathrm{exact}} / \mathrm{CNOT}_{\mathrm{uniform}}$")
    ax.set_facecolor("#F9F9FB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        plot_folder / f"ratio_uniform_exact_n_{n_qubit}.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # --------------------------------------------------
    # Figure 3: ratio approx / exact
    # --------------------------------------------------
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    ratio_exact_approx = num_gates_approx / num_gates_exact

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        y = ratio_exact_approx[mask]

        unique_x = np.unique(x)
        means_y = np.array([np.mean(y[x == ux]) for ux in unique_x])
        stds_y = np.array([np.std(y[x == ux]) for ux in unique_x])

        color = line_colors[i % len(line_colors)]
        D_formatted = "{:.0e}".format(D_values[i])

        ax.errorbar(
            unique_x,
            means_y,
            yerr=stds_y,
            color=color,
            ls="-",
            marker="s",
            label=f"D = {D_formatted}",
        )

    ax.set_xlabel("minimum allowed overlap")
    ax.set_ylabel(r"$\mathrm{CNOT}_{\mathrm{approx}} / \mathrm{CNOT}_{\mathrm{exact}}$")
    ax.set_facecolor("#F9F9FB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        plot_folder / f"ratio_exact_approx_n_{n_qubit}.pdf",
        dpi=600,
        bbox_inches="tight",
    )
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

    (
        min_ov,
        d_arr,
        overlap_estimate,
        actual_overlap,
        num_gates_approx,
        num_gates_uniform,
        num_gates_exact,
    ) = _load_data()

    diff = overlap_estimate - actual_overlap

    _set_plot_style()
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
            unique_x,
            means_diff,
            yerr=stds_diff,
            color=color,
            ls="-",
            marker="s",
            markersize=4,
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
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Overlap difference plot saved: {outpath}")


if __name__ == "__main__":
    collect(force_recollect=force)
    plot()
    plot_overlap_difference()