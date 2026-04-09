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
    optimize_full_dict,
    ordering_geometric_series,
    gate_count,
    hybrid_CNOT_count,
    GR_circuit,
    optimize_full_dict_support_aware_exact
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M = 20
repeat = 20
n_points = 10
vec_type = "real"
n_qubit = 15
D_values = [ 1e-4, 5e-4, 1e-3, 5e-3 ] # 15 qubits
#D_values = [ 1e-5, 5e-4, 1e-4, 5e-4 ] # 20 qubits
d_values = [ int(D * 2**n_qubit) for D in D_values ]
min_overlap_values = np.linspace(0.75, 1, num=n_points)

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 1
FILEPATH = data_folder / f"ordering_alg_n_{n_qubit}_vector.npy"
RATIO_FILEPATH = data_folder / f"ratios_n_{n_qubit}.npy"


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


def compute_ratio_values(min_overlap: float, n_qubits: int, sparsity: int):
    """Return npy line: min_overlap  d  num_gates_approx  num_gates_uniform  num_gates_eps_zero"""
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    angles_phases = build_dictionary(psi)
    angles_phases_copy = copy.deepcopy(angles_phases)

    # eps=0 optimisation
    angle_phases_zero = optimize_full_dict_support_aware_exact(angles_phases_copy)
    num_gates_eps_zero = hybrid_CNOT_count(angle_phases_zero)
    num_gates_uniform = (2**n_qubits) - 1

    # approximate ordering
    ordering_geometric_series(angles_phases, min_overlap, M)
    num_gates_approx = hybrid_CNOT_count(angles_phases)

    return (
        f"{min_overlap}\t{sparsity}\t{num_gates_approx}\t"
        f"{num_gates_uniform}\t{num_gates_eps_zero}\n"
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
    else:
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


def collect_ratios():
    expected = repeat * len(min_overlap_values) * len(d_values)
    if not _file_needs_update(RATIO_FILEPATH, expected):
        print(f"Ratio data file already complete: {RATIO_FILEPATH}")
        return

    print("──────── Collecting: Ratios ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for min_ov in min_overlap_values:
                for d in d_values:
                    r = pool.apply_async(compute_ratio_values, (min_ov, n_qubit, d))
                    results.append(r)
        pool.close()
        _output_progress(results, expected)
        pool.join()

    print("Writing results …")
    with open(RATIO_FILEPATH, "w") as f:
        for r in results:
            f.write(r.get())
    print()


def _load_data():
    return np.loadtxt(FILEPATH, unpack=True)


def _load_ratio_data():
    return np.loadtxt(RATIO_FILEPATH, unpack=True)


def plot():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    min_overlap, d, overlap_estimate, actual_overlap, _, _ = _load_data()

    SMALL_SIZE = 19
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 23
    params = {
            "ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"],
            "font.size" : SMALL_SIZE,
            "axes.titlesize" : SMALL_SIZE,
            "axes.labelsize" : MEDIUM_SIZE,
            "xtick.labelsize" : SMALL_SIZE,
            "ytick.labelsize" : SMALL_SIZE,
            "legend.fontsize" : SMALL_SIZE,
            #"figure.titlesize" : BIGGER_SIZE,
            }
    plt.rcParams.update(params)
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
        D_formatted = '{:.0e}'.format(D_values[i])
        ax.errorbar(unique_x, means_ao, yerr=stds_ao,
                    label=f"D = {D_formatted}", color=color, fmt="--")
        ax.errorbar(unique_x, means_oe, yerr=stds_oe,
                    fmt="o", color=color, alpha=0.5)

    ax.set_xlabel("minimum allowed overlap")
    ax.set_ylabel("overlap")
    ax.set_facecolor("#F9F9FB")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, fancybox=True, shadow=True)
    ax.grid()
    plt.savefig(plot_folder / f"ordering_n_{n_qubit}_vector_single.pdf", dpi=600, bbox_inches='tight')
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


def plot_ratios():
    """Plot CNOT count ratios: approx/uniform and approx/eps=0"""
    if not RATIO_FILEPATH.exists():
        print(f"No ratio data file found: {RATIO_FILEPATH}")
        return

    (
        min_overlap, d, num_gates_approx, num_gates_uniform, num_gates_eps_zero,
    ) = _load_ratio_data()

    ratio_uniform = num_gates_approx / num_gates_uniform
    ratio_eps_zero = num_gates_approx / num_gates_eps_zero

    def _grouped_errorbar(y_values, ylabel, filename):
        plt.figure(figsize=(10, 7))
        for i, fixed_d in enumerate(d_values):
            mask = np.isclose(d, fixed_d)
            x = min_overlap[mask]
            y = y_values[mask]
            unique_x = np.unique(x)
            means = [np.mean(y[x == ux]) for ux in unique_x]
            stds = [np.std(y[x == ux]) for ux in unique_x]
            color = line_colors[i % len(line_colors)]
            plt.errorbar(unique_x, means, yerr=stds,
                         label=f"d = {fixed_d}", color=color)
        plt.xlabel("Min overlap allowed")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder / filename)
        plt.show()
        plt.close()
        print(f"Plot saved: {plot_folder / filename}")

    _grouped_errorbar(
        ratio_uniform,
        "CNOTs Approx / CNOTs Uniform",
        f"ratio_uniform_n_{n_qubit}.pdf",
    )
    _grouped_errorbar(
        ratio_eps_zero,
        "CNOTs Approx / CNOTs ε=0",
        f"ratio_eps_zero_n_{n_qubit}.pdf",
    )


if __name__ == "__main__":
    collect()
    collect_ratios()
    plot()
    plot_overlap_difference()
    plot_ratios()
