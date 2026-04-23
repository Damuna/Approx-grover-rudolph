"""
Approximate simulation.
Generates three plots:
  1. CNOTs Exact / CNOTs Uniform vs min overlap allowed
  2. CNOTs Approx / CNOTs Exact  vs min overlap allowed
     Both with curves for different d values
  3. CNOTs Approx / CNOTs Exact vs min overlap allowed
     for fixed sparsity D = 5e-5 and different values of M
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
    optimize_full_dict_support_aware_exact,
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M = 20
M_values = [10, 50, 100]  # values used for the new fixed-D plot
repeat = 10
n_points = 5
vec_type = "real"
n_qubit = 15

D_values = [1e-5, 5e-5, 1e-4, 5e-4]
d_values = [max(1, int(D * 2**n_qubit)) for D in D_values]
min_overlap_values = np.linspace(0.75, 1, num=n_points)

# Fixed sparsity for the new M-sweep plot
D_fixed = 5e-4
d_fixed = max(1, int(D_fixed * 2**n_qubit))

# Forse recollection
FORCE_RECOLLECT = True

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 8
FILEPATH = data_folder / f"ratios_n_{n_qubit}.npy"
FILEPATH_M_SWEEP = data_folder / f"ratios_fixed_D_{D_fixed:.0e}_vs_M_n_{n_qubit}.npy"


def compute_values(min_overlap: float, n_qubits: int, sparsity: int):
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    # Standard GR
    baseline_angles = build_dictionary(psi, n_qubits)

    # Exact merging
    exact_angles = optimize_full_dict_support_aware_exact(
        copy.deepcopy(baseline_angles)
    )
    num_gates_exact = hybrid_CNOT_count(exact_angles)

    # Approx mergings with the default/global M
    approx_angles = copy.deepcopy(exact_angles)
    ordering_geometric_series(
        approx_angles,
        min_overlap,
        M,
        baseline_gate_operations=baseline_angles,
        use_rigorous_bound=False,
        regional_merges=False,
    )
    num_gates_approx = hybrid_CNOT_count(approx_angles)

    num_gates_uniform = (2**n_qubits) - 1

    return (
        f"{min_overlap}\t{sparsity}\t{num_gates_approx}\t"
        f"{num_gates_uniform}\t{num_gates_exact}\n"
    )


def compute_values_M_sweep(
    min_overlap: float,
    n_qubits: int,
    sparsity: int,
    M_value: int,
):
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    # Standard GR
    baseline_angles = build_dictionary(psi, n_qubits)

    # Exact merging
    exact_angles = optimize_full_dict_support_aware_exact(
        copy.deepcopy(baseline_angles)
    )
    num_gates_exact = hybrid_CNOT_count(exact_angles)

    # Approx mergings with variable M
    approx_angles = copy.deepcopy(exact_angles)
    ordering_geometric_series(
        approx_angles,
        min_overlap,
        M_value,
        baseline_gate_operations=baseline_angles,
        use_rigorous_bound=False,
        regional_merges=False,
    )
    num_gates_approx = hybrid_CNOT_count(approx_angles)

    return (
        f"{min_overlap}\t{sparsity}\t{M_value}\t{num_gates_approx}\t{num_gates_exact}\n"
    )


def _output_progress(results, total):
    n_completed = 0
    n_percent_old = 0
    while n_completed < total:
        n_completed = sum(1 for r in results if r.ready())
        n_percent = int(n_completed / total * 100)
        if n_percent > n_percent_old:
            print(f"{n_percent}% finished")
        n_percent_old = n_percent
        time.sleep(1)


def _file_needs_update(filepath, expected_lines, force=False):
    if force:
        return True
    if not filepath.exists():
        return True
    with open(filepath, "r") as f:
        existing = sum(1 for line in f if line.strip())
    return existing != expected_lines


def collect():
    expected = repeat * len(min_overlap_values) * len(d_values)
    if not _file_needs_update(FILEPATH, expected, force=FORCE_RECOLLECT):
        print(f"Data file already complete: {FILEPATH}")
        return

    print("──────── Collecting: Ratios ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for min_ov in min_overlap_values:
                for d in d_values:
                    r = pool.apply_async(compute_values, (min_ov, n_qubit, d))
                    results.append(r)
        pool.close()
        _output_progress(results, expected)
        pool.join()

    print("Writing results …")
    with open(FILEPATH, "w") as f:
        for r in results:
            f.write(r.get())
    print()


def collect_M_sweep():
    expected = repeat * len(min_overlap_values) * len(M_values)
    if not _file_needs_update(FILEPATH_M_SWEEP, expected, force=FORCE_RECOLLECT):
        print(f"Data file already complete: {FILEPATH_M_SWEEP}")
        return

    print(f"──────── Collecting: Fixed D = {D_fixed:.0e}, varying M ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for min_ov in min_overlap_values:
                for M_value in M_values:
                    r = pool.apply_async(
                        compute_values_M_sweep,
                        (min_ov, n_qubit, d_fixed, M_value),
                    )
                    results.append(r)
        pool.close()
        _output_progress(results, expected)
        pool.join()

    print("Writing results …")
    with open(FILEPATH_M_SWEEP, "w") as f:
        for r in results:
            f.write(r.get())
    print()


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
    plt.figure(figsize=(7, 6))


def _finalize_plot(filename, legend_ncol=2):
    plt.xlabel("minimum allowed overlap ")
    plt.yscale("log")
    plt.grid(True)
    ax = plt.gca()
    ax.set_facecolor("#F9F9FB")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=legend_ncol,
        fancybox=True,
        shadow=True,
    )
    plt.savefig(plot_folder / filename, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Plot saved: {plot_folder / filename}")


def _grouped_errorbar_by_d(x_values, d_data, y_values, ylabel, filename):
    _set_plot_style()

    for i, fixed_d_val in enumerate(d_values):
        mask = np.isclose(d_data, fixed_d_val)
        x = x_values[mask]
        y = y_values[mask]

        unique_x = np.unique(x)
        means = [np.mean(y[x == ux]) for ux in unique_x]
        stds = [np.std(y[x == ux]) for ux in unique_x]

        color = line_colors[i % len(line_colors)]
        D_formatted = "{:.0e}".format(D_values[i])

        plt.errorbar(
            unique_x,
            means,
            yerr=stds,
            label=f"D = {D_formatted}",
            color=color,
            fmt="--",
        )

    plt.ylabel(ylabel)
    _finalize_plot(filename, legend_ncol=2)


def _grouped_errorbar_by_M(x_values, M_data, y_values, ylabel, filename):
    _set_plot_style()

    for i, fixed_M in enumerate(M_values):
        mask = np.isclose(M_data, fixed_M)
        x = x_values[mask]
        y = y_values[mask]

        unique_x = np.unique(x)
        means = [np.mean(y[x == ux]) for ux in unique_x]
        stds = [np.std(y[x == ux]) for ux in unique_x]

        color = line_colors[i % len(line_colors)]

        plt.errorbar(
            unique_x,
            means,
            yerr=stds,
            label=rf"$M = {fixed_M}$",
            color=color,
            fmt="--",
        )

    plt.ylabel(ylabel)
    _finalize_plot(
        filename,
        legend_ncol=min(2, len(M_values)) if len(M_values) > 1 else 1,
    )


def plot():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        num_gates_approx,
        num_gates_uniform,
        num_gates_exact,
    ) = np.loadtxt(FILEPATH, unpack=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_exact_uniform = np.divide(
            num_gates_exact,
            num_gates_uniform,
            out=np.full_like(num_gates_exact, np.nan, dtype=float),
            where=num_gates_uniform > 0,
        )
        ratio_approx_exact = np.divide(
            num_gates_approx,
            num_gates_exact,
            out=np.full_like(num_gates_approx, np.nan, dtype=float),
            where=num_gates_exact > 0,
        )

    _grouped_errorbar_by_d(
        min_overlap,
        d,
        ratio_exact_uniform,
        r"$N_{\mathrm{CNOT,exact}}$ / $N_{\mathrm{CNOT,uniform}}$",
        f"ratio_uniform_exact_n_{n_qubit}.pdf",
    )

    _grouped_errorbar_by_d(
        min_overlap,
        d,
        ratio_approx_exact,
        r"$N_{\mathrm{CNOT,approx}}$ / $N_{\mathrm{CNOT,exact}}$",
        f"ratio_exact_approx_n_{n_qubit}.pdf",
    )

    if not FILEPATH_M_SWEEP.exists():
        print(f"No data file found for fixed-D M-sweep: {FILEPATH_M_SWEEP}")
        return

    (
        min_overlap_M,
        d_M,
        M_data,
        num_gates_approx_M,
        num_gates_exact_M,
    ) = np.loadtxt(FILEPATH_M_SWEEP, unpack=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_approx_exact_M = np.divide(
            num_gates_approx_M,
            num_gates_exact_M,
            out=np.full_like(num_gates_approx_M, np.nan, dtype=float),
            where=num_gates_exact_M > 0,
        )

    _grouped_errorbar_by_M(
        min_overlap_M,
        M_data,
        ratio_approx_exact_M,
        r"$N_{\mathrm{CNOT,approx}}$ / $N_{\mathrm{CNOT,exact}}$",
        f"ratio_exact_approx_fixed_D_{D_fixed:.0e}_vs_M_n_{n_qubit}.pdf",
    )


if __name__ == "__main__":
    # collect()
    collect_M_sweep()
    plot()
