"""
Approximate simulation.
Generates two plots:
  1. CNOTs Exact / CNOTs Uniform vs min overlap allowed
  2. CNOTs Approx / CNOTs Exact  vs min overlap allowed
Both with curves for different d values.
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
    hybrid_CNOT_count,
    optimize_full_dict_support_aware_exact,
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M = 20
repeat = 10
n_points = 5
vec_type = "real"
n_qubit = 20
D_values = [1e-5, 5e-5, 1e-4, 5e-4]
d_values = [max(1, int(D * 2**n_qubit)) for D in D_values]
min_overlap_values = np.linspace(0.75, 1, num=n_points)

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 8
FILEPATH = data_folder / f"ratios_n_{n_qubit}.npy"


def compute_values(min_overlap: float, n_qubits: int, sparsity: int):
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

    ratio_exact_uniform = num_gates_exact / num_gates_uniform
    ratio_approx_exact = num_gates_approx / num_gates_exact

    def _grouped_errorbar(y_values, ylabel, filename):
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

        for i, fixed_d in enumerate(d_values):
            mask = np.isclose(d, fixed_d)
            x = min_overlap[mask]
            y = y_values[mask]
            unique_x = np.unique(x)
            means = [np.mean(y[x == ux]) for ux in unique_x]
            stds = [np.std(y[x == ux]) for ux in unique_x]
            
            color = line_colors[i % len(line_colors)]
            D_formatted = '{:.0e}'.format(D_values[i])
            
            plt.errorbar(
                unique_x, means, yerr=stds,
                label=f"D = {D_formatted}", color=color, fmt="--"
            )

        plt.xlabel("minimum allowed overlap ")
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(True)
        ax = plt.gca()
        ax.set_facecolor("#F9F9FB")
        
        # Used ncol=2 because there are 4 D_values, making a neat 2x2 grid.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True, shadow=True)
        
        plt.savefig(plot_folder / filename, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Plot saved: {plot_folder / filename}")

    _grouped_errorbar(
        ratio_exact_uniform,
        '$N_{\\mathrm{CNOT,exact}}$ / $N_{\\mathrm{CNOT,uniform}}$',
        f"ratio_uniform_exact_n_{n_qubit}.pdf",
    )
    _grouped_errorbar(
        ratio_approx_exact,
        '$N_{\\mathrm{CNOT,approx}}$ / $N_{\\mathrm{CNOT,exact}}$',
        f"ratio_exact_approx_n_{n_qubit}.pdf",
    )


if __name__ == "__main__":
    collect()
    plot()