"""
Approximate simulation.
Generates two plots:
  1. CNOTs Approx / CNOTs Uniform  vs  min overlap allowed
  2. CNOTs Approx / CNOTs eps=0    vs  min overlap allowed
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
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M = 20
repeat = 20
n_points = 5
vec_type = "real"
n_qubit = 15
d_values = [10, 50, 100]
min_overlap_values = np.linspace(0.8, 1, num=n_points)

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 8
FILEPATH = data_folder / f"ratios_n_{n_qubit}.npy"


def compute_values(min_overlap: float, n_qubits: int, sparsity: int):
    """Return npy line: min_overlap  d  num_gates_approx  num_gates_uniform  num_gates_eps_zero"""
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    angles_phases = build_dictionary(psi)
    angles_phases_copy = copy.deepcopy(angles_phases)

    # eps=0 optimisation
    angle_phases_zero = optimize_full_dict(angles_phases_copy)
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
        min_overlap, d, num_gates_approx, num_gates_uniform, num_gates_eps_zero,
    ) = np.loadtxt(FILEPATH, unpack=True)

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
    plot()