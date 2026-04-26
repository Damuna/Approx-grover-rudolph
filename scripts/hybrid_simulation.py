"""
No approximate simulation.
Generates one plot: Uniform vs Single Rotations w/ mergings (CNOT count) as a function of sparsity d.
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
    optimize_full_dict_support_aware_exact,
    hybrid_CNOT_count,
    single_rotation_count,
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})

# ── Parameters ──
M = 20
repeat = 20
vec_type = "real"
n_qubit = 20
n_points = 15
d_values = np.logspace(1, 5, num=n_points).astype(int).tolist()

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 8
FILEPATH = data_folder / f"cnots_vs_d_n_{n_qubit}.npy"


def compute_values(n_qubits: int, sparsity: int):
    """Return npy line: d  num_gates_uniform  num_gates_eps_zero  num_gates_single_rot"""
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)
    angles_phases = build_dictionary(psi, n_qubits)
    angle_phases_zero = optimize_full_dict_support_aware_exact(copy.deepcopy(angles_phases))

    num_gates_uniform = (2**n_qubits) - 1
    num_gates_eps_zero = hybrid_CNOT_count(angle_phases_zero)
    num_gates_single_rot = single_rotation_count(angle_phases_zero)

    return f"{sparsity}\t{num_gates_uniform}\t{num_gates_eps_zero}\t{num_gates_single_rot}\n"


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
    expected = repeat * len(d_values)
    if not _file_needs_update(FILEPATH, expected):
        print(f"Data file already complete: {FILEPATH}")
        return

    print("──────── Collecting: CNOTs vs d for {n_qubit} qubits ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for d in d_values:
                r = pool.apply_async(compute_values, (n_qubit, d))
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

    d, num_uniform, num_eps_zero, num_single_rot = np.loadtxt(FILEPATH, unpack=True)

    d_unique = np.unique(d)
    means_uniform = [np.mean(num_uniform[np.isclose(d, di)]) for di in d_unique]
    stds_uniform = [np.std(num_uniform[np.isclose(d, di)]) for di in d_unique]
    means_single = [np.mean(num_single_rot[np.isclose(d, di)]) for di in d_unique]
    stds_single = [np.std(num_single_rot[np.isclose(d, di)]) for di in d_unique]
    means_hybrid = [np.mean(num_eps_zero[np.isclose(d, di)]) for di in d_unique]
    stds_hybrid = [np.std(num_eps_zero[np.isclose(d, di)]) for di in d_unique]


    d_unique /= 2 ** n_qubit
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
    plt.errorbar(d_unique, means_uniform, yerr=stds_uniform, fmt="o--",
                 label="UCR", color="#2D2F92")
    plt.errorbar(d_unique, means_single, yerr=stds_single, fmt="s--",
             label="single", color="#DC3977")
    plt.errorbar(d_unique, means_hybrid, yerr=stds_hybrid, fmt="^--", label="hybrid", color="green")
    plt.xlabel("D")
    plt.ylabel("\\# CNOT gates")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.set_facecolor("#F9F9FB")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
    plt.savefig(plot_folder / f"num_gates_vs_d_n_{n_qubit}.pdf", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Plot saved: {plot_folder / f'num_gates_vs_d_n_{n_qubit}.pdf'}")


if __name__ == "__main__":
    collect()
    plot()