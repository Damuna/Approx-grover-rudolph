"""
Approximate simulation.
Generates two plots:
  1. CNOTs Approx / CNOTs eps=0 vs min overlap allowed (with actual overlap mapped to opacity)
  2. Actual vs Estimated Overlap comparison
Both feature curves for different M values, and constant sparsity (d) and n_qubits.
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
    GR_circuit,
)

print = functools.partial(print, flush=True)
matplotlib.rcParams.update({"font.size": 15})
line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#39737C", "#7DC36D"]

# ── Parameters ──
M_values = [5, 10, 20, 40]  # Different M values for the curves
d = 10                      # Constant sparsity
n_qubit = 4                 # Constant number of qubits
repeat = 20
n_points = 5
vec_type = "real"
min_overlap_values = np.linspace(0.8, 1, num=n_points)

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 8
# Changed filename to ensure we collect the new overlap columns
FILEPATH = data_folder / f"ratios_and_overlaps_M_n_{n_qubit}_d_{d}.npy"


def compute_values(min_overlap: float, n_qubits: int, sparsity: int, M: int):
    """
    Return npy line: 
    min_overlap \t M \t num_gates_approx \t num_gates_eps_zero \t overlap_estimate \t actual_overlap
    """
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    angles_phases = build_dictionary(psi)
    angles_phases_copy = copy.deepcopy(angles_phases)

    # eps=0 optimisation
    angle_phases_zero = optimize_full_dict(angles_phases_copy)
    num_gates_eps_zero = hybrid_CNOT_count(angle_phases_zero)

    # approximate ordering
    overlap_estimate, _, _ = ordering_geometric_series(angles_phases, min_overlap, M)
    num_gates_approx = hybrid_CNOT_count(angles_phases)

    # Actual overlap via GR_circuit
    psi_approx = GR_circuit(angles_phases)
    psi_dense = psi.toarray().flatten()[: len(psi_approx)]
    actual_overlap = abs(np.real(psi_dense.conj() @ psi_approx))

    return (
        f"{min_overlap}\t{M}\t{num_gates_approx}\t{num_gates_eps_zero}\t"
        f"{overlap_estimate}\t{actual_overlap}\n"
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
    expected = repeat * len(min_overlap_values) * len(M_values)
    if not _file_needs_update(FILEPATH, expected):
        print(f"Data file already complete: {FILEPATH}")
        return

    print("──────── Collecting: Ratios and Overlaps (Varying M) ────────")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for min_ov in min_overlap_values:
                for M in M_values:
                    r = pool.apply_async(compute_values, (min_ov, n_qubit, d, M))
                    results.append(r)
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


def plot_ratio():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap, M_arr, num_gates_approx, num_gates_eps_zero, _, actual_overlap,
    ) = _load_data()

    ratio_eps_zero = num_gates_approx / num_gates_eps_zero

    plt.figure(figsize=(10, 7))
    for i, fixed_M in enumerate(M_values):
        mask = np.isclose(M_arr, fixed_M)
        x = min_overlap[mask]
        y = ratio_eps_zero[mask]
        ao = actual_overlap[mask]
        
        unique_x = np.unique(x)
        means = [np.mean(y[x == ux]) for ux in unique_x]
        stds = [np.std(y[x == ux]) for ux in unique_x]
        means_ao = [np.mean(ao[x == ux]) for ux in unique_x]
        
        color = line_colors[i % len(line_colors)]
        
        # Plot the main line for the legend
        plt.plot(unique_x, means, color=color, label=f"M = {fixed_M}")
        
        # Plot each point/errorbar with its specific opacity mapped to the actual overlap
        for ux, my, sy, mao in zip(unique_x, means, stds, means_ao):
            # Clip actual overlap between 0 and 1 for the alpha parameter
            alpha_val = max(0.0, min(1.0, mao))
            plt.errorbar(
                ux, my, yerr=sy, color=color, 
                alpha=alpha_val, fmt="o", capsize=5
            )
                     
    plt.xlabel("Min overlap allowed")
    plt.ylabel("CNOTs Approx / CNOTs ε=0")
    plt.title(f"Approximation Ratio (Point Opacity = Actual Overlap)\nd={d}, n={n_qubit}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    filename = f"ratio_eps_zero_M_n_{n_qubit}_opacity.pdf"
    plt.savefig(plot_folder / filename)
    plt.show()
    plt.close()
    print(f"Plot saved: {plot_folder / filename}")


def plot_overlap_comparison():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    min_overlap, M_arr, _, _, overlap_estimate, actual_overlap = _load_data()

    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    for i, fixed_M in enumerate(M_values):
        mask = np.isclose(M_arr, fixed_M)
        x = min_overlap[mask]
        ao = actual_overlap[mask]
        oe = overlap_estimate[mask]

        unique_x = np.unique(x)
        means_ao = [np.mean(ao[x == ux]) for ux in unique_x]
        stds_ao = [np.std(ao[x == ux]) for ux in unique_x]
        means_oe = [np.mean(oe[x == ux]) for ux in unique_x]
        stds_oe = [np.std(oe[x == ux]) for ux in unique_x]

        color = line_colors[i % len(line_colors)]
        
        # Actual Overlap (Solid Line with Errorbars)
        ax.errorbar(unique_x, means_ao, yerr=stds_ao,
                    label=f"M = {fixed_M} (actual)", color=color, fmt="-o")
        
        # Overlap Estimate (Faded Scatter / Errorbars)
        ax.errorbar(unique_x, means_oe, yerr=stds_oe,
                    fmt="s", color=color, alpha=0.3,
                    label=f"M = {fixed_M} (estimate)")

    ax.set_xlabel("Min Overlap allowed")
    ax.set_ylabel("Overlap")
    ax.set_title(f"Actual vs Estimated Overlap (d={d}, n={n_qubit})")
    ax.set_facecolor("#F9F9FB")
    
    # Legend adjusted to not overlap the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid()
    plt.tight_layout()
    
    filename = f"overlap_comparison_M_n_{n_qubit}.pdf"
    plt.savefig(plot_folder / filename)
    plt.show()
    plt.close()
    print(f"Plot saved: {plot_folder / filename}")


if __name__ == "__main__":
    collect()
    plot_ratio()
    plot_overlap_comparison()