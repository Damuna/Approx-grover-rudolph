"""
Approximate simulation with explicit overlap via GR_circuit.
One plot: overlap estimate (dots) and actual overlap (lines) vs min overlap allowed,
with curves for different d values.
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
n_qubit = 10
d_values = [12]
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
    Return npy line: min_overlap  d  overlap_estimate  actual_overlap

    - overlap_estimate: bound returned by ordering_geometric_series
    - actual_overlap:   |<psi | psi_approx>| computed explicitly via GR_circuit
    """
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    # original dictionary
    angles_phases = build_dictionary(psi)
    angles_phases_copy = copy.deepcopy(angles_phases)

    # eps=0 optimised dictionary (for the ratio denominator)
    angle_phases_zero = optimize_full_dict(angles_phases_copy)

    # approximate ordering
    overlap_estimate = ordering_geometric_series(angles_phases, min_overlap, M)

    # reconstruct approximate state vector
    psi_approx = GR_circuit(angles_phases)

    # actual overlap (trim psi to match psi_approx length if needed)
    psi_dense = psi.toarray().flatten()[: len(psi_approx)]
    actual_overlap = abs(np.real(psi_dense.conj() @ psi_approx))

    return (
        f"{min_overlap}\t{sparsity}\t{overlap_estimate}\t"
        f"{actual_overlap}\n"
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


def plot():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap, d, overlap_estimate, actual_overlap,
    ) = np.loadtxt(FILEPATH, unpack=True)

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
    print(f"Plot saved: {plot_folder / f'ordering_n_{n_qubit}_vector_single.pdf'}")


if __name__ == "__main__":
    collect()
    plot()