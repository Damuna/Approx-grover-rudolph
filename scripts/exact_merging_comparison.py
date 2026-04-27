"""
Exact-merge comparison.
Generates one plot showing how much the exact support-aware mergings reduce
Grover-Rudolph CNOT counts.

Curves:
  1. single rotations after exact merging / single rotations before merging
  2. hybrid after exact merging / hybrid before merging
  3. hybrid after exact merging / UCR

The first two curves isolate the effect of exact merging.
The third curve shows the final hybrid exact algorithm relative to UCR.
"""

from pathlib import Path
import copy
import functools
import multiprocessing as mp
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
repeat = 20
vec_type = "real"
n_qubit = 20
n_points = 15
D_values = np.logspace(-4, -2, num=n_points)
d_values = [max(1, int(D * 2**n_qubit)) for D in D_values]

FORCE_RECOLLECT = False

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 128
FILEPATH = data_folder / f"exact_merging_comparison_n_{n_qubit}.npy"

line_colors = ["#2D2F92", "#DC3977", "#39737C"]


def compute_values(n_qubits: int, sparsity: int):
    """
    Return one data line with CNOT counts before and after exact merging.

    Columns:
        d
        num_uniform
        num_single_before
        num_single_after
        num_hybrid_before
        num_hybrid_after
    """
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    # Raw Grover-Rudolph dictionary, before exact merging.
    baseline_angles = build_dictionary(psi, n_qubits)

    # Exact support-aware merging.
    exact_angles = optimize_full_dict_support_aware_exact(
        copy.deepcopy(baseline_angles)
    )

    num_uniform = (2**n_qubits) - 1

    num_single_before = single_rotation_count(baseline_angles)
    num_single_after = single_rotation_count(exact_angles)

    num_hybrid_before = hybrid_CNOT_count(baseline_angles)
    num_hybrid_after = hybrid_CNOT_count(exact_angles)

    return (
        f"{sparsity}\t{num_uniform}\t"
        f"{num_single_before}\t{num_single_after}\t"
        f"{num_hybrid_before}\t{num_hybrid_after}\n"
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
    expected = repeat * len(d_values)

    if not _file_needs_update(FILEPATH, expected, force=FORCE_RECOLLECT):
        print(f"Data file already complete: {FILEPATH}")
        return

    print(f"──────── Collecting: Exact merging comparison, n = {n_qubit} ────────")

    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for _ in range(repeat):
            for d in d_values:
                results.append(pool.apply_async(compute_values, (n_qubit, d)))

        pool.close()
        _output_progress(results, expected)
        pool.join()

    print("Writing results …")

    with open(FILEPATH, "w") as f:
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


def _mean_std_by_d(d, values):
    d_unique = np.unique(d)
    means = np.array([np.mean(values[np.isclose(d, di)]) for di in d_unique])
    stds = np.array([np.std(values[np.isclose(d, di)]) for di in d_unique])
    return d_unique, means, stds


def plot():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        d,
        num_uniform,
        num_single_before,
        num_single_after,
        num_hybrid_before,
        num_hybrid_after,
    ) = np.loadtxt(FILEPATH, unpack=True)

    D = d / 2**n_qubit

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_single_after_before = np.divide(
            num_single_after,
            num_single_before,
            out=np.full_like(num_single_after, np.nan, dtype=float),
            where=num_single_before > 0,
        )

    _set_plot_style()
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    D_unique, means, stds = _mean_std_by_d(D, ratio_single_after_before)

    ax.errorbar(
        D_unique,
        means,
        yerr=stds,
        fmt="o--",
        color=line_colors[0],
        label=r"single after / single before",
    )

    #ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$D=d/2^n$")
    ax.set_ylabel(
        r"$N_{\mathrm{CNOT},\mathrm{w/\,\,merge}}/"
        r"N_{\mathrm{CNOT},\mathrm{w/o\,\,merge}}$"
    )
    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_facecolor("#F9F9FB")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = plot_folder / f"single_exact_merging_comparison_n_{n_qubit}.pdf"
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Plot saved: {outpath}")


if __name__ == "__main__":
    collect()
    plot()
