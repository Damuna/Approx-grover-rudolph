"""
Approximate simulation with explicit overlap via GR_circuit.
Includes three overlap measures:
  1. True overlap (computed explicitly, dense)
  2. Overlap estimate (cluster-based)
  3. Rigorous lower bound
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
n_qubit = 15

D_values = [1e-4, 5e-4, 1e-3]
d_values = [int(D * 2**n_qubit) for D in D_values]
min_overlap_values = np.linspace(0.75, 1, num=n_points)

# Overwrite cached data after logic changes.
force = False

# ── Folders ──
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)

N_PROCESSES = 1
FILEPATH = data_folder / f"overlap_comparison_n_{n_qubit}_vector.npy"


def _nonempty_layers_repr(dict_list):
    """
    Return only the non-empty layers in a readable format.
    """
    out = []
    for depth, layer in enumerate(dict_list):
        if layer:
            out.append(f"layer {depth}: {layer}")
    return "\n".join(out)


def compute_values(min_overlap: float, n_qubits: int, sparsity: int):
    psi = generate_sparse_unit_vector(n_qubits, sparsity, vector_type=vec_type)

    baseline_angles = build_dictionary(psi, n_qubit)

    exact_angles = optimize_full_dict_support_aware_exact(
        copy.deepcopy(baseline_angles)
    )
    num_gates_exact = hybrid_CNOT_count(exact_angles)

    approx_angles = copy.deepcopy(exact_angles)
    overlap_estimate, rigorous_bound = ordering_geometric_series(
        approx_angles,
        min_overlap,
        M,
        baseline_gate_operations=baseline_angles,
        use_rigorous_bound=True,
        regional_merges=False
    )

    num_gates_approx = hybrid_CNOT_count(approx_angles)

    # Sparse overlap
    psi_approx_sparse = GR_circuit_sparse(approx_angles, return_sparse=True)
    actual_overlap = abs(psi.conj().multiply(psi_approx_sparse).sum())

    if rigorous_bound is not None and rigorous_bound - actual_overlap > 1e-2:
        print("=" * 120)
        print(
            f"LOWER BOUND VIOLATION | min_overlap={min_overlap:.6f}, d={sparsity}"
        )
        print(f"LOWER BOUND: {rigorous_bound}")
        print(f"OVERLAP:     {actual_overlap}")
        print(f"ESTIMATE:    {overlap_estimate}")

        print("ORIGINAL STATE:", psi)
        print("RESULTING STATE:", psi_approx_sparse)

        print("\nORIGINAL DICT (non-empty layers)")
        print(_nonempty_layers_repr(baseline_angles))

        print("\nEXACT DICT (non-empty layers)")
        print(_nonempty_layers_repr(exact_angles))

        print("\nAPPROX DICT (non-empty layers)")
        print(_nonempty_layers_repr(approx_angles))
        print("=" * 120)

    num_gates_uniform = (2**n_qubits) - 1

    return (
        f"{min_overlap}\t{sparsity}\t{overlap_estimate}\t"
        f"{rigorous_bound}\t{actual_overlap}\t{num_gates_approx}\t"
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
        return

    if force_recollect:
        print(f"Force recollecting data (overwriting {FILEPATH})")

    print("──────── Collecting: Overlap Comparison ────────")
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

    if data.shape[0] != 8:
        raise ValueError(f"Expected 8 columns in {FILEPATH}, found {data.shape[0]}")

    return data


def _set_plot_style():
    SMALL_SIZE = 19
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 23
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
        #"figure.titlesize": BIGGER_SIZE,
    }
    plt.rcParams.update(params)


def plot_overlap_comparison():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        overlap_estimate,
        rigorous_bound,
        actual_overlap,
        num_gates_approx,
        num_gates_uniform,
        num_gates_exact,
    ) = _load_data()

    _set_plot_style()

    fig, ax = plt.subplots(len(D_values), 2, sharex=True, sharey='row')

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        ao = actual_overlap[mask]
        oe = overlap_estimate[mask]
        rb = rigorous_bound[mask]

        unique_x = np.unique(x)
        means_ao = [np.mean(ao[x == ux]) for ux in unique_x]
        means_oe = [np.mean(oe[x == ux]) for ux in unique_x]
        means_rb = [np.mean(rb[x == ux]) for ux in unique_x]

        # new plot
        diffs_overlap_vs_estimate_mean = [ np.mean(ao[x == ux] - oe[x == ux]) for ux in unique_x ]
        diffs_overlap_vs_bound_mean = [ np.mean(ao[x == ux] - rb[x == ux]) for ux in unique_x ]
        diffs_estimate_vs_bound_mean = [ np.mean(oe[x == ux] - rb[x == ux]) for ux in unique_x ]

        diffs_overlap_vs_estimate_min = [ np.min(ao[x == ux] - oe[x == ux]) for ux in unique_x ]
        diffs_overlap_vs_bound_min = [ np.min(ao[x == ux] - rb[x == ux]) for ux in unique_x ]
        diffs_estimate_vs_bound_min = [ np.min(oe[x == ux] - rb[x == ux]) for ux in unique_x ]

        diffs_overlap_vs_estimate_max = [ np.max(ao[x == ux] - oe[x == ux]) for ux in unique_x ]
        diffs_overlap_vs_bound_max = [ np.max(ao[x == ux] - rb[x == ux]) for ux in unique_x ]
        diffs_estimate_vs_bound_max = [ np.max(oe[x == ux] - rb[x == ux]) for ux in unique_x ]

        color = line_colors[i % len(line_colors)]
        D_formatted = "{:.0e}".format(D_values[i])

        ax[i, 0].plot(
            unique_x,
            diffs_overlap_vs_estimate_mean,
            label=f"D = {D_formatted}",
            color=color,
            marker="o",
            linestyle="--"
        )

        ax[i, 1].plot(
            unique_x,
            diffs_overlap_vs_bound_mean,
            color=color,
            marker="o",
            linestyle="--"
        )
        ax[i, 0].fill_between(unique_x, diffs_overlap_vs_estimate_min, diffs_overlap_vs_estimate_max, facecolor=color, alpha=0.3)
        ax[i, 1].fill_between(unique_x, diffs_overlap_vs_bound_min, diffs_overlap_vs_bound_max, facecolor=color, alpha=0.3)

        ax[i, 0].axhline(0,color='black', linestyle=':')
        ax[i, 1].axhline(0,color='black', linestyle=':')

        ax[i, 0].set_facecolor("#F9F9FB")
        ax[i, 1].set_facecolor("#F9F9FB")

    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    ax[0, 0].set_title(r'$\mathcal{F} - \mathcal{F}_{\text{est}}$')
    ax[0, 1].set_title(r'$\mathcal{F} - \mathcal{F}_{\text{LB}}$')
    ax[len(D_values) - 1, 0].set_xlabel(r"$\mathcal{F}_{\text{min}}$")
    ax[len(D_values) - 1, 1].set_xlabel(r"$\mathcal{F}_{\text{min}}$")
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=4, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.1))

    plt.savefig(
        plot_folder / f"overlap_comparison_n_{n_qubit}_vector.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    print("Overlap comparison plot saved")


def plot_overlap_difference():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_ov,
        d_arr,
        overlap_estimate,
        rigorous_bound,
        actual_overlap,
        num_gates_approx,
        num_gates_uniform,
        num_gates_exact,
    ) = _load_data()

    diff_est = overlap_estimate - actual_overlap
    diff_bound = rigorous_bound - actual_overlap

    _set_plot_style()
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d_arr, fixed_d)
        x = min_ov[mask]
        de = diff_est[mask]
        db = diff_bound[mask]

        unique_x = np.unique(x)
        means_de = np.array([np.mean(de[x == ux]) for ux in unique_x])
        stds_de = np.array([np.std(de[x == ux]) for ux in unique_x])
        means_db = np.array([np.mean(db[x == ux]) for ux in unique_x])
        stds_db = np.array([np.std(db[x == ux]) for ux in unique_x])

        color = line_colors[i % len(line_colors)]
        D_formatted = "{:.0e}".format(D_values[i])

        ax.errorbar(
            unique_x,
            means_de,
            yerr=stds_de,
            color=color,
            ls="-",
            marker="o",
            markersize=4,
            label=f"D = {D_formatted} (estimate error)",
        )

        ax.errorbar(
            unique_x,
            means_db,
            yerr=stds_db,
            color=color,
            ls="--",
            marker="^",
            markersize=4,
            label=f"D = {D_formatted} (bound error)",
        )

    ax.axhline(y=0, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("minimum allowed overlap")
    ax.set_ylabel("overlap error")
    ax.set_facecolor("#F9F9FB")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, alpha=0.3)

    outpath = plot_folder / f"overlap_error_n_{n_qubit}.pdf"
    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Overlap error plot saved: {outpath}")


def plot_gate_ratios():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        overlap_estimate,
        rigorous_bound,
        actual_overlap,
        num_gates_approx,
        num_gates_uniform,
        num_gates_exact,
    ) = _load_data()

    ratio_exact_uniform = num_gates_exact / num_gates_uniform
    ratio_approx_exact = num_gates_approx / num_gates_exact

    _set_plot_style()

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        y = ratio_exact_uniform[mask]

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
    ax.set_ylabel("CNOTs Exact / CNOTs Uniform")
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

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        y = ratio_approx_exact[mask]

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
    ax.set_ylabel("CNOTs Approx / CNOTs Exact")
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

    print("Gate ratio plots saved")


if __name__ == "__main__":
    collect(force_recollect=force)
    plot_overlap_comparison()
    plot_overlap_difference()
    plot_gate_ratios()