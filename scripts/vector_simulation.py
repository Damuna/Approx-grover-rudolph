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
        regional_merges=False,
    )

    num_gates_approx = hybrid_CNOT_count(approx_angles)

    # Sparse overlap
    psi_approx_sparse = GR_circuit_sparse(approx_angles, return_sparse=True)
    actual_overlap = abs(psi.conj().multiply(psi_approx_sparse).sum())

    if rigorous_bound is not None and rigorous_bound - actual_overlap > 1e-2:
        print("=" * 120)
        print(f"LOWER BOUND VIOLATION | min_overlap={min_overlap:.6f}, d={sparsity}")
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
        # "figure.titlesize": BIGGER_SIZE,
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


COL_WIDTH = 3.4  # inches, good for IEEE one-column


def _set_plot_style_ieee():
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    }
    plt.rcParams.update(params)


def _summary_by_x(x, y):
    ux = np.unique(x)
    med = np.array([np.median(y[np.isclose(x, u)]) for u in ux])
    q25 = np.array([np.quantile(y[np.isclose(x, u)], 0.25) for u in ux])
    q75 = np.array([np.quantile(y[np.isclose(x, u)], 0.75) for u in ux])
    return ux, med, q25, q75


def plot_style_stacked_panels():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        overlap_estimate,
        rigorous_bound,
        actual_overlap,
        *_,
    ) = _load_data()

    _set_plot_style_ieee()

    fig, axes = plt.subplots(
        len(d_values),
        1,
        figsize=(COL_WIDTH, 1.45 * len(d_values) + 0.25),
        sharex=True,
        constrained_layout=True,
    )

    if len(d_values) == 1:
        axes = [axes]

    for i, fixed_d in enumerate(d_values):
        ax = axes[i]
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]

        ux, ao_med, ao_q25, ao_q75 = _summary_by_x(x, actual_overlap[mask])
        _, oe_med, oe_q25, oe_q75 = _summary_by_x(x, overlap_estimate[mask])
        _, rb_med, rb_q25, rb_q75 = _summary_by_x(x, rigorous_bound[mask])

        ax.plot(ux, ux, color="0.55", ls=":", lw=0.9, label=r"$\mathcal{F}_{\min}$")
        ax.plot(ux, ao_med, color="black", lw=1.4, label="true overlap")
        ax.fill_between(ux, ao_q25, ao_q75, color="black", alpha=0.10, linewidth=0)

        ax.plot(
            ux,
            oe_med,
            color="0.30",
            lw=1.0,
            marker="o",
            ms=2.5,
            mfc="white",
            label="estimate",
        )
        ax.fill_between(ux, oe_q25, oe_q75, color="0.30", alpha=0.08, linewidth=0)

        ax.plot(ux, rb_med, color="black", lw=1.1, ls="--", label="lower bound")
        ax.fill_between(ux, rb_q25, rb_q75, color="black", alpha=0.06, linewidth=0)

        ax.set_ylim(0.72, 1.01)
        ax.grid(True, alpha=0.20, lw=0.4)
        ax.text(
            0.03,
            0.08,
            rf"$D={D_values[i]:.0e}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

        if i == 0:
            ax.legend(
                loc="lower right",
                frameon=True,
                borderpad=0.3,
                handlelength=1.8,
            )

    axes[-1].set_xlabel(r"minimum allowed overlap $\mathcal{F}_{\min}$")
    fig.supylabel("overlap")

    outpath = plot_folder / f"overlap_stacked_panels_n_{n_qubit}.pdf"
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {outpath}")


def plot_style_parity():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        overlap_estimate,
        rigorous_bound,
        actual_overlap,
        *_,
    ) = _load_data()

    _set_plot_style_ieee()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(COL_WIDTH, 4.2),
        constrained_layout=True,
        sharex=False,
    )

    markers = ["o", "s", "^"]

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        color = line_colors[i % len(line_colors)]

        axes[0].scatter(
            actual_overlap[mask],
            overlap_estimate[mask],
            s=10,
            alpha=0.55,
            color=color,
            marker=markers[i],
            label=rf"$D={D_values[i]:.0e}$",
        )
        axes[1].scatter(
            actual_overlap[mask],
            rigorous_bound[mask],
            s=10,
            alpha=0.55,
            color=color,
            marker=markers[i],
            label=rf"$D={D_values[i]:.0e}$",
        )

    for ax, ylabel in zip(axes, ["overlap estimate", "rigorous lower bound"]):
        lo = min(
            np.min(actual_overlap), np.min(rigorous_bound), np.min(overlap_estimate)
        )
        lo = max(0.72, lo - 0.01)
        hi = 1.005
        ax.plot([lo, hi], [lo, hi], color="0.5", ls=":", lw=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.20, lw=0.4)

    axes[1].set_xlabel("true overlap")
    axes[0].legend(loc="lower right", frameon=True, borderpad=0.3)

    outpath = plot_folder / f"overlap_parity_n_{n_qubit}.pdf"
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {outpath}")


def plot_style_gaps():
    if not FILEPATH.exists():
        print(f"No data file found: {FILEPATH}")
        return

    (
        min_overlap,
        d,
        overlap_estimate,
        rigorous_bound,
        actual_overlap,
        *_,
    ) = _load_data()

    _set_plot_style_ieee()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(COL_WIDTH, 4.2),
        sharex=True,
        constrained_layout=True,
    )

    for i, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x = min_overlap[mask]
        color = line_colors[i % len(line_colors)]

        gap_true = actual_overlap[mask] - x
        gap_bound = rigorous_bound[mask] - x
        err_est = overlap_estimate[mask] - actual_overlap[mask]
        slack_bound = actual_overlap[mask] - rigorous_bound[mask]

        ux, gt_med, gt_q25, gt_q75 = _summary_by_x(x, gap_true)
        _, gb_med, gb_q25, gb_q75 = _summary_by_x(x, gap_bound)
        _, ee_med, ee_q25, ee_q75 = _summary_by_x(x, err_est)
        _, sb_med, sb_q25, sb_q75 = _summary_by_x(x, slack_bound)

        axes[0].plot(
            ux,
            gt_med,
            color=color,
            lw=1.3,
            label=rf"$D={D_values[i]:.0e}$ true$-\mathcal{{F}}_{{\min}}$",
        )
        axes[0].plot(
            ux,
            gb_med,
            color=color,
            lw=1.1,
            ls="--",
            label=rf"$D={D_values[i]:.0e}$ bound$-\mathcal{{F}}_{{\min}}$",
        )

        axes[1].plot(
            ux,
            ee_med,
            color=color,
            lw=1.2,
            marker="o",
            ms=2.4,
            mfc="white",
            label=rf"$D={D_values[i]:.0e}$ est.-true",
        )
        axes[1].plot(
            ux,
            sb_med,
            color=color,
            lw=1.1,
            ls="--",
            label=rf"$D={D_values[i]:.0e}$ true-bound",
        )

    for ax in axes:
        ax.axhline(0, color="0.5", ls=":", lw=0.9)
        ax.grid(True, alpha=0.20, lw=0.4)

    axes[0].set_ylabel(r"gap to $\mathcal{F}_{\min}$")
    axes[1].set_ylabel("error / slack")
    axes[1].set_xlabel(r"minimum allowed overlap $\mathcal{F}_{\min}$")

    axes[0].legend(loc="lower left", frameon=True, borderpad=0.25, ncol=1)
    axes[1].legend(loc="upper left", frameon=True, borderpad=0.25, ncol=1)

    outpath = plot_folder / f"overlap_gaps_n_{n_qubit}.pdf"
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    collect(force_recollect=force)
    # plot_overlap_comparison()
    # plot_overlap_difference()
    plot_style_parity()
    plot_style_gaps()
    plot_style_stacked_panels()
