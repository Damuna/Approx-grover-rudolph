import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
from collect_data import (
    n_qubit,
    sparsity,
    n_values,
    d_values,
    repeat,
    n_points,
)

matplotlib.rcParams.update({"font.size": 15})

line_colors = ["#2D2F92", "#DC3977", "#FBB982", "#009473"]


ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
plot_folder = ROOT / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)


# ------------------ FIXING N ----------------------

(
    min_overlap,
    d,
    overlap_estimate,
    num_gates_approx,
    num_gates_uniform,
    num_gates_eps_zero,
    num_gates_eps_zero_single_rot,
) = np.loadtxt(data_folder / f"ordering_alg_n_{n_qubit}.npy", unpack=True)

# Compute ratios
ratio_uniform = num_gates_approx / num_gates_uniform
ratio_eps_zero = num_gates_approx / num_gates_eps_zero


def plot_vs_min_overlap_fixed_d(y_dict, ylabel, filename):
    plt.figure(figsize=(10, 7))
    for j, (fixed_d, values) in enumerate(y_dict.items()):
        color = line_colors[j % len(line_colors)]
        plt.errorbar(
            values["x"],
            values["mean"],
            yerr=values["std"],
            label=f"d = {fixed_d}",
            color=color,
        )
    plt.xlabel("Min overlap allowed")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder / filename)


def group_by_fixed_d(y_values):
    grouped = {}
    for j, fixed_d in enumerate(d_values):
        mask = np.isclose(d, fixed_d)
        x1 = min_overlap[mask]
        y = y_values[mask]
        unique_x1 = np.unique(x1)
        means = [np.mean(y[x1 == ux1]) for ux1 in unique_x1]
        stds = [np.std(y[x1 == ux1]) for ux1 in unique_x1]
        grouped[fixed_d] = {"x": unique_x1, "mean": means, "std": stds}
    return grouped


# Plot ratios for fixed N
plot_vs_min_overlap_fixed_d(
    group_by_fixed_d(ratio_uniform),
    "CNOTs Approx / CNOTs Uniform",
    f"ratio_uniform_n_{n_qubit}.pdf",
)

plot_vs_min_overlap_fixed_d(
    group_by_fixed_d(ratio_eps_zero),
    "CNOTs Approx / CNOTs ε=0",
    f"ratio_eps_zero_n_{n_qubit}.pdf",
)


# Function to group by d (for fixed N) and compute mean & std
def group_by_d(y_values):
    grouped = {}
    for fixed_d in np.unique(d):
        mask = np.isclose(d, fixed_d)
        y = y_values[mask]
        mean = np.mean(y)
        std = np.std(y)
        grouped[fixed_d] = {"mean": mean, "std": std}
    return grouped


# Plot num_gates_uniform and num_gates_eps_zero vs d with averaging
group_uniform = group_by_d(num_gates_uniform)
# group_eps_zero = group_by_d(num_gates_eps_zero_single_rot)

plt.figure(figsize=(10, 7))
d_unique = sorted(group_uniform.keys())
means_uniform = [group_uniform[di]["mean"] for di in d_unique]
stds_uniform = [group_uniform[di]["std"] for di in d_unique]
# means_eps = [group_eps_zero[di]["mean"] for di in d_unique]
# stds_eps = [group_eps_zero[di]["std"] for di in d_unique]

plt.errorbar(
    d_unique,
    means_uniform,
    yerr=stds_uniform,
    fmt="o-",
    label="Uniform",
    color="#2D2F92",
)
# plt.errorbar(
#     d_unique,
#     means_eps,
#     yerr=stds_eps,
#     fmt="s--",
#     label="Single Rotations w/ mergings",
#     color="#DC3977",
# )
plt.xlabel("d")
plt.ylabel("Number of CNOT gates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_folder / f"num_gates_vs_d_n_{n_qubit}.pdf")
plt.show()


# # ------------------ FIXING D ----------------------

# (
#     min_overlap,
#     n,
#     overlap_estimate,
#     num_gates_approx,
#     num_gates_uniform,
#     num_gates_eps_zero,
# ) = np.loadtxt(os.path.join(data_folder, f"ordering_alg_d_{sparsity}.npy"), unpack=True)

# ratio_uniform = num_gates_approx / num_gates_uniform
# ratio_eps_zero = num_gates_approx / num_gates_eps_zero


# def group_by_fixed_n(y_values):
#     grouped = {}
#     for fixed_n in n_values:
#         mask = np.isclose(n, fixed_n)
#         x1 = min_overlap[mask]
#         y = y_values[mask]
#         unique_x1 = np.unique(x1)
#         means = [np.mean(y[x1 == ux1]) for ux1 in unique_x1]
#         stds = [np.std(y[x1 == ux1]) for ux1 in unique_x1]
#         grouped[fixed_n] = {"x": unique_x1, "mean": means, "std": stds}
#     return grouped


# plot_vs_min_overlap_fixed_d(group_by_fixed_n(ratio_uniform),
#     "CNOTs Approx / CNOTs Uniform", f"ratio_uniform_d_{sparsity}.pdf")

# plot_vs_min_overlap_fixed_d(group_by_fixed_n(ratio_eps_zero),
#     "CNOTs Approx / CNOTs ε=0", f"ratio_eps_zero_d_{sparsity}.pdf")

# # Plot num_gates_uniform and num_gates_eps_zero vs n (no dependence on min_overlap)
# plt.figure(figsize=(10, 7))
# plt.plot(n, num_gates_uniform, "o-", label="Uniform", color="#2D2F92")
# plt.plot(n, num_gates_eps_zero, "s--", label="ε=0", color="#DC3977")
# plt.xlabel("n")
# plt.ylabel("Number of CNOT gates")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(plot_folder / f"num_gates_vs_n_d_{sparsity}.pdf")
# plt.show()
