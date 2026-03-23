from pathlib import Path
import copy
import functools
import multiprocessing as mp
import time
import numpy as np

from approx_grover_rudolph import (
    generate_sparse_unit_vector,
    build_dictionary,
    optimize_full_dict,
    ordering_geometric_series,
    hybrid_CNOT_count,
    single_rotation_count,
)

# always flush the stream when using print so parallel output also works when writing to a file stream (which is often the case for IDE output)
print = functools.partial(print, flush=True)


# Fixed Data
M = 20
repeat = 1
n_points = 5
vec_type = "real"
sparsity = 100
n_values = [5]

# Data for sparse plots
sparse_optimization = True
n_qubit = 20
sparse_optimization = True
d_values = [10, 100, 500, 1000]
min_overlap_values = np.linspace(0.8, 1, num=n_points)

# Data for not sparse plot
# sparse_optimization = False
# n_qubit = 20
# d_values = np.linspace(10, 5000, num=n_points)
# sparse_optimization = False
# min_overlap_values = [1.0]

# Folders
ROOT = Path(__file__).resolve().parents[1]
data_folder = ROOT / "data"
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist


# this function does the whole computation and then returns a string that can be written to the data file
# depending on n_fixed, it will write the correct value in the second column
def compute_values(
    min_overlap: float,
    n_qubits: int,
    sparsity: int,
    n_fixed: bool,
    sparse_optimization: bool,
):
    # vector generation
    psi = generate_sparse_unit_vector(n_qubit, sparsity, vector_type=vec_type)

    # original dictionary
    angles_phases = build_dictionary(psi)
    angles_phases_copy = copy.deepcopy(angles_phases)

    # eps = 0 optimization dictionary
    angle_phases_zero = optimize_full_dict(angles_phases_copy)
    num_gates_eps_zero = hybrid_CNOT_count(angle_phases_zero)
    num_gates_uniform = (2**n_qubit) - 1
    num_gates_eps_zero_single_rotation = single_rotation_count(angle_phases_zero)

    # optimization dictionary
    if sparse_optimization:
        overlap_estimate = ordering_geometric_series(angles_phases, min_overlap, M)
        num_gates_approx = hybrid_CNOT_count(angles_phases)
    else:
        num_gates_approx = 1.0
        overlap_estimate = 1.0

    if n_fixed:
        return f"{min_overlap}\t{sparsity}\t{overlap_estimate}\t{num_gates_approx}\t{num_gates_uniform}\t{num_gates_eps_zero}\t{num_gates_eps_zero_single_rotation}\n"
    else:
        return f"{min_overlap}\t{sparsity}\t{overlap_estimate}\t{num_gates_approx}\t{num_gates_uniform}\t{num_gates_eps_zero}\t{num_gates_eps_zero_single_rotation}\n"


def output_progress(results, n_fixed: bool):
    # Use the actual number of min_overlap_values instead of n_points
    inner_loop_size = len(d_values) if n_fixed else len(n_values)
    n_outer_loop = len(min_overlap_values)  # This is now 1 instead of 20
    n_iterations_total = repeat * n_outer_loop * inner_loop_size  # 20 * 1 * 20 = 400

    one_percent = n_iterations_total / 100

    n_completed = 0
    n_percent_old = 0
    # we count the number of completed tasks using the ready() method which returns if a task
    # output progress until all tasks are completed
    # output the current percentage value each second (if it has changed)
    # this means that the output might skip some values if the computation is fast
    # the performance overhead of this method is that for very large datasets the counting might
    # need a long time to finish so one process is used up for the progress most of the time
    while n_completed < n_iterations_total:
        # use a generator comprehension to count the number of completed results so we do not unnecessarily build a big list
        n_completed = sum(1 for result in results if result.ready())
        n_percent = int(n_completed / n_iterations_total * 100)
        if n_percent > n_percent_old:
            print(f"{n_percent}% finished")
        n_percent_old = n_percent
        time.sleep(1)


if __name__ == "__main__":
    n_processes = 8

    print("---------------FIXING N-------------------")
    results = []
    n_fixed = True

    with mp.Pool(n_processes) as pool:
        for i in range(repeat):
            for min_overlap in min_overlap_values:
                for d in d_values:
                    result = pool.apply_async(
                        compute_values,
                        (min_overlap, n_qubit, d, n_fixed, sparse_optimization),
                    )

                    results.append(result)

        # close the pool so no additional tasks can be added (this needs to be done before join())
        pool.close()

        output_progress(results, n_fixed)

        # join the pool (wait for all tasks to finish)
        pool.join()

    print("Writing results to file ...")
    # collect the results and write them to a file in a single process
    with open(data_folder / f"ordering_alg_n_{n_qubit}.tsv", "w") as f:
        for result in results:
            f.write(result.get())
    print()

    # print("----------------FIXING D------------------------")
    # results = []
    # n_fixed=False

    # with mp.Pool(n_processes) as pool:
    #     for i in range(repeat):
    #         for min_overlap in min_overlap_values:
    #             for n in n_values:
    #                 result = pool.apply_async(compute_values, (min_overlap, n, sparsity, n_fixed))
    #                 results.append(result)

    #     # close the pool so no additional tasks can be added (this needs to be done before join())
    #     pool.close()

    #     output_progress(results, n_fixed)

    #     # join the pool (wait for all tasks to finish)
    #     pool.join()

    # print("Writing results to file ...")
    # # collect the results and write them to a file in a single process
    # with open(data_folder / f"ordering_alg_n_{n_qubit}.tsv", "w") as f:
    #     for result in results:
    #         f.write(result.get())
