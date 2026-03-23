import numpy as np

from .helping_functions import (
    ControlledRotationGateMap,
    neighbour_dict,
    generate_strings,
    f_cs,
    replace_first_non_e,
)

__all__=[
"merging_formula",
"ordering_geometric_series",
"order_pairs_optimally",
"optimize_dict",
"run_one_merge_step",
"optimize_full_dict",
]


def merging_formula(controls, gate_operations, original_keys, eps):
    # Given a mearging of the keys k1 and k2 in the dictionary angles_phases, with precision epsilon, returns the new overlap

    B_set = generate_strings(controls)
    Delta = 0

    for bit_string in B_set:
        f_product = 1
        for i in range(len(bit_string)):
            b_i = bit_string[:i]
            # print("b", b_i)
            angles_phases = gate_operations[len(b_i)]
            if b_i in angles_phases:
                theta = angles_phases[b_i][0]
            elif b_i in original_keys:
                theta = original_keys[b_i][0]
            else:
                theta = 0.0
            f_product *= f_cs(theta, bit_string[i])
        Delta += f_product

    overlap = 1 - ((1 - np.cos(eps / 2)) * Delta)

    return overlap

def ordering_geometric_series(gate_operations, min_overlap, m_steps, error = np.pi/2):

    # Initialize
    flag = True
    overlap = 1.
    min_overlap_step = 1.
    original_keys = {}

    # Linear step
    step = (1 - min_overlap) / m_steps

    # Merge all possible merges, increasing min_overlap with a geometric series
    for i in range(m_steps):
        min_overlap_step -= step
        overlap, flag = order_pairs_optimally(gate_operations, min_overlap_step, error, overlap = overlap, original_keys = original_keys)

    # Take care of the remaing mergings if left
    while flag == True:
        overlap, flag = order_pairs_optimally(gate_operations, min_overlap, error, overlap = overlap, original_keys = original_keys)
    return overlap


def order_pairs_optimally(gate_operations, min_overlap, error, overlap = 1, original_keys = None):
    pairs = []
    if original_keys == None:
        original_keys = {}

    for angles_phases_dict in gate_operations:

        for k1, v1 in angles_phases_dict.items():
            neighbours = neighbour_dict(k1)

            # Merge with an imaginary (0,0) gate on the first control that is not 'e'
            if k1 != "e" * len(k1) and abs(v1[0]) <= 2 * error:
                new_key, merging_position = replace_first_non_e(k1)
                overlap_estimate = merging_formula(
                    new_key,
                    gate_operations,
                    original_keys,
                    eps=np.abs(v1[0]) / 2,
                )
                if abs(overlap - (1 - overlap_estimate)) >= min_overlap:
                    new_value = (v1[0] / 2, v1[1] / 2)
                    pairs.append(
                        (k1, None, new_key, new_value, overlap_estimate, np.abs(v1[0]) / 2)
                    )

            # Try merging with other gates
            for k2, position in neighbours.items():
                if k2 not in angles_phases_dict:
                    continue

                v2 = angles_phases_dict[k2]
                eps_theta = abs(v1[0] - v2[0]) / 2
                eps_phi = abs(v1[1] - v2[1]) / 2

                # Consider only different items with same angle and phase up to some error
                if (eps_theta > error) or (eps_phi > error):
                    continue

                new_key = k1[:position] + "e" + k1[position + 1 :]
                new_value = (
                    min(v2[0], v1[0]) + (np.abs(v2[0] - v1[0]) / 2),
                    min(v2[1], v1[1]) + (np.abs(v2[1] - v1[1]) / 2),
                )

                overlap_estimate = merging_formula(
                    new_key,
                    gate_operations,
                    original_keys,
                    eps=np.abs(v1[0] - v2[0]) / 2,
                )

                if abs(overlap - (1 - overlap_estimate)) >= min_overlap:
                    pairs.append(
                        (k1, k2, new_key, new_value, overlap_estimate, np.abs(v1[0] - v2[0]) / 2)
                    )
    if pairs == []:
        return overlap, False

    # Sort pairs by overlap (key at index 4)
    pairs = sorted(pairs, key=lambda x: x[4], reverse=True)

    for k1, k2, new_key, new_value, overlap_estimate, eps in pairs:
        angles_phases_dict = gate_operations[len(k1)]
        if (k1 in angles_phases_dict) and (k2 in angles_phases_dict or k2 == None):
            # Compute trace distance exactly (with merged values)
            overlap_merging = merging_formula(
                new_key,
                gate_operations,
                original_keys,
                eps=eps,
            )
            overlap = abs(overlap - (1 - overlap_merging))

            # Stop merging, since you have reached the desired precision
            if overlap < min_overlap:
                return abs(overlap + (1 - overlap_merging)), False

            # Merge k1 and k2
            angles_phases_dict.pop(k1)
            if k2 != None:
                angles_phases_dict.pop(k2)
            angles_phases_dict[new_key] = new_value

            # Update original keys with merged value
            original_keys[k1] = new_value
            if k2 != None:
                original_keys[k2] = new_value

    # Mergings done
    return overlap, True


def optimize_dict(
    gate_operations: ControlledRotationGateMap,
    error,
) -> ControlledRotationGateMap:
    """
    Optimize the dictionary by merging some gates in one:
    if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
    >> {'11':[3.14,0] ; '10':[3.14,0]} becomes {'1e':[3.14,0]} where 'e' means no control (identity)

    >>> assert optimize_dict({"11": (3.14, 0), "10": (3.14, 0)}) == {"1e": (3.14, 0)}

    Args:
        gate_operations: collection of controlled gates to be applied
    Returns:
        optimized collection of controlled gates
    """
    merged = True

    while merged:
        merged = run_one_merge_step(gate_operations, error)

    return gate_operations


def run_one_merge_step(gate_operations: ControlledRotationGateMap, error) -> bool:
    """
    Run a single merging step, modifying the input dictionary.

    Args:
        gate_operations: collection of controlled gates to be applied
    Returns:
        True if some merge happened
    """

    if len(gate_operations) <= 1:
        return False

    for k1, v1 in gate_operations.items():
        neighbours = neighbour_dict(k1)

        for k2, position in neighbours.items():
            if k2 not in gate_operations:
                continue

            v2 = gate_operations[k2]

            # Consider only different items with same angle and phase up to sone error
            if (abs(v1[0] - v2[0]) / 2 > error) or (abs(v1[1] - v2[1]) / 2 > error):
                continue

            # Replace the different char with 'e' and remove the old items
            gate_operations.pop(k1)
            gate_operations.pop(k2)
            eps = np.abs(v2[0] - v1[0]) / 2
            gate_operations[k1[:position] + "e" + k1[position + 1 :]] = (
                min(v2[0] + eps, v1[0] + eps),
                min(v2[1] + eps, v1[1] + eps),
            )
            return True

    return False


def optimize_full_dict(
    total_gate_operations: list[ControlledRotationGateMap],
    optimization_error: float = 0,
):
    final_gates: list[ControlledRotationGateMap] = []

    for gate_operations in total_gate_operations:
        gate_operations = optimize_dict(gate_operations, error=optimization_error)
        final_gates.append(gate_operations)

    return final_gates