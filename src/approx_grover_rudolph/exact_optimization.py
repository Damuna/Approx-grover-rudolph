import numpy as np 

from .helping_functions import (
    ControlledRotationGateMap,
    _build_baseline_support,
    neighbour_dict,
    _pattern_matches,
    _branch_has_no_support
    )

__all__=["optimize_dict_support_aware_exact",
    "optimize_full_dict_support_aware_exact",
    ]


def _same_gate(v1, v2, tol=1e-12):
    return (abs(v1[0] - v2[0]) <= tol) and (abs(v1[1] - v2[1]) <= tol)


def _drop_unreachable_control_once(gate_operations, baseline_support, tol=1e-12):
    """
    Exact support-aware simplification:
    remove one control from a gate if the newly added branch has zero support.
    """
    for key, value in list(gate_operations.items()):
        for pos, ch in enumerate(key):
            if ch == "e":
                continue

            flipped = "1" if ch == "0" else "0"
            partner = key[:pos] + flipped + key[pos + 1 :]
            new_key = key[:pos] + "e" + key[pos + 1 :]

            if not _branch_has_no_support(partner, baseline_support):
                continue

            existing = gate_operations.get(new_key)
            if (existing is not None) and (not _same_gate(existing, value, tol)):
                # Cannot overwrite a different exact gate.
                continue

            gate_operations.pop(key)
            if new_key not in gate_operations:
                gate_operations[new_key] = value
            return True

    return False


def _merge_identical_neighbours_once(gate_operations, tol=1e-12):
    """
    Exact merge of two neighbouring gates with identical parameters.
    """
    for k1, v1 in list(gate_operations.items()):
        neighbours = neighbour_dict(k1)

        for k2, position in neighbours.items():
            if k2 not in gate_operations:
                continue
            if k2 <= k1:
                continue

            v2 = gate_operations[k2]
            if not _same_gate(v1, v2, tol):
                continue

            new_key = k1[:position] + "e" + k1[position + 1 :]
            gate_operations.pop(k1)
            gate_operations.pop(k2)
            gate_operations[new_key] = v1
            return True

    return False


def optimize_dict_support_aware_exact(
    gate_operations: ControlledRotationGateMap,
    baseline_support,
    tol: float = 1e-12,
) -> ControlledRotationGateMap:
    """
    Exact optimization using:
      1. standard identical-neighbour merges
      2. support-aware control removal on unreachable branches

    This preserves the prepared state exactly.
    """
    changed = True
    while changed:
        changed = False

        # First try the standard exact merge.
        if _merge_identical_neighbours_once(gate_operations, tol=tol):
            changed = True
            continue

        # Then try support-aware exact control removal.
        if _drop_unreachable_control_once(
            gate_operations, baseline_support, tol=tol
        ):
            changed = True
            continue

    return gate_operations


def optimize_full_dict_support_aware_exact(
    total_gate_operations: list[ControlledRotationGateMap],
    tol: float = 1e-12,
):
    """
    Exact support-aware optimization for the full Grover-Rudolph circuit.
    The baseline support is computed once from the original exact circuit.

    Note:
        This should be applied to the exact Grover-Rudolph circuit returned by
        build_dictionary, before any approximate merging.
    """
    baseline_layers = [dict(layer_dict) for layer_dict in total_gate_operations]
    baseline_support = _build_baseline_support(baseline_layers)

    final_gates = [dict(layer_dict) for layer_dict in total_gate_operations]

    for gate_operations in final_gates:
        optimize_dict_support_aware_exact(
            gate_operations,
            baseline_support=baseline_support,
            tol=tol,
        )

    return final_gates