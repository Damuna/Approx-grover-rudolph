from .helping_functions import (
    ControlledRotationGateMap,
    _build_baseline_support,
    neighbour_dict,
    _branch_has_no_support,
)

__all__ = [
    "optimize_full_dict_support_aware_exact",
]


def _same_gate(v1, v2, tol=1e-12):
    return (abs(v1[0] - v2[0]) <= tol) and (abs(v1[1] - v2[1]) <= tol)


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


def strip_zero_support_controls_maximally(key, baseline_support):
    changed = True
    while changed:
        changed = False
        for pos, ch in enumerate(key):
            if ch == "e":
                continue

            flipped = "1" if ch == "0" else "0"
            partner = key[:pos] + flipped + key[pos + 1 :]

            if _branch_has_no_support(partner, baseline_support):
                key = key[:pos] + "e" + key[pos + 1 :]
                changed = True
                break

    return key


def optimize_dict_support_aware_exact(gate_operations, baseline_support, tol=1e-12):
    stripped = {}
    for key, value in list(gate_operations.items()):
        new_key = strip_zero_support_controls_maximally(key, baseline_support)

        existing = stripped.get(new_key)
        if existing is not None and not _same_gate(existing, value, tol):
            raise ValueError(
                f"Collision on key {new_key} with different gate values."
            )
        stripped[new_key] = value

    gate_operations.clear()
    gate_operations.update(stripped)

    while _merge_identical_neighbours_once(gate_operations, tol=tol):
        pass

    return gate_operations


def optimize_full_dict_support_aware_exact(
    total_gate_operations: list[ControlledRotationGateMap],
    tol: float = 1e-12,
):
    """
    Exact support-aware optimization for the full Grover-Rudolph circuit.
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
