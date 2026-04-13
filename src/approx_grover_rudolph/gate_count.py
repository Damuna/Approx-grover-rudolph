import numpy as np

from .helping_functions import ControlledRotationGateMap

GateCounts = np.ndarray

__all__=[
    "single_rotation_count",
    "hybrid_CNOT_count",
    "_single_gate_cnot_cost",
    "_layer_single_cnot_cost",
    "_candidate_cnot_gain_local"
    ]

def _single_gate_cnot_cost(key: str) -> int:
    n_controls = key.count("0") + key.count("1")

    if n_controls == 0:
        return 0
    elif n_controls == 1:
        return 2
    else:
        return 16 * n_controls - 24


def _layer_single_cnot_cost(layer_dict: ControlledRotationGateMap) -> int:
    return sum(_single_gate_cnot_cost(k) for k in layer_dict)


def _candidate_cnot_gain_local(gate_operations, candidate, layer_single_costs=None):
    """
    Exact CNOT gain for the current hybrid_CNOT_count, computed from the
    affected layer only.
    """
    k1 = candidate["k1"]
    k2 = candidate["k2"]
    new_key = candidate["new_key"]

    layer_idx = len(k1)
    layer_dict = gate_operations[layer_idx]

    if layer_single_costs is None:
        before_single = _layer_single_cnot_cost(layer_dict)
    else:
        before_single = layer_single_costs[layer_idx]

    uniform_cost = 2 ** layer_idx

    after_single = before_single - _single_gate_cnot_cost(k1)
    if k2 is not None:
        after_single -= _single_gate_cnot_cost(k2)
    after_single += _single_gate_cnot_cost(new_key)

    before_hybrid = min(before_single, uniform_cost)
    after_hybrid = min(after_single, uniform_cost)

    return before_hybrid - after_hybrid


def single_rotation_count(
    total_gate_operations: list[ControlledRotationGateMap],
) -> GateCounts:
    """
    Counts how many gates you need to build the circuit for Grover Rudolph (optimized or not optimized, but without permutations) in terms of elemental ones (single rotation gates, one-control-one-target
    gates on the |1⟩ state, refered as 2 qubits gates, and Toffoli gates)

    Args:
        total_gate_operations = the list of dictionaries of the form dict[str] = [float,float], where str is made of '0','1','e'

    Returns:
        N_cnots
    """
    N_cnot = 0

    for gate_operations in total_gate_operations:
        # Build the unitary for each dictonary
        N_cnot_layer = 0

        for k in gate_operations:
            # Count number of controls
            count0 = k.count("0")
            count1 = k.count("1")
            N_controls = count0 + count1

            if N_controls == 0:
                N_cnot_layer += 0
            elif N_controls == 1:
                N_cnot_layer += 2
            else:
                N_cnot_layer += 16 * (N_controls) - 24

        N_cnot += N_cnot_layer

    return N_cnot


def hybrid_CNOT_count(
    total_gate_operations: list[ControlledRotationGateMap],
) -> int:
    N_cnot = 0

    for gate_operations in total_gate_operations:
        N_cnot_layer = 0

        n_qubits_layer = 0
        for k in gate_operations:
            count0 = k.count("0")
            count1 = k.count("1")
            N_controls = count0 + count1

            if N_controls == 0:
                N_cnot_layer += 0
            elif N_controls == 1:
                N_cnot_layer += 2
            else:
                N_cnot_layer += 16 * (N_controls) - 24

            n_qubits_layer = len(k)

        # Take the cheaper option: gate-by-gate vs uniform rotation
        uniform_cost = 2 ** n_qubits_layer
        N_cnot += min(N_cnot_layer, uniform_cost)

    return N_cnot