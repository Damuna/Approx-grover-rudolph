import numpy as np

from .helping_functions import ControlledRotationGateMap

GateCounts = np.ndarray

__all__ = [
    "single_rotation_count",
    "hybrid_CNOT_count",
]


def _single_gate_cnot_cost(key: str) -> int:
    n_controls = key.count("0") + key.count("1")

    if n_controls == 0:
        return 0
    if n_controls == 1:
        return 2
    return 16 * n_controls - 24


def single_rotation_count(
    total_gate_operations: list[ControlledRotationGateMap],
) -> GateCounts:
    """
    Count the CNOT cost of implementing every controlled rotation gate directly.
    """
    return sum(
        _single_gate_cnot_cost(k)
        for gate_operations in total_gate_operations
        for k in gate_operations
    )


def hybrid_CNOT_count(
    total_gate_operations: list[ControlledRotationGateMap],
) -> int:
    total = 0

    for gate_operations in total_gate_operations:
        layer_cost = sum(_single_gate_cnot_cost(k) for k in gate_operations)
        n_qubits_layer = max((len(k) for k in gate_operations), default=0)
        uniform_cost = 2 ** n_qubits_layer
        total += min(layer_cost, uniform_cost)

    return total
