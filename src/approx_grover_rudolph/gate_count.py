import numpy as np

from .helping_functions import ControlledRotationGateMap

GateCounts = np.ndarray

__all__=[
    "single_rotation_count",
    "hybrid_CNOT_count"
    ]


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

        # Uniform rotation counting
        if N_cnot_layer > 2 ** len(k):
            N_cnot_layer = 2 ** len(k)

        N_cnot += N_cnot_layer

    return N_cnot
