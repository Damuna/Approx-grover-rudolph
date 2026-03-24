import numpy as np
from functools import reduce

from .helping_functions import (
    ZERO,
    ControlledRotationGateMap,
    number_of_qubits,
    sanitize_sparse_state_vector,
    StateVector,
    RotationGate,
)

from .approx_algorithm import (
    optimize_dict,
    optimize_full_dict,
)


__all__ = [
    "build_dictionary",
    "grover_rudolph",
    "GR_circuit",
]

GateCounts = np.ndarray
"""Gate counts stored as a numpy array with three integers: Tofolli, CNOT, 1-qubit"""

def build_dictionary(vector: StateVector) -> list[ControlledRotationGateMap]:
    """
    Generate a list of dictonaries for the angles given the amplitude vector
    Each dictonary is of the form:
        {key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy) : value = [angle, phase]
        {'00' : [1.2, 0.]} the gate is a rotation of 1.2 and a phase gate with phase 0, controlled on the state |00>

    You are basically building the cicuit vertically, where each element of the dictionary is one layer of the circuit
    if the dictonary is in position 'i' of the list (starting from 0), its key will be of length 'i', thus the controls act on the fist i qubits

    Args:
        vector: compressed version (only non-zero elements) of the sparse state vector to be prepared
        optimization: decide if optimize the angles or not, defaults to True
        optimization_error: the error in merging similar angles

    Returns:
        a sequence of controlled gates to be applied.
    """

    vector = sanitize_sparse_state_vector(vector)
    nonzero_values = vector.data
    nonzero_locations = vector.nonzero()[1]

    N_qubit = number_of_qubits(int(max(nonzero_locations)) + 1)

    final_gates: list[ControlledRotationGateMap] = []

    merging_count = 0

    for qbit in range(N_qubit):
        new_nonzero_values = []
        new_nonzero_locations = []

        gate_operations: ControlledRotationGateMap = {}
        sparsity = len(nonzero_locations)

        phases: np.ndarray = np.angle(nonzero_values)

        i = 0
        while i in range(sparsity):
            angle: float
            phase: float

            loc = nonzero_locations[i]

            # last step of the while loop
            if i + 1 == sparsity:
                new_nonzero_locations.append(loc // 2)
                if nonzero_locations[i] % 2 == 0:
                    # if the non_zero element is at the very end of the vector
                    angle = 0.0
                    phase = -phases[i]
                    new_nonzero_values.append(nonzero_values[i])
                else:
                    # if the non_zero element is second-last
                    angle = np.pi
                    phase = phases[i]
                    new_nonzero_values.append(abs(nonzero_values[i]))
            else:
                # divide the non_zero locations in pairs
                loc0 = nonzero_locations[i]
                loc1 = nonzero_locations[i + 1]

                # if the non_zero locations are consecutive, with the first one in an even position
                if (loc1 - loc0 == 1) and (loc0 % 2 == 0):
                    new_component = np.exp(1j * phases[i]) * np.sqrt(
                        abs(nonzero_values[i]) ** 2 + abs(nonzero_values[i + 1]) ** 2
                    )
                    new_nonzero_values.append(new_component)
                    new_nonzero_locations.append(loc0 // 2)

                    angle = (
                        2
                        * np.arccos(
                            np.clip(abs(nonzero_values[i] / new_component), -1, 1)
                        )
                        if abs(new_component) > ZERO
                        else 0.0
                    )
                    phase = -phases[i] + phases[i + 1]
                    i += 1
                else:
                    # the non_zero location is on the right of the pair
                    if loc0 % 2 == 0:
                        angle = 0.0
                        phase = -phases[i]
                        new_nonzero_values.append(nonzero_values[i])
                        new_nonzero_locations.append(loc0 // 2)

                    else:
                        angle = np.pi
                        phase = phases[i]
                        new_nonzero_values.append(abs(nonzero_values[i]))
                        new_nonzero_locations.append(loc0 // 2)

            i += 1

            # add in the dictionary gate_operations if they are not zero
            if abs(angle) > ZERO or abs(phase) > ZERO:
                # number of control qubits for the current rotation gates
                num_controls = N_qubit - qbit - 1
                gate: RotationGate = (angle, phase)

                if num_controls == 0:
                    gate_operations = {"": gate}
                else:
                    controls = str(bin(loc // 2)[2:]).zfill(num_controls)
                    gate_operations[controls] = gate

        nonzero_values, nonzero_locations = (new_nonzero_values, new_nonzero_locations)

        final_gates.append(gate_operations)

    final_gates.reverse()

    return final_gates


def grover_rudolph(
    vector: StateVector, *, optimization: bool = True, optimization_error: float = 0
) -> list[ControlledRotationGateMap]:
    """
    Generate a list of dictonaries for the angles given the amplitude vector
    Each dictonary is of the form:
        {key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy) : value = [angle, phase]
        {'00' : [1.2, 0.]} the gate is a rotation of 1.2 and a phase gate with phase 0, controlled on the state |00>

    You are basically building the cicuit vertically, where each element of the dictionary is one layer of the circuit
    if the dictonary is in position 'i' of the list (starting from 0), its key will be of length 'i', thus the controls act on the fist i qubits

    Args:
        vector: compressed version (only non-zero elements) of the sparse state vector to be prepared
        optimization: decide if optimize the angles or not, defaults to True
        optimization_error: the error in merging similar angles

    Returns:
        a sequence of controlled gates to be applied.
    """

    vector = sanitize_sparse_state_vector(vector)
    nonzero_values = vector.data
    nonzero_locations = vector.nonzero()[1]

    N_qubit = number_of_qubits(int(max(nonzero_locations)) + 1)

    final_gates: list[ControlledRotationGateMap] = []

    merging_count = 0

    for qbit in range(N_qubit):
        new_nonzero_values = []
        new_nonzero_locations = []

        gate_operations: ControlledRotationGateMap = {}
        sparsity = len(nonzero_locations)

        phases: np.ndarray = np.angle(nonzero_values)

        i = 0
        while i in range(sparsity):
            angle: float
            phase: float

            loc = nonzero_locations[i]

            # last step of the while loop
            if i + 1 == sparsity:
                new_nonzero_locations.append(loc // 2)
                if nonzero_locations[i] % 2 == 0:
                    # if the non_zero element is at the very end of the vector
                    angle = 0.0
                    phase = -phases[i]
                    new_nonzero_values.append(nonzero_values[i])
                else:
                    # if the non_zero element is second-last
                    angle = np.pi
                    phase = phases[i]
                    new_nonzero_values.append(abs(nonzero_values[i]))
            else:
                # divide the non_zero locations in pairs
                loc0 = nonzero_locations[i]
                loc1 = nonzero_locations[i + 1]

                # if the non_zero locations are consecutive, with the first one in an even position
                if (loc1 - loc0 == 1) and (loc0 % 2 == 0):
                    new_component = np.exp(1j * phases[i]) * np.sqrt(
                        abs(nonzero_values[i]) ** 2 + abs(nonzero_values[i + 1]) ** 2
                    )
                    new_nonzero_values.append(new_component)
                    new_nonzero_locations.append(loc0 // 2)

                    angle = (
                        2
                        * np.arccos(
                            np.clip(abs(nonzero_values[i] / new_component), -1, 1)
                        )
                        if abs(new_component) > ZERO
                        else 0.0
                    )
                    phase = -phases[i] + phases[i + 1]
                    i += 1
                else:
                    # the non_zero location is on the right of the pair
                    if loc0 % 2 == 0:
                        angle = 0.0
                        phase = -phases[i]
                        new_nonzero_values.append(nonzero_values[i])
                        new_nonzero_locations.append(loc0 // 2)

                    else:
                        angle = np.pi
                        phase = phases[i]
                        new_nonzero_values.append(abs(nonzero_values[i]))
                        new_nonzero_locations.append(loc0 // 2)

            i += 1

            # add in the dictionary gate_operations if they are not zero
            if abs(angle) > ZERO or abs(phase) > ZERO:
                # number of control qubits for the current rotation gates
                num_controls = N_qubit - qbit - 1
                gate: RotationGate = (angle, phase)

                if num_controls == 0:
                    gate_operations = {"": gate}
                else:
                    controls = str(bin(loc // 2)[2:]).zfill(num_controls)
                    gate_operations[controls] = gate

        nonzero_values, nonzero_locations = (new_nonzero_values, new_nonzero_locations)

        if optimization:
            gate_operations = optimize_dict(gate_operations, error=optimization_error)

        final_gates.append(gate_operations)

    final_gates.reverse()

    return final_gates


def GR_circuit(dict_list: list[ControlledRotationGateMap]) -> np.ndarray:
    """
    The same procedure is applied to the phases, and at the end the two dictionaries are merged together, taking into account the commutation rules.
    Build the circuit of the state preparation with as input the list of dictonaries (good to check if the preparation is succesfull)

    Returns:
         The final state and the number of gates needed (Number of Toffoli gates, 2qubits gate and 1qubit gate)
    """

    # Vector to apply the circuit to
    psi = np.zeros(2 ** len(dict_list))
    psi[0] = float(1)

    e0 = np.array([float(1), float(0)])  # zero state
    e1 = np.array([float(0), float(1)])

    Id = np.eye(2)

    control_matrix: dict[str, np.ndarray] = {
        "e": Id,
        "0": np.outer(e0, e0),
        "1": np.outer(e1, e1),
    }

    for i, gates in enumerate(dict_list):
        # Build the unitary for each dictonary
        for k, (theta, phase) in gates.items():
            if theta is None:
                R = Id
            else:
                R = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ]
                )

            if phase is None:
                P_phase = np.eye(2)
            else:
                P_phase = np.array([[1.0, 0.0], [0.0, np.exp(1j * phase)]])

            # tensor product of all the 2x2 control matrices
            P = reduce(np.kron, [control_matrix[s] for s in k], np.eye(1))

            U = np.kron(P, P_phase @ R) + np.kron(np.eye(2**i) - P, Id)

            extra = len(dict_list) - i - 1
            U = np.kron(U, np.eye(2**extra))

            psi = U @ psi

    return psi
