import numpy as np
import scipy.sparse as sp


from .helping_functions import (
    ZERO,
    ControlledRotationGateMap,
    sanitize_sparse_state_vector,
    StateVector,
    RotationGate,
)

__all__ = [
    "build_dictionary",
    "GR_circuit_sparse",
]


def build_dictionary(vector: StateVector, N_qubit) -> list[ControlledRotationGateMap]:
    """
    Generate a list of dictionaries for the angles given the amplitude vector.
    """
    vector = sanitize_sparse_state_vector(vector)
    nonzero_values = vector.data
    nonzero_locations = vector.nonzero()[1]

    final_gates: list[ControlledRotationGateMap] = []

    for qbit in range(N_qubit):
        new_nonzero_values = []
        new_nonzero_locations = []

        gate_operations: ControlledRotationGateMap = {}
        sparsity = len(nonzero_locations)
        phases: np.ndarray = np.angle(nonzero_values)

        i = 0
        while i in range(sparsity):
            loc = nonzero_locations[i]

            if i + 1 == sparsity:
                new_nonzero_locations.append(loc // 2)
                if nonzero_locations[i] % 2 == 0:
                    angle = 0.0
                    phase = -phases[i]
                    new_nonzero_values.append(nonzero_values[i])
                else:
                    angle = np.pi
                    phase = phases[i]
                    new_nonzero_values.append(abs(nonzero_values[i]))
            else:
                loc0 = nonzero_locations[i]
                loc1 = nonzero_locations[i + 1]

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

            if abs(angle) > ZERO or abs(phase) > ZERO:
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


def _pattern_matches_index(index: int, pattern: str, n_qubit: int) -> bool:
    for pos, ch in enumerate(pattern):
        if ch == "e":
            continue
        bitpos = n_qubit - 1 - pos
        bit = (index >> bitpos) & 1
        if bit != int(ch):
            return False
    return True


def GR_circuit_sparse(
    dict_list: list[ControlledRotationGateMap],
    *,
    atol: float = 1e-15,
    return_sparse: bool = False,
):
    """
    Sparse simulation of GR_circuit for the real-vector case.
    """
    n_qubit = len(dict_list)
    psi: dict[int, float] = {0: 1.0}

    for i, gates in enumerate(dict_list):
        target_bitpos = n_qubit - 1 - i

        for pattern, (theta, phase) in gates.items():
            if theta is None and phase is None:
                continue

            c = 1.0 if theta is None else float(np.cos(theta / 2))
            s = 0.0 if theta is None else float(np.sin(theta / 2))

            if phase is None:
                phase_factor = 1.0
            else:
                phase_factor = np.exp(1j * phase)
                if abs(np.imag(phase_factor)) > atol:
                    raise ValueError(
                        "GR_circuit_sparse only supports the real-vector case. "
                        "Found a genuinely complex phase."
                    )
                phase_factor = float(np.real_if_close(phase_factor))

            updates: dict[int, float] = {}

            for idx0, amp0 in list(psi.items()):
                if ((idx0 >> target_bitpos) & 1) != 0:
                    continue
                if not _pattern_matches_index(idx0, pattern, n_qubit):
                    continue

                idx1 = idx0 | (1 << target_bitpos)
                amp1 = psi.get(idx1, 0.0)

                new0 = c * amp0 - s * amp1
                new1 = phase_factor * (s * amp0 + c * amp1)

                updates[idx0] = float(np.real_if_close(new0))
                updates[idx1] = float(np.real_if_close(new1))

            for idx, value in updates.items():
                if abs(value) < atol:
                    psi.pop(idx, None)
                else:
                    psi[idx] = value

    cols = np.fromiter(psi.keys(), dtype=np.int64, count=len(psi))
    data = np.fromiter(psi.values(), dtype=float, count=len(psi))

    if return_sparse:
        rows = np.zeros(len(data), dtype=np.int64)
        return sp.csr_matrix((data, (rows, cols)), shape=(1, 2**n_qubit))

    out = np.zeros(2**n_qubit, dtype=float)
    out[cols] = data
    return out
