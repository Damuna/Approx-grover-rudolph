from typing import Sized, Union
from itertools import product

import numpy as np
import scipy as sp


__all__ = [
    "RotationGate",
    "ControlledRotationGateMap",
    "StateVector",
    "ZERO",
    "neighbour_dict",
    "generate_sparse_unit_vector",
    "number_of_qubits",
    "sanitize_sparse_state_vector",
    "replace_first_non_e",
    "f_cs",
    "generate_strings",
    "hybrid_CNOT_count",
]

# Some useful type aliases:
RotationGate = tuple[float, float]
r"""(\theta, \phi) pair describing a rotation gate defined by

.. math::

    Ry(\theta) \cdot P(\phi)
"""

Controls = str
"""a sequence of control bits. each bit is one of {0, 1, e}"""

ControlledRotationGateMap = dict[Controls, RotationGate]
"""keys are control bits, target is a rotation gate description"""

StateVector = Union[np.ndarray, sp.sparse.spmatrix, list[float]]
"""A row vector representing a quantum state"""

ZERO = 1e-8
"""global zero precision"""


def neighbour_dict(controls: Controls) -> dict[Controls, int]:
    """
    Finds the neighbours of a string (ignoring e), i.e. the mergeble strings
    Returns a dictionary with as keys the neighbours and as value the position in which they differ

    >>> assert neighbour_dict("10") == {"00": 0, "11": 1}
    >>> assert neighbour_dict("1e") == {'0e': 0}

    Args:
        controls: string made of '0', '1', 'e'
    Returns:
        A dictionary {control-string: swapped-index}
    """
    neighbours = {}
    for i, c in enumerate(controls):
        if c == "e":
            continue

        c_opposite = "1" if c == "0" else "0"
        key = controls[:i] + c_opposite + controls[i + 1 :]
        neighbours[key] = i

    return neighbours


def generate_sparse_unit_vector(
    n_qubit: int, d: int, *, vector_type: str = "complex"
) -> sp.sparse.spmatrix:
    """
    Generate random complex amplitudes vector of N qubits (length 2^N) with sparsity d
    as couples: position and value of the i-th non zero element
    The sign of the first entry  is  always real positive to fix the overall phase

    Args:
        n_qubit: number of qubits
        d: number of non-zero entries required in the output state vector
        vector_type: refers to the type of the state to be prepared.
                     'complex' generates complex random vectors, 'real' random real vector, and 'uniform' random uniform vector.

    Returns:
        A state vector stored as a scipy.sparse.spmatrix object with shape (1, 2**n_qubit), having exactly d non-zero elements.
    """
    N = 2**n_qubit

    if d > N:
        raise ValueError(
            "Sparsity must be less or equal than the dimension of the vector"
        )

    if vector_type == "complex":
        sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype="complex")
    elif vector_type == "real":
        sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype="float")
    elif vector_type == "negative_real":
        sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype="float")
        sparse_v.data = 2 * sparse_v.data - 1
    elif vector_type == "uniform":
        sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype="float")
        sparse_v.data[:] = 1.0
    else:
        raise ValueError(
            "Invalid input for the variable state_vector, select complex, real, negative_real, uniform"
        )

    sparse_v /= sp.linalg.norm(sparse_v.data)

    return sparse_v


def replace_first_non_e(s):
    # Convert string to a list since strings are immutable in Python
    s_list = list(s)

    # Iterate through the string
    for i in range(len(s_list)):
        if s_list[i] != "e":
            s_list[i] = "e"  # Replace the first non-'e' character
            break

    # Convert the list back to a string
    return "".join(s_list), i


def number_of_qubits(vec: int | Sized) -> int:
    """number of qubits needed to represent the vector/vector size."""
    sz: int = vec if isinstance(vec, int) else len(vec)
    if sz == 1:
        return 1
    return int(np.ceil(np.log2(sz)))


def sanitize_sparse_state_vector(
    vec: StateVector, *, copy=True
) -> sp.sparse.csr_matrix:
    """given a list of complex numbers, build a normalized state vector stored as a scipy CSR matrix"""

    vec = sp.sparse.csr_matrix(vec)
    if copy:
        vec = vec.copy()

    vec /= sp.linalg.norm(vec.data)  # normalize
    vec.sort_indices()  # order non-zero locations

    return vec


def f_cs(theta, i):
    if i == "0":
        return np.cos(theta / 2) ** 2
    elif i == "1":
        return np.sin(theta / 2) ** 2
    else:
        raise ValueError("smth wrong in input i of f_cs")


def generate_strings(s):
    # Find all positions of 'e' in the string
    positions = [i for i, char in enumerate(s) if char == "e"]
    num_e = len(positions)

    # Generate all combinations of '0' and '1' for the 'e' positions
    replacements = product("01", repeat=num_e)

    result = []
    for replacement in replacements:
        s_list = list(s)
        for i, char in zip(positions, replacement):
            s_list[i] = char
        result.append("".join(s_list))

    return result


def _branch_has_no_support(pattern, baseline_support, tol=1e-15):
    """
    True iff no nonzero leaf of the exact baseline circuit has a prefix
    compatible with 'pattern'.
    """
    k = len(pattern)
    for leaf, prob in baseline_support:
        if prob > tol and _pattern_matches(pattern, leaf[:k]):
            return False
    return True

    
def _pattern_matches(pattern, bit_string):
    return all(p == "e" or p == b for p, b in zip(pattern, bit_string))


def _build_baseline_support(baseline_ops, tol=1e-15):
    """
    Build the nonzero leaves of the baseline circuit together with their
    probabilities. For a d-sparse target state, this list has size O(d).
    """
    n_layers = len(baseline_ops)
    support = []

    def dfs(prefix, prob):
        depth = len(prefix)
        if depth == n_layers:
            if prob > tol:
                support.append((prefix, prob))
            return

        theta = _matching_value(prefix, baseline_ops[depth])[0]

        p0 = prob * f_cs(theta, "0")
        if p0 > tol:
            dfs(prefix + "0", p0)

        p1 = prob * f_cs(theta, "1")
        if p1 > tol:
            dfs(prefix + "1", p1)

    dfs("", 1.0)
    return support


def _matching_value(prefix, angles_phases_dict):
    """
    Return the most specific gate in angles_phases_dict matching the concrete prefix.
    Needed because the dictionaries can contain keys with 'e'.
    """
    best_value = None
    best_specificity = -1

    for pattern, value in angles_phases_dict.items():
        if len(pattern) != len(prefix):
            continue
        if _pattern_matches(pattern, prefix):
            specificity = sum(ch != "e" for ch in pattern)
            if specificity > best_specificity:
                best_specificity = specificity
                best_value = value

    return best_value if best_value is not None else (0.0, 0.0)


def _matching_value(prefix, layer_dict):
    """
    Return the most specific gate in layer_dict matching the concrete prefix.
    If nothing matches, return the trivial zero-angle gate.
    """
    best_value = None
    best_specificity = -1

    for pattern, value in layer_dict.items():
        if len(pattern) != len(prefix):
            continue
        if not _pattern_matches(pattern, prefix):
            continue

        specificity = sum(ch != "e" for ch in pattern)
        if specificity > best_specificity:
            best_specificity = specificity
            best_value = value

    return best_value if best_value is not None else (0.0, 0.0)


def _supported_prefixes_from_baseline_support(baseline_support, tol=1e-15):
    """
    Build the set of all prefixes that lie on nonzero baseline leaves.
    """
    prefixes = {""}
    for leaf, prob in baseline_support:
        if prob <= tol:
            continue
        for j in range(len(leaf) + 1):
            prefixes.add(leaf[:j])
    return prefixes


def _exact_overlap_from_prefix(prefix, n_layers, baseline_ops, active_ops, supported_prefixes, memo):
    """
    Exact overlap recursion restricted to the support trie of the baseline state.
    """
    if prefix in memo:
        return memo[prefix]

    # If the baseline has no support below this prefix, this branch contributes zero.
    if prefix not in supported_prefixes:
        memo[prefix] = 0.0
        return 0.0

    # Reached a full leaf of the preparation tree.
    if len(prefix) == n_layers:
        memo[prefix] = 1.0
        return 1.0

    theta_base = _matching_value(prefix, baseline_ops[len(prefix)])[0]
    theta_active = _matching_value(prefix, active_ops[len(prefix)])[0]

    c = np.cos(theta_base / 2.0) * np.cos(theta_active / 2.0)
    s = np.sin(theta_base / 2.0) * np.sin(theta_active / 2.0)

    value = (
        c * _exact_overlap_from_prefix(
            prefix + "0", n_layers, baseline_ops, active_ops, supported_prefixes, memo
        )
        + s * _exact_overlap_from_prefix(
            prefix + "1", n_layers, baseline_ops, active_ops, supported_prefixes, memo
        )
    )

    memo[prefix] = value
    return value


def exact_overlap_current_circuit(baseline_ops, active_ops, baseline_support=None):
    """
    Exact overlap <psi_baseline | psi_active> computed on the baseline support trie.
    """
    if baseline_support is None:
        baseline_support = _build_baseline_support(baseline_ops)

    supported_prefixes = _supported_prefixes_from_baseline_support(baseline_support)
    n_layers = len(baseline_ops)

    return _exact_overlap_from_prefix(
        "",
        n_layers,
        baseline_ops,
        active_ops,
        supported_prefixes,
        memo={},
    )