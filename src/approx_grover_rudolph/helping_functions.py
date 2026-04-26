import numpy as np
import scipy as sp

from typing import Union


__all__ = [
    "RotationGate",
    "ControlledRotationGateMap",
    "StateVector",
    "ZERO",
    "neighbour_dict",
    "generate_sparse_unit_vector",
    "sanitize_sparse_state_vector",
    "f_cs",
]

RotationGate = tuple[float, float]
Controls = str
ControlledRotationGateMap = dict[Controls, RotationGate]
StateVector = Union[np.ndarray, sp.sparse.spmatrix, list[float]]
ZERO = 1e-8


def neighbour_dict(controls: Controls) -> dict[Controls, int]:
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


def sanitize_sparse_state_vector(
    vec: StateVector, *, copy=True
) -> sp.sparse.csr_matrix:
    vec = sp.sparse.csr_matrix(vec)
    if copy:
        vec = vec.copy()

    vec /= sp.linalg.norm(vec.data)
    vec.sort_indices()
    return vec


def f_cs(theta, i):
    if i == "0":
        return np.cos(theta / 2) ** 2
    if i == "1":
        return np.sin(theta / 2) ** 2
    raise ValueError("smth wrong in input i of f_cs")


def _branch_has_no_support(pattern, baseline_support, tol=1e-15):
    k = len(pattern)
    for leaf, prob in baseline_support:
        if prob > tol and _pattern_matches(pattern, leaf[:k]):
            return False
    return True


def _pattern_matches(pattern, bit_string):
    return all(p == "e" or p == b for p, b in zip(pattern, bit_string))


def _matching_value(prefix, layer_dict):
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


def _build_baseline_support(baseline_ops, tol=1e-15):
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
