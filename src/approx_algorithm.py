import numpy as np

from .helping_functions import (
    ControlledRotationGateMap,
    neighbour_dict,
    f_cs,
    replace_first_non_e,
    exact_overlap_current_circuit,
)
from .gate_count import _single_gate_cnot_cost

__all__ = [
    "merging_formula",
    "ordering_geometric_series",
    "order_pairs_optimally",
    "optimize_dict",
    "run_one_merge_step",
    "optimize_full_dict",
]


def _pattern_matches(pattern, bit_string):
    return all(p == "e" or p == b for p, b in zip(pattern, bit_string))


def _patterns_overlap(p1, p2):
    return all(a == "e" or b == "e" or a == b for a, b in zip(p1, p2))


def _matching_value(prefix, angles_phases_dict):
    best_value = None
    best_specificity = -1

    for pattern, value in angles_phases_dict.items():
        if len(pattern) != len(prefix):
            continue
        if not _pattern_matches(pattern, prefix):
            continue

        specificity = sum(ch != "e" for ch in pattern)
        if specificity > best_specificity:
            best_specificity = specificity
            best_value = value

    return best_value if best_value is not None else (0.0, 0.0)


def _active_key_for_baseline_key(baseline_key, active_layer):
    best_key = None
    best_specificity = -1

    for active_key in active_layer:
        if len(active_key) != len(baseline_key):
            continue
        if not _pattern_matches(active_key, baseline_key):
            continue

        specificity = sum(ch != "e" for ch in active_key)
        if specificity > best_specificity:
            best_specificity = specificity
            best_key = active_key
        elif specificity == best_specificity and active_key != best_key:
            raise ValueError(
                f"Ambiguous assignment of baseline key {baseline_key} "
                f"to active keys {best_key} and {active_key}."
            )

    if best_key is None:
        raise ValueError(
            f"No active key found that represents baseline key {baseline_key}."
        )

    return best_key


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


def _probability_weight(pattern, baseline_support, prob_cache):
    if pattern in prob_cache:
        return prob_cache[pattern]

    k = len(pattern)
    total = 0.0
    for leaf, prob in baseline_support:
        if _pattern_matches(pattern, leaf[:k]):
            total += prob

    prob_cache[pattern] = total
    return total


def _initialize_merge_state(gate_operations):
    baseline_ops = [dict(layer_dict) for layer_dict in gate_operations]
    baseline_support = _build_baseline_support(baseline_ops)
    return {
        "baseline_ops": baseline_ops,
        "baseline_support": baseline_support,
        "sources": {},
        "losses": {},
        "prob_cache": {},
    }



def _initialize_merge_state_from_baseline(baseline_ops, active_ops):
    if len(baseline_ops) != len(active_ops):
        raise ValueError("baseline_ops and active_ops must have the same number of layers.")

    baseline_ops = [dict(layer) for layer in baseline_ops]
    active_ops = [dict(layer) for layer in active_ops]
    baseline_support = _build_baseline_support(baseline_ops)

    merge_state = {
        "baseline_ops": baseline_ops,
        "baseline_support": baseline_support,
        "sources": {},
        "losses": {},
        "prob_cache": {},
    }

    for depth, baseline_layer in enumerate(baseline_ops):
        active_layer = active_ops[depth]
        sources_by_active = {active_key: {} for active_key in active_layer}

        for base_key, (theta_base, _) in baseline_layer.items():
            active_key = _active_key_for_baseline_key(base_key, active_layer)
            sources_by_active[active_key][base_key] = theta_base

        for active_key in active_layer:
            cid = _cluster_id(active_key)
            source_map = sources_by_active[active_key]
            if not source_map:
                raise ValueError(
                    f"Active key {active_key} in layer {depth} has no baseline source keys."
                )
            merge_state["sources"][cid] = source_map

    for active_layer in active_ops:
        for active_key, (theta_active, _) in active_layer.items():
            cid = _cluster_id(active_key)
            source_map = merge_state["sources"][cid]
            merge_state["losses"][cid] = _cluster_loss(
                source_map, theta_active, merge_state
            )

    return merge_state



def _cluster_id(key):
    return (len(key), key)



def _get_cluster_sources(key, gate_operations, merge_state):
    cid = _cluster_id(key)
    if cid not in merge_state["sources"]:
        theta = gate_operations[len(key)][key][0]
        merge_state["sources"][cid] = {key: theta}
        merge_state["losses"][cid] = 0.0

    return dict(merge_state["sources"][cid])



def _cluster_loss(source_map, new_theta, merge_state):
    loss = 0.0
    baseline_support = merge_state["baseline_support"]
    prob_cache = merge_state["prob_cache"]

    for pattern, theta_orig in source_map.items():
        p = _probability_weight(pattern, baseline_support, prob_cache)
        loss += (1.0 - np.cos((theta_orig - new_theta) / 2.0)) * p

    return loss



def _optimal_cluster_angle(source_map, merge_state):
    x = 0.0
    y = 0.0
    baseline_support = merge_state["baseline_support"]
    prob_cache = merge_state["prob_cache"]

    for pattern, theta_orig in source_map.items():
        p = _probability_weight(pattern, baseline_support, prob_cache)
        x += p * np.cos(theta_orig / 2.0)
        y += p * np.sin(theta_orig / 2.0)

    return 2.0 * np.arctan2(y, x)



def _zero_partner_pattern(key, position):
    flipped = "1" if key[position] == "0" else "0"
    return key[:position] + flipped + key[position + 1 :]



def _support_is_empty(pattern, angles_phases_dict, excluded_keys=()):
    excluded_keys = set(excluded_keys)
    for other_key in angles_phases_dict:
        if other_key in excluded_keys:
            continue
        if _patterns_overlap(pattern, other_key):
            return False
    return True



def _candidate_source_map(k1, k2, gate_operations, merge_state, zero_partner=None):
    source_map = _get_cluster_sources(k1, gate_operations, merge_state)
    old_loss = merge_state["losses"][_cluster_id(k1)]

    if k2 is not None:
        source_map.update(_get_cluster_sources(k2, gate_operations, merge_state))
        old_loss += merge_state["losses"][_cluster_id(k2)]
    elif zero_partner is not None:
        source_map[zero_partner] = 0.0

    return source_map, old_loss



def _single_layer_gain(candidate):
    k1 = candidate["k1"]
    k2 = candidate["k2"]
    new_key = candidate["new_key"]

    before = _single_gate_cnot_cost(k1)
    if k2 is not None:
        before += _single_gate_cnot_cost(k2)
    after = _single_gate_cnot_cost(new_key)
    return before - after



def _trial_circuit_with_candidate(gate_operations, candidate, new_value):
    trial_gate_operations = [dict(layer) for layer in gate_operations]
    layer_idx = len(candidate["k1"])
    trial_layer = trial_gate_operations[layer_idx]

    trial_layer.pop(candidate["k1"], None)
    if candidate["k2"] is not None:
        trial_layer.pop(candidate["k2"], None)
    trial_layer[candidate["new_key"]] = new_value

    return trial_gate_operations



def _is_exact_candidate(candidate, tol=1e-12):
    return candidate["exact_loss"] <= tol



def _candidate_sort_key(candidate):
    eps = 1e-15
    exact_flag = 1 if _is_exact_candidate(candidate) else 0
    efficiency = candidate["single_gain"] / max(candidate["exact_loss"], eps)
    return (
        exact_flag,
        efficiency,
        candidate["single_gain"],
        candidate["exact_overlap"],
    )



def merging_formula(
    k1,
    k2,
    new_key,
    gate_operations,
    merge_state,
    overlap,
    zero_partner=None,
    new_phase=0.0,
):
    source_map, old_loss = _candidate_source_map(
        k1, k2, gate_operations, merge_state, zero_partner=zero_partner
    )

    theta_new = _optimal_cluster_angle(source_map, merge_state)
    new_loss = _cluster_loss(source_map, theta_new, merge_state)
    overlap_estimate = overlap + old_loss - new_loss
    new_value = (theta_new, new_phase)

    return overlap_estimate, new_value, source_map, new_loss



def _build_candidates(gate_operations, min_overlap, error, overlap, merge_state):
    candidates = []

    for angles_phases_dict in gate_operations:
        for k1, v1 in list(angles_phases_dict.items()):
            neighbours = neighbour_dict(k1)

            if k1 != "e" * len(k1) and abs(v1[0]) <= 2 * error:
                new_key, merging_position = replace_first_non_e(k1)
                zero_partner = _zero_partner_pattern(k1, merging_position)

                if _support_is_empty(zero_partner, angles_phases_dict, excluded_keys=(k1,)):
                    overlap_estimate, new_value, source_map, new_loss = merging_formula(
                        k1=k1,
                        k2=None,
                        new_key=new_key,
                        gate_operations=gate_operations,
                        merge_state=merge_state,
                        overlap=overlap,
                        zero_partner=zero_partner,
                        new_phase=v1[1],
                    )
                    candidate = {
                        "k1": k1,
                        "k2": None,
                        "new_key": new_key,
                        "new_value": new_value,
                        "overlap_estimate": overlap_estimate,
                        "zero_partner": zero_partner,
                        "source_map": source_map,
                        "new_loss": new_loss,
                    }
                    candidate["single_gain"] = _single_layer_gain(candidate)
                    if candidate["single_gain"] > 0:
                        trial_gate_operations = _trial_circuit_with_candidate(
                            gate_operations, candidate, new_value
                        )
                        exact_overlap = exact_overlap_current_circuit(
                            merge_state["baseline_ops"],
                            trial_gate_operations,
                            baseline_support=merge_state["baseline_support"],
                        )
                        if exact_overlap >= min_overlap:
                            candidate["exact_overlap"] = exact_overlap
                            candidate["exact_loss"] = max(0.0, overlap - exact_overlap)
                            candidates.append(candidate)

            for k2, position in neighbours.items():
                if k2 not in angles_phases_dict:
                    continue
                if k2 <= k1:
                    continue

                v2 = angles_phases_dict[k2]
                eps_theta = abs(v1[0] - v2[0]) / 2.0
                eps_phi = abs(v1[1] - v2[1]) / 2.0
                if (eps_theta > error) or (eps_phi > error):
                    continue

                new_key = k1[:position] + "e" + k1[position + 1 :]
                overlap_estimate, new_value, source_map, new_loss = merging_formula(
                    k1=k1,
                    k2=k2,
                    new_key=new_key,
                    gate_operations=gate_operations,
                    merge_state=merge_state,
                    overlap=overlap,
                    zero_partner=None,
                    new_phase=(v1[1] + v2[1]) / 2.0,
                )
                candidate = {
                    "k1": k1,
                    "k2": k2,
                    "new_key": new_key,
                    "new_value": new_value,
                    "overlap_estimate": overlap_estimate,
                    "zero_partner": None,
                    "source_map": source_map,
                    "new_loss": new_loss,
                }
                candidate["single_gain"] = _single_layer_gain(candidate)
                if candidate["single_gain"] <= 0:
                    continue

                trial_gate_operations = _trial_circuit_with_candidate(
                    gate_operations, candidate, new_value
                )
                exact_overlap = exact_overlap_current_circuit(
                    merge_state["baseline_ops"],
                    trial_gate_operations,
                    baseline_support=merge_state["baseline_support"],
                )
                if exact_overlap >= min_overlap:
                    candidate["exact_overlap"] = exact_overlap
                    candidate["exact_loss"] = max(0.0, overlap - exact_overlap)
                    candidates.append(candidate)

    return candidates



def ordering_geometric_series(
    gate_operations,
    min_overlap,
    m_steps,
    error=np.pi / 2,
    baseline_gate_operations=None,
):
    if not (0.0 <= min_overlap <= 1.0):
        raise ValueError("min_overlap must lie in [0, 1].")
    if m_steps <= 0:
        raise ValueError("m_steps must be a positive integer.")

    if baseline_gate_operations is None:
        merge_state = _initialize_merge_state(gate_operations)
    else:
        merge_state = _initialize_merge_state_from_baseline(
            baseline_gate_operations,
            gate_operations,
        )

    overlap = exact_overlap_current_circuit(
        merge_state["baseline_ops"],
        gate_operations,
        baseline_support=merge_state["baseline_support"],
    )

    step = (1.0 - min_overlap) / m_steps
    current_threshold = 1.0

    for _ in range(m_steps):
        current_threshold -= step
        while True:
            overlap, changed = order_pairs_optimally(
                gate_operations,
                current_threshold,
                error,
                overlap=overlap,
                merge_state=merge_state,
            )
            if not changed:
                break

    return overlap



def order_pairs_optimally(
    gate_operations,
    min_overlap,
    error,
    overlap=1.0,
    merge_state=None,
):
    if merge_state is None:
        merge_state = _initialize_merge_state(gate_operations)

    candidates = _build_candidates(
        gate_operations,
        min_overlap,
        error,
        overlap,
        merge_state,
    )
    if not candidates:
        return overlap, False

    candidates.sort(key=_candidate_sort_key, reverse=True)
    best = candidates[0]

    k1 = best["k1"]
    k2 = best["k2"]
    new_key = best["new_key"]
    angles_phases_dict = gate_operations[len(k1)]

    if k1 not in angles_phases_dict:
        return overlap, False
    if (k2 is not None) and (k2 not in angles_phases_dict):
        return overlap, False

    angles_phases_dict.pop(k1)
    merge_state["sources"].pop(_cluster_id(k1), None)
    merge_state["losses"].pop(_cluster_id(k1), None)

    if k2 is not None:
        angles_phases_dict.pop(k2)
        merge_state["sources"].pop(_cluster_id(k2), None)
        merge_state["losses"].pop(_cluster_id(k2), None)

    angles_phases_dict[new_key] = best["new_value"]
    merge_state["sources"][_cluster_id(new_key)] = best["source_map"]
    merge_state["losses"][_cluster_id(new_key)] = best["new_loss"]

    return best["exact_overlap"], True



def optimize_dict(
    gate_operations: ControlledRotationGateMap,
    error,
) -> ControlledRotationGateMap:
    merged = True
    while merged:
        merged = run_one_merge_step(gate_operations, error)
    return gate_operations



def run_one_merge_step(gate_operations: ControlledRotationGateMap, error) -> bool:
    if len(gate_operations) <= 1:
        return False

    for k1, v1 in list(gate_operations.items()):
        neighbours = neighbour_dict(k1)

        for k2, position in neighbours.items():
            if k2 not in gate_operations:
                continue

            v2 = gate_operations[k2]
            if (abs(v1[0] - v2[0]) / 2 > error) or (abs(v1[1] - v2[1]) / 2 > error):
                continue

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
