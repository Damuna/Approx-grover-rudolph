import numpy as np

from .helping_functions import (
    ControlledRotationGateMap,
    neighbour_dict,
    generate_strings,
    f_cs,
    replace_first_non_e,
)

__all__=[
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


def _probability_weight(pattern, baseline_ops, prob_cache):
    """
    P(pattern) computed from the fixed baseline circuit, not from the updated one.
    """
    if pattern in prob_cache:
        return prob_cache[pattern]

    total = 0.0
    for bit_string in generate_strings(pattern):
        f_product = 1.0
        for i in range(len(bit_string)):
            prefix = bit_string[:i]
            theta = _matching_value(prefix, baseline_ops[len(prefix)])[0]
            f_product *= f_cs(theta, bit_string[i])
        total += f_product

    prob_cache[pattern] = total
    return total


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


def _probability_weight(pattern, baseline_support, prob_cache):
    """
    P(pattern) computed from the sparse baseline support, not by expanding
    all bit strings compatible with the pattern.
    """
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


def _cluster_id(key):
    return (len(key), key)


def _get_cluster_sources(key, gate_operations, merge_state):
    """
    Lazily initialize a cluster if it has never been merged before.
    """
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
    return key[:position] + flipped + key[position + 1:]


def _support_is_empty(pattern, angles_phases_dict, excluded_keys=()):
    """
    True iff no active gate overlaps the support of 'pattern',
    except for keys explicitly excluded.
    """
    excluded_keys = set(excluded_keys)
    for other_key in angles_phases_dict:
        if other_key in excluded_keys:
            continue
        if _patterns_overlap(pattern, other_key):
            return False
    return True


def _candidate_source_map(k1, k2, gate_operations, merge_state, zero_partner=None):
    """
    Build the baseline source map of the candidate merged cluster and
    return the total old loss that will be replaced by the new cluster loss.
    """
    source_map = _get_cluster_sources(k1, gate_operations, merge_state)
    old_loss = merge_state["losses"][_cluster_id(k1)]

    if k2 is not None:
        source_map.update(_get_cluster_sources(k2, gate_operations, merge_state))
        old_loss += merge_state["losses"][_cluster_id(k2)]
    elif zero_partner is not None:
        source_map[zero_partner] = 0.0

    return source_map, old_loss


def merging_formula(k1, k2, new_key, gate_operations, merge_state, overlap,
                    zero_partner=None, new_phase=0.0):
    """
    Return the global overlap estimate after replacing the old cluster(s)
    by the new merged cluster.

    This no longer uses
        overlap_new = overlap - Delta_i
    blindly.

    Instead, it does
        overlap_new = overlap + old_cluster_loss - new_cluster_loss,
    which is what fixes repeated non-disjoint merges.
    """
    source_map, old_loss = _candidate_source_map(
        k1, k2, gate_operations, merge_state, zero_partner=zero_partner
    )

    theta_new = _optimal_cluster_angle(source_map, merge_state)
    new_loss = _cluster_loss(source_map, theta_new, merge_state)
    overlap_estimate = overlap + old_loss - new_loss
    new_value = (theta_new, new_phase)

    return overlap_estimate, new_value, source_map, new_loss


def ordering_geometric_series(gate_operations, min_overlap, m_steps, error=np.pi / 2):
    flag = True
    overlap = 1.0
    merge_state = _initialize_merge_state(gate_operations)
    total_merges = 0
    zero_merges = 0

    step = (1.0 - min_overlap) / m_steps
    min_overlap_step = 1.0

    for _ in range(m_steps):
        min_overlap_step -= step
        overlap, flag, tm, zm = order_pairs_optimally(
            gate_operations,
            min_overlap_step,
            error,
            overlap=overlap,
            merge_state=merge_state,
        )
        total_merges += tm
        zero_merges += zm

    while flag:
        overlap, flag, tm, zm = order_pairs_optimally(
            gate_operations,
            min_overlap,
            error,
            overlap=overlap,
            merge_state=merge_state,
        )
        total_merges += tm
        zero_merges += zm

    return overlap, total_merges, zero_merges


def order_pairs_optimally(gate_operations, min_overlap, error, overlap=1.0, merge_state=None):
    if merge_state is None:
        merge_state = _initialize_merge_state(gate_operations)

    pairs = []
    total_merges = 0
    zero_merges = 0

    for angles_phases_dict in gate_operations:
        # list(...) avoids "dictionary keys changed during iteration"
        for k1, v1 in list(angles_phases_dict.items()):
            neighbours = neighbour_dict(k1)

            # ----- merge with an imaginary zero -----
            if k1 != "e" * len(k1) and abs(v1[0]) <= 2 * error:
                new_key, merging_position = replace_first_non_e(k1)
                zero_partner = _zero_partner_pattern(k1, merging_position)

                # Only treat it as an imaginary zero if that support is really empty
                if _support_is_empty(zero_partner, angles_phases_dict, excluded_keys=(k1,)):
                    overlap_estimate, new_value, _, _ = merging_formula(
                        k1=k1,
                        k2=None,
                        new_key=new_key,
                        gate_operations=gate_operations,
                        merge_state=merge_state,
                        overlap=overlap,
                        zero_partner=zero_partner,
                        new_phase=v1[1] / 2.0,
                    )

                    if overlap_estimate >= min_overlap:
                        pairs.append(
                            (k1, None, new_key, new_value, overlap_estimate, zero_partner)
                        )

            # ----- merge with a real neighbour -----
            for k2, position in neighbours.items():
                if k2 not in angles_phases_dict:
                    continue

                # Avoid adding the same pair twice
                if k2 <= k1:
                    continue

                v2 = angles_phases_dict[k2]
                eps_theta = abs(v1[0] - v2[0]) / 2.0
                eps_phi = abs(v1[1] - v2[1]) / 2.0

                if (eps_theta > error) or (eps_phi > error):
                    continue

                new_key = k1[:position] + "e" + k1[position + 1:]
                overlap_estimate, new_value, _, _ = merging_formula(
                    k1=k1,
                    k2=k2,
                    new_key=new_key,
                    gate_operations=gate_operations,
                    merge_state=merge_state,
                    overlap=overlap,
                    zero_partner=None,
                    new_phase=(v1[1] + v2[1]) / 2.0,
                )

                if overlap_estimate >= min_overlap:
                    pairs.append(
                        (k1, k2, new_key, new_value, overlap_estimate, None)
                    )

    if pairs == []:
        return overlap, False, 0, 0

    pairs = sorted(pairs, key=lambda x: x[4], reverse=True)
    merged_any = False

    for k1, k2, new_key, _, _, zero_partner in pairs:
        angles_phases_dict = gate_operations[len(k1)]

        if k1 not in angles_phases_dict:
            continue
        if (k2 is not None) and (k2 not in angles_phases_dict):
            continue

        # Recompute using the current state, because previous accepted merges
        # may have changed the source cluster of k1 or k2.
        if k2 is None:
            current_phase = angles_phases_dict[k1][1] / 2.0
        else:
            current_phase = (angles_phases_dict[k1][1] + angles_phases_dict[k2][1]) / 2.0

        overlap_estimate, new_value, source_map, new_loss = merging_formula(
            k1=k1,
            k2=k2,
            new_key=new_key,
            gate_operations=gate_operations,
            merge_state=merge_state,
            overlap=overlap,
            zero_partner=zero_partner,
            new_phase=current_phase,
        )

        if overlap_estimate < min_overlap:
            continue

        # Remove old active gates
        angles_phases_dict.pop(k1)
        merge_state["sources"].pop(_cluster_id(k1), None)
        merge_state["losses"].pop(_cluster_id(k1), None)

        if k2 is not None:
            angles_phases_dict.pop(k2)
            merge_state["sources"].pop(_cluster_id(k2), None)
            merge_state["losses"].pop(_cluster_id(k2), None)

        # Add the new active gate
        angles_phases_dict[new_key] = new_value
        merge_state["sources"][_cluster_id(new_key)] = source_map
        merge_state["losses"][_cluster_id(new_key)] = new_loss

        overlap = overlap_estimate
        total_merges += 1
        merged_any = True

        if k2 is None:
            zero_merges += 1

    return overlap, merged_any, total_merges, zero_merges


def optimize_dict(
    gate_operations: ControlledRotationGateMap,
    error,
) -> ControlledRotationGateMap:
    """
    Optimize the dictionary by merging some gates in one:
    if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
    >> {'11':[3.14,0] ; '10':[3.14,0]} becomes {'1e':[3.14,0]} where 'e' means no control (identity)

    >>> assert optimize_dict({"11": (3.14, 0), "10": (3.14, 0)}) == {"1e": (3.14, 0)}

    Args:
        gate_operations: collection of controlled gates to be applied
    Returns:
        optimized collection of controlled gates
    """
    merged = True

    while merged:
        merged = run_one_merge_step(gate_operations, error)

    return gate_operations


def run_one_merge_step(gate_operations: ControlledRotationGateMap, error) -> bool:
    """
    Run a single merging step, modifying the input dictionary.

    Args:
        gate_operations: collection of controlled gates to be applied
    Returns:
        True if some merge happened
    """

    if len(gate_operations) <= 1:
        return False

    for k1, v1 in gate_operations.items():
        neighbours = neighbour_dict(k1)

        for k2, position in neighbours.items():
            if k2 not in gate_operations:
                continue

            v2 = gate_operations[k2]

            # Consider only different items with same angle and phase up to sone error
            if (abs(v1[0] - v2[0]) / 2 > error) or (abs(v1[1] - v2[1]) / 2 > error):
                continue

            # Replace the different char with 'e' and remove the old items
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