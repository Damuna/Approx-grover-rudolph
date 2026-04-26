import numpy as np

from .helping_functions import (
    neighbour_dict,
    _pattern_matches,
    _matching_value,
    _build_baseline_support,
)

__all__ = [
    "merging_formula",
    "ordering_geometric_series",
    "order_pairs_optimally",
]



def _build_prefix_tables(baseline_support, n_layers):
    """
    Build sparse prefix tables from the nonzero baseline leaves.

    prefix_probs[k][p] = total probability of all leaves whose first k bits equal p
    supported_prefixes[k] = tuple of supported concrete prefixes of length k

    Both are O(d n) to build.
    """
    prefix_probs = [dict() for _ in range(n_layers + 1)]
    prefix_probs[0][""] = 1.0

    for leaf, prob in baseline_support:
        for k in range(1, n_layers + 1):
            prefix = leaf[:k]
            prefix_probs[k][prefix] = prefix_probs[k].get(prefix, 0.0) + prob

    supported_prefixes = [tuple(tbl.keys()) for tbl in prefix_probs]
    return prefix_probs, supported_prefixes


def _active_key_for_baseline_key(baseline_key, active_layer):
    """
    Find the unique active key in the same layer that currently represents
    the original baseline key.
    """
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


def _probability_weight(pattern, baseline_support, prob_cache, prefix_probs=None):
    """
    P(pattern) from sparse baseline support.

    Scan the sparse baseline support in O(d).
    """
    if pattern in prob_cache:
        return prob_cache[pattern]

    if prefix_probs is not None and "e" not in pattern:
        total = prefix_probs[len(pattern)].get(pattern, 0.0)
        prob_cache[pattern] = total
        return total

    k = len(pattern)
    total = 0.0
    for leaf, prob in baseline_support:
        if _pattern_matches(pattern, leaf[:k]):
            total += prob

    prob_cache[pattern] = total
    return total


def _initialize_merge_state(gate_operations, use_rigorous_bound=False):
    baseline_ops = [dict(layer_dict) for layer_dict in gate_operations]
    baseline_support = _build_baseline_support(baseline_ops)
    prefix_probs, supported_prefixes = _build_prefix_tables(
        baseline_support, len(baseline_ops)
    )

    return {
        "baseline_ops": baseline_ops,
        "baseline_support": baseline_support,
        "prefix_probs": prefix_probs,
        "supported_prefixes": supported_prefixes,
        "sources": {},
        "losses": {},
        "prob_cache": {},
    }


def _initialize_merge_state_from_baseline(
    baseline_ops, active_ops, use_rigorous_bound=False
):
    """
    Initialize the merge state when the active circuit may already contain 'e'
    keys due to exact support-aware optimization.
    """
    if len(baseline_ops) != len(active_ops):
        raise ValueError(
            "baseline_ops and active_ops must have the same number of layers."
        )

    baseline_ops = [dict(layer) for layer in baseline_ops]
    active_ops = [dict(layer) for layer in active_ops]
    baseline_support = _build_baseline_support(baseline_ops)
    prefix_probs, supported_prefixes = _build_prefix_tables(
        baseline_support, len(baseline_ops)
    )

    merge_state = {
        "baseline_ops": baseline_ops,
        "baseline_support": baseline_support,
        "prefix_probs": prefix_probs,
        "supported_prefixes": supported_prefixes,
        "sources": {},
        "losses": {},
        "prob_cache": {},
    }

    # Assign every original concrete baseline key to the active key that represents it.
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

    # Compute the current loss of each active cluster relative to the baseline.
    for depth, active_layer in enumerate(active_ops):
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
    """
    Lazily initialize a cluster if it has never been merged before.
    The source map always stores disjoint concrete baseline patterns
    of the same length as the active key.
    """
    cid = _cluster_id(key)
    if cid not in merge_state["sources"]:
        theta = gate_operations[len(key)][key][0]
        merge_state["sources"][cid] = {key: theta}
        merge_state["losses"][cid] = 0.0

    return dict(merge_state["sources"][cid])


def _cluster_loss(source_map, new_theta, merge_state):
    """
    source_map must contain disjoint concrete baseline patterns.
    """
    loss = 0.0
    baseline_support = merge_state["baseline_support"]
    prob_cache = merge_state["prob_cache"]
    prefix_probs = merge_state["prefix_probs"]

    for pattern, theta_orig in source_map.items():
        p = _probability_weight(
            pattern,
            baseline_support,
            prob_cache,
            prefix_probs=prefix_probs,
        )
        loss += (1.0 - np.cos((theta_orig - new_theta) / 2.0)) * p

    return loss


def _optimal_cluster_angle(source_map, merge_state):
    """
    source_map must contain disjoint concrete baseline patterns.
    """
    x = 0.0
    y = 0.0
    baseline_support = merge_state["baseline_support"]
    prob_cache = merge_state["prob_cache"]
    prefix_probs = merge_state["prefix_probs"]

    for pattern, theta_orig in source_map.items():
        p = _probability_weight(
            pattern,
            baseline_support,
            prob_cache,
            prefix_probs=prefix_probs,
        )
        x += p * np.cos(theta_orig / 2.0)
        y += p * np.sin(theta_orig / 2.0)

    return 2.0 * np.arctan2(y, x)


def _find_absorbed_active_keys(new_key, active_layer):
    """
    Active keys whose entire region lies inside B(new_key).
    """
    return [key for key in active_layer if _pattern_matches(new_key, key)]


def _candidate_source_map(
    k1,
    k2,
    new_key,
    gate_operations,
    merge_state,
    regional_merges=False,
):
    """
    Build the candidate merged cluster for the region B(new_key).

    Returns:
        (source_map, old_loss, absorbed_keys, did_regional_merge)

    If regional_merges=False and the enlarged region would absorb extra
    active keys beyond the intended ones, return None to block the merge.
    """
    depth = len(new_key)
    active_layer = gate_operations[depth]

    absorbed_keys = _find_absorbed_active_keys(new_key, active_layer)

    required_keys = {k1} if k2 is None else {k1, k2}
    if not required_keys.issubset(absorbed_keys):
        raise ValueError(
            f"Candidate region {new_key} does not contain the required keys "
            f"{required_keys}. Found absorbed_keys={absorbed_keys}."
        )

    extra_keys = [key for key in absorbed_keys if key not in required_keys]
    did_regional_merge = len(extra_keys) > 0

    if did_regional_merge and not regional_merges:
        return None

    source_map = {}
    old_loss = 0.0

    for key in absorbed_keys:
        cid = _cluster_id(key)
        source_map.update(_get_cluster_sources(key, gate_operations, merge_state))
        old_loss += merge_state["losses"][cid]

    # Add supported concrete prefixes in the region that are not already covered.
    supported_prefixes = merge_state["supported_prefixes"][depth]
    for prefix in supported_prefixes:
        if prefix in source_map:
            continue
        if _pattern_matches(new_key, prefix):
            source_map[prefix] = 0.0

    return source_map, old_loss, absorbed_keys, did_regional_merge



def _g_amp(theta, bit):
    """
    Single-branch amplitude factor.
    """
    if bit == "0":
        return np.cos(theta / 2.0)
    return np.sin(theta / 2.0)


def _build_prefix_amplification_table(
    baseline_ops, active_ops, baseline_support, tol=1e-15
):
    """
    For each baseline leaf b and each depth k, compute the amplification
    R_b^{(<k)} induced by layers 0,...,k-1:

        R_b^{(<k)} = prod_{j<k} g(theta'_j, b_j) / g(theta_j, b_j)

    Returned as a dict keyed by (leaf, depth).
    """
    n_layers = len(baseline_ops)
    prefix_amp = {}

    for leaf, _ in baseline_support:
        amp_ratio = 1.0
        prefix_amp[(leaf, 0)] = 1.0

        for depth in range(n_layers):
            prefix = leaf[:depth]
            bit = leaf[depth]

            theta_base = _matching_value(prefix, baseline_ops[depth])[0]
            theta_act = _matching_value(prefix, active_ops[depth])[0]

            amp_old = _g_amp(theta_base, bit)
            amp_new = _g_amp(theta_act, bit)

            if abs(amp_old) <= tol:
                if abs(amp_new) <= tol:
                    step_ratio = 1.0
                else:
                    step_ratio = np.inf
            else:
                step_ratio = amp_new / amp_old

            amp_ratio *= step_ratio
            prefix_amp[(leaf, depth + 1)] = amp_ratio

    return prefix_amp


def _cluster_prefix_amplification(source_map, depth, baseline_support, prefix_amp):
    """
    R_C = max_{x in C, b in B(x)} R_b^{(<depth)}.

    Since source_map stores concrete baseline patterns of length 'depth',
    this is just the max over leaves whose first 'depth' bits equal one
    of those concrete patterns.
    """
    max_R = 1.0
    found_match = False

    for leaf, _ in baseline_support:
        prefix = leaf[:depth]
        if prefix in source_map:
            max_R = max(max_R, prefix_amp[(leaf, depth)])
            found_match = True

    return max_R if found_match else 1.0


def _compute_rigorous_bound_from_active_circuit(merge_state, active_ops):
    """
    Lower Bound:

        <psi'|psi> >= 1 - sum_C R_C L_C,

    where for a cluster C in layer k,
    R_C uses only ancestor-layer amplification R_b^{(<k)}.
    """
    baseline_ops = merge_state["baseline_ops"]
    baseline_support = merge_state["baseline_support"]

    prefix_amp = _build_prefix_amplification_table(
        baseline_ops=baseline_ops,
        active_ops=active_ops,
        baseline_support=baseline_support,
    )

    total_loss_bound = 0.0

    for cid, loss_C in merge_state["losses"].items():
        if loss_C <= 0.0:
            continue

        depth, _ = cid
        source_map = merge_state["sources"][cid]
        R_C = _cluster_prefix_amplification(
            source_map=source_map,
            depth=depth,
            baseline_support=baseline_support,
            prefix_amp=prefix_amp,
        )
        total_loss_bound += R_C * loss_C

    lower_bound = 1.0 - total_loss_bound
    lower_bound = max(0.0, lower_bound)
    return lower_bound


def merging_formula(
    k1,
    k2,
    new_key,
    gate_operations,
    merge_state,
    overlap,
    new_phase=0.0,
    regional_merges=False,
):
    """
    Return the global overlap estimate after replacing the old cluster(s)
    in B(new_key) by one merged cluster.

    Returns:
        None if the candidate is blocked.
        Otherwise:
        (overlap_estimate, new_value, source_map, new_loss, absorbed_keys, did_regional_merge)
    """
    candidate = _candidate_source_map(
        k1=k1,
        k2=k2,
        new_key=new_key,
        gate_operations=gate_operations,
        merge_state=merge_state,
        regional_merges=regional_merges,
    )

    if candidate is None:
        return None

    source_map, old_loss, absorbed_keys, did_regional_merge = candidate

    theta_new = _optimal_cluster_angle(source_map, merge_state)
    new_loss = _cluster_loss(source_map, theta_new, merge_state)
    overlap_estimate = overlap + old_loss - new_loss
    new_value = (theta_new, new_phase)

    return (
        overlap_estimate,
        new_value,
        source_map,
        new_loss,
        absorbed_keys,
        did_regional_merge,
    )


def ordering_geometric_series(
    gate_operations,
    min_overlap,
    m_steps,
    error=np.pi / 2,
    baseline_gate_operations=None,
    use_rigorous_bound=False,
    regional_merges=False,
):
    """
    Perform approximate merging with geometric series of overlap thresholds.

    Returns:
        overlap_estimate, rigorous_bound
    """
    if not (0.0 <= min_overlap <= 1.0):
        raise ValueError("min_overlap must lie in [0, 1].")
    if m_steps <= 0:
        raise ValueError("m_steps must be a positive integer.")

    flag = True
    overlap = 1.0

    if baseline_gate_operations is None:
        merge_state = _initialize_merge_state(gate_operations, use_rigorous_bound)
    else:
        merge_state = _initialize_merge_state_from_baseline(
            baseline_gate_operations,
            gate_operations,
            use_rigorous_bound,
        )

    step = (1.0 - min_overlap) / m_steps
    min_overlap_step = 1.0

    for _ in range(m_steps):
        min_overlap_step -= step
        overlap, flag = order_pairs_optimally(
            gate_operations,
            min_overlap_step,
            error,
            overlap=overlap,
            merge_state=merge_state,
            use_rigorous_bound=False,
            regional_merges=regional_merges,
        )

    while flag:
        overlap, flag = order_pairs_optimally(
            gate_operations,
            min_overlap,
            error,
            overlap=overlap,
            merge_state=merge_state,
            use_rigorous_bound=False,
            regional_merges=regional_merges,
        )

    if use_rigorous_bound:
        rigorous_bound = _compute_rigorous_bound_from_active_circuit(
            merge_state=merge_state,
            active_ops=gate_operations,
        )
        return overlap, rigorous_bound

    return overlap, None


def order_pairs_optimally(
    gate_operations,
    min_overlap,
    error,
    overlap=1.0,
    merge_state=None,
    use_rigorous_bound=False,
    regional_merges=False,
):
    if merge_state is None:
        merge_state = _initialize_merge_state(gate_operations, use_rigorous_bound)

    pairs = []

    for angles_phases_dict in gate_operations:
        for k1, v1 in list(angles_phases_dict.items()):
            neighbours = neighbour_dict(k1)

            # ----- control-removal candidate -----
            if k1 != "e" * len(k1) and abs(v1[0]) <= 2 * error:
                for position in range(len(k1)):
                    if k1[position] == "e":
                        continue

                    new_key = k1[:position] + "e" + k1[position + 1 :]

                    result = merging_formula(
                        k1=k1,
                        k2=None,
                        new_key=new_key,
                        gate_operations=gate_operations,
                        merge_state=merge_state,
                        overlap=overlap,
                        new_phase=v1[1],
                        regional_merges=regional_merges,
                    )

                    if result is None:
                        continue

                    (
                        overlap_estimate,
                        new_value,
                        _source_map,
                        _new_loss,
                        absorbed_keys,
                        did_regional_merge,
                    ) = result

                    if overlap_estimate >= min_overlap:
                        pairs.append(
                            (
                                k1,
                                None,
                                new_key,
                                new_value,
                                overlap_estimate,
                                absorbed_keys,
                                did_regional_merge,
                            )
                        )

            # ----- merge with a real neighbour -----
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

                result = merging_formula(
                    k1=k1,
                    k2=k2,
                    new_key=new_key,
                    gate_operations=gate_operations,
                    merge_state=merge_state,
                    overlap=overlap,
                    new_phase=(v1[1] + v2[1]) / 2.0,
                    regional_merges=regional_merges,
                )

                if result is None:
                    continue

                (
                    overlap_estimate,
                    new_value,
                    _source_map,
                    _new_loss,
                    absorbed_keys,
                    did_regional_merge,
                ) = result

                if overlap_estimate >= min_overlap:
                    pairs.append(
                        (
                            k1,
                            k2,
                            new_key,
                            new_value,
                            overlap_estimate,
                            absorbed_keys,
                            did_regional_merge,
                        )
                    )

    if not pairs:
        return overlap, False

    pairs = sorted(pairs, key=lambda x: x[4], reverse=True)
    merged_any = False

    for k1, k2, new_key, _new_value, _old_estimate, _absorbed_keys, _did_regional_merge in pairs:
        angles_phases_dict = gate_operations[len(k1)]

        if k1 not in angles_phases_dict:
            continue
        if (k2 is not None) and (k2 not in angles_phases_dict):
            continue

        if k2 is None:
            current_phase = angles_phases_dict[k1][1]
        else:
            current_phase = (
                angles_phases_dict[k1][1] + angles_phases_dict[k2][1]
            ) / 2.0

        result = merging_formula(
            k1=k1,
            k2=k2,
            new_key=new_key,
            gate_operations=gate_operations,
            merge_state=merge_state,
            overlap=overlap,
            new_phase=current_phase,
            regional_merges=regional_merges,
        )

        if result is None:
            continue

        (
            overlap_estimate,
            new_value,
            source_map,
            new_loss,
            absorbed_keys,
            did_regional_merge,
        ) = result

        if overlap_estimate < min_overlap:
            continue

        for key in absorbed_keys:
            angles_phases_dict.pop(key, None)
            merge_state["sources"].pop(_cluster_id(key), None)
            merge_state["losses"].pop(_cluster_id(key), None)

        angles_phases_dict[new_key] = new_value
        merge_state["sources"][_cluster_id(new_key)] = source_map
        merge_state["losses"][_cluster_id(new_key)] = new_loss

        overlap = overlap_estimate
        merged_any = True

    return overlap, merged_any