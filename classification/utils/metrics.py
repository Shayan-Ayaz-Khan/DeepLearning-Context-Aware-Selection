import math
from collections import OrderedDict

import numpy as np


def _as_float_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def compute_wilcoxon(cas_per_batch_acc, eatta_per_batch_acc):
    """
    Wilcoxon signed-rank test comparing per-batch accuracy sequences.
    Returns: statistic, p_value, is_significant (p < 0.05)
    """
    cas = np.asarray(cas_per_batch_acc, dtype=float)
    eatta = np.asarray(eatta_per_batch_acc, dtype=float)
    n = min(cas.size, eatta.size)
    if n == 0:
        return None, None, False

    diff = cas[:n] - eatta[:n]
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0, 1.0, False

    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(diff)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        abs_diff = np.abs(diff)
        order = np.argsort(abs_diff)
        ranks = np.empty_like(abs_diff, dtype=float)
        ranks[order] = np.arange(1, diff.size + 1, dtype=float)
        w_plus = ranks[diff > 0].sum()
        w_minus = ranks[diff < 0].sum()
        statistic = float(min(w_plus, w_minus))
        mean_w = diff.size * (diff.size + 1) / 4.0
        var_w = diff.size * (diff.size + 1) * (2 * diff.size + 1) / 24.0
        if var_w <= 0:
            p_value = 1.0
        else:
            z = (statistic - mean_w) / math.sqrt(var_w)
            p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))

    return statistic, p_value, bool(p_value < 0.05)


def compute_cohens_d(cas_per_batch_acc, eatta_per_batch_acc):
    """
    Cohen's d effect size between CAS and EATTA per-batch accuracy distributions.
    Returns: cohens_d (float)
    """
    cas = np.asarray(cas_per_batch_acc, dtype=float)
    eatta = np.asarray(eatta_per_batch_acc, dtype=float)
    n = min(cas.size, eatta.size)
    if n == 0:
        return None

    diff = cas[:n] - eatta[:n]
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return 0.0

    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return float(diff.mean() / std)


def compute_snr(novelty_series, corruption_boundaries):
    """
    Signal-to-noise ratio of BN anchor.
    Signal = mean Nt at transition batches (first batch of each new block)
    Noise = mean Nt during stable periods (all other batches)
    Returns: snr_ratio, mean_transition, mean_stable
    """
    novelty = np.asarray(novelty_series, dtype=float)
    novelty = np.where(np.isfinite(novelty), novelty, np.nan)
    if novelty.size == 0:
        return None, None, None

    boundaries = sorted({int(b) for b in corruption_boundaries if 0 <= int(b) < novelty.size})
    transition_idx = [b for b in boundaries if b != 0]
    stable_mask = np.ones(novelty.size, dtype=bool)
    if transition_idx:
        stable_mask[transition_idx] = False

    transition_values = novelty[transition_idx] if transition_idx else np.array([], dtype=float)
    stable_values = novelty[stable_mask]
    transition_values = transition_values[np.isfinite(transition_values)]
    stable_values = stable_values[np.isfinite(stable_values)]

    mean_transition = float(transition_values.mean()) if transition_values.size else None
    mean_stable = float(stable_values.mean()) if stable_values.size else None
    if mean_transition is None or mean_stable is None or abs(mean_stable) < 1e-12:
        snr_ratio = None
    else:
        snr_ratio = float(mean_transition / mean_stable)
    return snr_ratio, mean_transition, mean_stable


def compute_oracle_efficiency(method_accuracy, source_accuracy, n_oracle_labels):
    """
    Accuracy gain per oracle label consumed.
    gain = method_accuracy - source_accuracy
    efficiency = gain / n_oracle_labels
    Returns: efficiency (float)
    """
    if method_accuracy is None or source_accuracy is None:
        return None
    try:
        n_oracle_labels = float(n_oracle_labels)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(n_oracle_labels) or n_oracle_labels <= 0:
        return None
    return float((method_accuracy - source_accuracy) / n_oracle_labels)


def compute_detection_latency(novelty_series, corruption_boundaries, threshold_multiplier=3.0):
    """
    For each corruption boundary, find how many batches until Nt exceeds
    mean_stable + threshold_multiplier * std_stable from the previous block.
    Returns: list of latencies (one per transition)
    """
    novelty = np.asarray(novelty_series, dtype=float)
    if novelty.size == 0:
        return []

    boundaries = sorted({int(b) for b in corruption_boundaries if 0 <= int(b) < novelty.size})
    if 0 not in boundaries:
        boundaries = [0] + boundaries

    latencies = []
    for idx, boundary in enumerate(boundaries):
        if boundary == 0:
            continue
        prev_boundary = boundaries[idx - 1]
        next_boundary = boundaries[idx + 1] if idx + 1 < len(boundaries) else novelty.size
        prev_block = novelty[prev_boundary:boundary]
        prev_block = prev_block[np.isfinite(prev_block)]
        if prev_block.size == 0:
            latencies.append(None)
            continue

        threshold = prev_block.mean() + threshold_multiplier * prev_block.std(ddof=0)
        search = novelty[boundary:next_boundary]
        detected = None
        for offset, value in enumerate(search):
            if np.isfinite(value) and value > threshold:
                detected = offset
                break
        latencies.append(detected)

    return latencies


def aggregate_block_metrics(metrics_log, stream_config=None):
    """
    Aggregate per-batch metrics into per-corruption-block summaries.
    Returns: dict keyed by corruption name with mean/std accuracy for all methods.
    """
    blocks = metrics_log.get("corruption_block", [])
    if not blocks and stream_config:
        blocks = stream_config.get("corruption_block", [])
    if not blocks:
        return {}

    methods = {
        "source": "batch_accuracy_source",
        "eatta": "batch_accuracy_eatta",
        "cas": "batch_accuracy_cas",
        "oatta": "batch_accuracy_oatta",
    }

    result = OrderedDict()
    for block in blocks:
        result.setdefault(str(block), {method: {"mean": None, "std": None} for method in methods})

    for block_name in result:
        indices = [idx for idx, block in enumerate(blocks) if str(block) == block_name]
        for method, key in methods.items():
            values = [metrics_log.get(key, [])[idx] for idx in indices if idx < len(metrics_log.get(key, []))]
            arr = _as_float_array(values)
            if arr.size:
                result[block_name][method] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
                }

    return dict(result)


def infer_boundaries_from_blocks(blocks):
    boundaries = []
    previous = object()
    for idx, block in enumerate(blocks):
        if idx == 0 or block != previous:
            boundaries.append(idx)
            previous = block
    return boundaries


def infer_boundaries_from_novelty(novelty_series, threshold_multiplier=3.0, min_distance=5):
    novelty = np.asarray([np.nan if value is None else float(value) for value in novelty_series], dtype=float)
    if novelty.size == 0 or np.all(~np.isfinite(novelty)):
        return []
    finite = novelty[np.isfinite(novelty)]
    threshold = finite.mean() + threshold_multiplier * finite.std(ddof=0)
    candidates = np.where(novelty > threshold)[0].tolist()
    boundaries = []
    last = -min_distance
    for idx in candidates:
        if idx - last >= min_distance:
            boundaries.append(int(idx))
            last = idx
    return boundaries
