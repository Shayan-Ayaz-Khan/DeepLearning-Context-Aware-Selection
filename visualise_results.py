from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B2",
}


def _style(fig, axes):
    fig.patch.set_facecolor("#111111")
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in np.ravel(axes):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#777777")
        ax.grid(True, color="#333333", linewidth=0.6, alpha=0.8)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#111111")
            legend.get_frame().set_edgecolor("#777777")
            for text in legend.get_texts():
                text.set_color("white")
    return fig


def _metrics(results):
    return results.get("metrics_log", {})


def _stream_name(results, stream_config):
    if stream_config and stream_config.get("stream_name"):
        return stream_config["stream_name"]
    return results.get("stream_name", "stream")


def _infer_spike_boundaries(novelty):
    values = np.asarray([np.nan if v is None else float(v) for v in novelty], dtype=float)
    if values.size == 0 or np.all(~np.isfinite(values)):
        return []
    finite = values[np.isfinite(values)]
    threshold = finite.mean() + 3.0 * finite.std()
    candidates = np.where(values > threshold)[0].tolist()
    boundaries = []
    last = -5
    for idx in candidates:
        if idx - last >= 5:
            boundaries.append(idx)
            last = idx
    return boundaries


def _boundaries(results, stream_config):
    metrics = _metrics(results)
    if stream_config:
        for key in ("corruption_boundaries", "boundaries"):
            if stream_config.get(key) is not None:
                return [int(x) for x in stream_config[key]]
    if metrics.get("corruption_boundaries"):
        return [int(x) for x in metrics["corruption_boundaries"]]
    blocks = metrics.get("corruption_block", [])
    if blocks:
        out = []
        prev = object()
        for idx, block in enumerate(blocks):
            if idx == 0 or block != prev:
                out.append(idx)
                prev = block
        return out
    return _infer_spike_boundaries(metrics.get("novelty_nt", []))


def _boundary_names(results, stream_config):
    metrics = _metrics(results)
    blocks = metrics.get("corruption_block", [])
    names = {}
    for boundary in _boundaries(results, stream_config):
        if 0 <= boundary < len(blocks):
            names[boundary] = str(blocks[boundary])
    return names


def _mark_boundaries(ax, results, stream_config):
    for boundary in _boundaries(results, stream_config):
        ax.axvline(boundary, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)


def _series(values):
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def _rolling_mean(values, window=5):
    arr = _series(values)
    if arr.size == 0:
        return arr
    out = np.full(arr.shape, np.nan)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        chunk = arr[start:idx + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[idx] = finite.mean()
    return out


def _oracle_label_count(results):
    for key in ("n_oracle_labels", "oracle_labels_used_total", "num_oracle_labels"):
        value = results.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            if value is not None and np.isfinite(value) and value > 0:
                return int(round(value))

    metrics = results.get("metrics_log", {}) or {}
    values = metrics.get("oracle_labels_used", results.get("oracle_labels_used", []))
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]

    total = 0
    seen = False
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            total += int(round(value))
            seen = True
    return total if seen and total > 0 else None


def _oracle_efficiency(method_accuracy, source_accuracy, n_oracle_labels):
    if method_accuracy is None or source_accuracy is None or n_oracle_labels is None:
        return None
    if n_oracle_labels <= 0:
        return None
    return float((method_accuracy - source_accuracy) / n_oracle_labels)


def merge_separate_run_results(source_results, eatta_results, cas_oatta_results):
    """
    Merge separate source, vanilla-EATTA, and CAS/OATTA run JSON objects into
    one result dict for plotting and summary tables.

    This keeps batch size and model memory clean during evaluation while still
    producing the four-method plots.
    """
    merged = deepcopy(cas_oatta_results)
    source_metrics = source_results.get("metrics_log", {})
    eatta_metrics = eatta_results.get("metrics_log", {})
    merged_metrics = merged.setdefault("metrics_log", {})

    source_accuracy = source_results.get("primary_accuracy", source_results.get("source_accuracy"))
    eatta_accuracy = eatta_results.get("primary_accuracy", eatta_results.get("eatta_accuracy"))
    cas_accuracy = cas_oatta_results.get("cas_accuracy", cas_oatta_results.get("eatta_accuracy"))
    oatta_accuracy = cas_oatta_results.get("oatta_accuracy", cas_oatta_results.get("primary_accuracy"))

    if source_accuracy is not None:
        merged["source_accuracy"] = source_accuracy
        merged["source_error"] = 1.0 - source_accuracy
    if eatta_accuracy is not None:
        merged["eatta_accuracy"] = eatta_accuracy
    if cas_accuracy is not None:
        merged["cas_accuracy"] = cas_accuracy
        merged["cas_pre_oatta_accuracy"] = cas_accuracy
    if oatta_accuracy is not None:
        merged["oatta_accuracy"] = oatta_accuracy
        merged["cas_oatta_accuracy"] = oatta_accuracy
        merged["primary_accuracy"] = oatta_accuracy
        merged["primary_error"] = 1.0 - oatta_accuracy

    if eatta_accuracy is not None and cas_accuracy is not None:
        merged["cas_gain_over_eatta"] = cas_accuracy - eatta_accuracy
    if eatta_accuracy is not None and oatta_accuracy is not None:
        merged["oatta_gain"] = oatta_accuracy - eatta_accuracy

    n_oracle_labels_cas = _oracle_label_count(cas_oatta_results)
    n_oracle_labels_eatta = _oracle_label_count(eatta_results)
    if n_oracle_labels_eatta is None:
        n_oracle_labels_eatta = n_oracle_labels_cas

    merged["n_oracle_labels_cas"] = n_oracle_labels_cas
    merged["n_oracle_labels_eatta"] = n_oracle_labels_eatta
    merged["oracle_efficiency_cas"] = _oracle_efficiency(
        cas_accuracy, source_accuracy, n_oracle_labels_cas
    )
    merged["oracle_efficiency_eatta"] = _oracle_efficiency(
        eatta_accuracy, source_accuracy, n_oracle_labels_eatta
    )
    merged_metrics["oracle_efficiency_cas"] = merged["oracle_efficiency_cas"]
    merged_metrics["oracle_efficiency_eatta"] = merged["oracle_efficiency_eatta"]

    if source_metrics.get("batch_accuracy_source"):
        merged_metrics["batch_accuracy_source"] = source_metrics["batch_accuracy_source"]
    if eatta_metrics.get("batch_accuracy_eatta"):
        merged_metrics["batch_accuracy_eatta"] = eatta_metrics["batch_accuracy_eatta"]

    block_metrics = deepcopy(cas_oatta_results.get("per_block_accuracy") or merged_metrics.get("block_accuracy", {}))
    source_blocks = source_results.get("per_block_accuracy") or source_metrics.get("block_accuracy", {})
    eatta_blocks = eatta_results.get("per_block_accuracy") or eatta_metrics.get("block_accuracy", {})

    for block_name, values in source_blocks.items():
        block_metrics.setdefault(block_name, {})
        if isinstance(values, dict) and "source" in values:
            block_metrics[block_name]["source"] = values["source"]

    for block_name, values in eatta_blocks.items():
        block_metrics.setdefault(block_name, {})
        if isinstance(values, dict) and "eatta" in values:
            block_metrics[block_name]["eatta"] = values["eatta"]

    merged["per_block_accuracy"] = block_metrics
    merged_metrics["block_accuracy"] = block_metrics
    merged["source_results_path"] = source_results.get("results_path")
    merged["eatta_results_path"] = eatta_results.get("results_path")
    merged["cas_oatta_results_path"] = cas_oatta_results.get("results_path")
    merged["method_variant"] = "merged_source_eatta_cas_oatta"
    return merged


def plot_accuracy_curves(results, stream_config):
    """
    Plot 1: Per-batch accuracy for Source, EATTA, CAS, OATTA.
    X-axis: batch index. Y-axis: accuracy (%).
    Four lines, vertical dashed lines at corruption boundaries.
    Title: 'Per-Batch Accuracy — {stream_name}'
    """
    metrics = _metrics(results)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(metrics.get("batch_accuracy_source", [])))
    ax.plot(x, _series(metrics.get("batch_accuracy_source", [])), label="Source", color=PALETTE["blue"])
    ax.plot(x, _series(metrics.get("batch_accuracy_eatta", [])), label="EATTA", color=PALETTE["orange"])
    ax.plot(x, _series(metrics.get("batch_accuracy_cas", [])), label="CAS (pre-OATTA)", color=PALETTE["green"])
    ax.plot(x, _series(metrics.get("batch_accuracy_oatta", [])), label="CAS+OATTA", color=PALETTE["red"])
    _mark_boundaries(ax, results, stream_config)
    ax.set_title(f"Per-Batch Accuracy — {_stream_name(results, stream_config)}")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    return _style(fig, ax)


def plot_novelty_timeseries(results, stream_config):
    """
    Plot 2: Novelty score Nt across all batches.
    X-axis: batch index. Y-axis: Nt (log scale recommended).
    Single line, vertical dashed lines at corruption boundaries.
    Annotate each spike with the corruption transition name.
    Title: 'BN Anchor Novelty Score — {stream_name}'
    """
    metrics = _metrics(results)
    novelty = _series(metrics.get("novelty_nt", []))
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(novelty))
    ax.plot(x, novelty, label="Nt", color=PALETTE["purple"])
    positive = novelty[np.isfinite(novelty) & (novelty > 0)]
    if positive.size:
        ax.set_yscale("log")
    _mark_boundaries(ax, results, stream_config)
    names = _boundary_names(results, stream_config)
    finite = novelty[np.isfinite(novelty)]
    y_top = finite.max() if finite.size else 1.0
    for boundary, name in names.items():
        if boundary == 0:
            continue
        ax.annotate(name, xy=(boundary, y_top), xytext=(4, -18),
                    textcoords="offset points", color="white", rotation=90,
                    va="top", fontsize=8)
    ax.set_title(f"BN Anchor Novelty Score — {_stream_name(results, stream_config)}")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Nt")
    ax.legend()
    return _style(fig, ax)


def plot_selection_behaviour(results, stream_config):
    """
    Plot 3: Two-panel figure.
    Top panel: rolling mean (window=5) of cas_eatta_divergence across batches.
    Bottom panel: novelty score of CAS-selected sample vs batch mean novelty.
    Shared X-axis with corruption boundaries.
    Title: 'CAS Selection Behaviour — {stream_name}'
    """
    metrics = _metrics(results)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    x = np.arange(len(metrics.get("cas_eatta_divergence", [])))
    axes[0].plot(x, _rolling_mean(metrics.get("cas_eatta_divergence", []), 5),
                 label="CAS vs EATTA Selection Divergence", color=PALETTE["red"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Divergence rate")
    axes[0].legend()

    axes[1].plot(x, _series(metrics.get("novelty_selected", [])),
                 label="CAS-selected sample novelty", color=PALETTE["green"])
    axes[1].plot(x, _series(metrics.get("novelty_batch_mean", [])),
                 label="Batch mean novelty", color=PALETTE["blue"])
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Novelty")
    axes[1].legend()
    for ax in axes:
        _mark_boundaries(ax, results, stream_config)
    fig.suptitle(f"CAS Selection Behaviour — {_stream_name(results, stream_config)}", color="white")
    return _style(fig, axes)


def plot_per_corruption_accuracy(results):
    """
    Plot 4: Grouped bar chart.
    One group per corruption type. Four bars per group: Source, EATTA, CAS, OATTA.
    Error bars from std if multiple runs available.
    X-axis: corruption name. Y-axis: accuracy (%).
    Title: 'Per-Corruption Accuracy Breakdown'
    """
    block_metrics = results.get("per_block_accuracy") or _metrics(results).get("block_accuracy", {})
    methods = ["source", "eatta", "cas", "oatta"]
    labels = {
        "source": "SOURCE",
        "eatta": "EATTA",
        "cas": "CAS (pre-OATTA)",
        "oatta": "CAS+OATTA",
    }
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
    names = list(block_metrics.keys())
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, (method, color) in enumerate(zip(methods, colors)):
        means = []
        stds = []
        for name in names:
            entry = block_metrics.get(name, {}).get(method, {})
            if isinstance(entry, dict):
                means.append(np.nan if entry.get("mean") is None else entry.get("mean"))
                stds.append(0.0 if entry.get("std") is None else entry.get("std"))
            else:
                means.append(np.nan)
                stds.append(0.0)
        ax.bar(x + (offset - 1.5) * width, means, width, label=labels[method],
               color=color, yerr=stds if results.get("n_runs", 1) > 1 else None,
               capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Corruption")
    ax.set_title("Per-Corruption Accuracy Breakdown")
    ax.legend()
    return _style(fig, ax)


def plot_oatta_gate(results, stream_config):
    """
    Plot 5: Two-panel figure.
    Top panel: LLR accumulator value across batches.
    Bottom panel: lambda values across batches.
    Both with corruption boundary markers.
    Add horizontal reference line at lambda=0.5 in bottom panel.
    Title: 'OATTA Gate Behaviour — {stream_name}'
    """
    metrics = _metrics(results)
    llr = _series(metrics.get("llr_accumulator", []))
    lambdas = _series(metrics.get("lambda_values", []))
    x = np.arange(max(len(llr), len(lambdas)))
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(np.arange(len(llr)), llr, label="LLR accumulator", color=PALETTE["purple"])
    axes[0].set_ylabel("LLR")
    axes[0].legend()
    axes[1].plot(np.arange(len(lambdas)), lambdas, label="Lambda", color=PALETTE["orange"])
    axes[1].axhline(0.5, color="#999999", linestyle=":", linewidth=1.0, label="lambda=0.5")
    if lambdas.size and np.nanmax(lambdas) <= 0.1:
        axes[1].text(0.02, 0.9, "Gate remained closed — no temporal class structure detected",
                     transform=axes[1].transAxes, color="white", fontsize=10)
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Lambda")
    axes[1].legend()
    for ax in axes:
        _mark_boundaries(ax, results, stream_config)
    axes[1].set_xlim(0, max(len(x) - 1, 1))
    fig.suptitle(f"OATTA Gate Behaviour — {_stream_name(results, stream_config)}", color="white")
    return _style(fig, axes)


def generate_all_plots(results, stream_config, output_dir):
    """
    Call all five plot functions, save each as a PNG to output_dir,
    and return a dict of {plot_name: figure} for inline notebook display.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures = {
        "accuracy_curves": plot_accuracy_curves(results, stream_config),
        "novelty_timeseries": plot_novelty_timeseries(results, stream_config),
        "selection_behaviour": plot_selection_behaviour(results, stream_config),
        "per_corruption_accuracy": plot_per_corruption_accuracy(results),
        "oatta_gate": plot_oatta_gate(results, stream_config),
    }
    for name, fig in figures.items():
        fig.savefig(output_path / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    return figures
