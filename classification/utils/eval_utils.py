# import torch
# import logging
# import numpy as np
# from typing import Union


# logger = logging.getLogger(__name__)


# def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
#     """
#     Separates the label prediction pairs by domain
#     Input:
#         domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
#         data: List containing [images, labels, domains, ...]
#         predictions: Tensor containing the predictions of the model
#     Returns:
#         domain_dict: Updated dictionary containing the domain seperated label prediction pairs
#     """

#     labels, domains = data[1], data[2]
#     assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

#     for i in range(labels.shape[0]):
#         if domains[i] in domain_dict.keys():
#             domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
#         else:
#             domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

#     return domain_dict


# def eval_domain_dict(domain_dict: dict, domain_seq: list):
#     """
#     Print detailed results for each domain. This is useful for settings where the domains are mixed
#     Input:
#         domain_dict: Dictionary containing the labels and predictions for each domain
#         domain_seq: Order to print the results (if all domains are contained in the domain dict)
#     """
#     correct = []
#     num_samples = []
#     avg_error_domains = []
#     domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
#     logger.info(f"Splitting the results by domain...")
#     for key in domain_names:
#         label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
#         correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
#         num_samples.append(label_prediction_arr.shape[0])
#         accuracy = correct[-1] / num_samples[-1]
#         error = 1 - accuracy
#         avg_error_domains.append(error)
#         logger.info(f"{key:<20} error: {error:.2%}")
#     logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
#     # The error across all samples differs if each domain contains different amounts of samples
#     logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


# def get_accuracy(model: torch.nn.Module,
#                  data_loader: torch.utils.data.DataLoader,
#                  dataset_name: str,
#                  domain_name: str,
#                  setting: str,
#                  domain_dict: dict,
#                  print_every: int,
#                  device: Union[str, torch.device]):

#     num_correct = 0.
#     num_samples = 0
#     with torch.no_grad():
#         for i, data in enumerate(data_loader):
#             imgs, labels = data[0], data[1]
#             output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device), labels.to(device))
#             predictions = output.argmax(1)


#             num_correct += (predictions == labels.to(device)).float().sum()

#             if "mixed_domains" in setting and len(data) >= 3:
#                 domain_dict = split_results_by_domain(domain_dict, data, predictions)

#             # track progress
#             num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
#             if print_every > 0 and (i+1) % print_every == 0:
#                 logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

#             if dataset_name == "ccc" and num_samples >= 7500000:
#                 break

#     accuracy = num_correct.item() / num_samples
#     return accuracy, domain_dict, num_samples


import torch
import logging
import numpy as np
from collections import Counter
from typing import Union
from utils.metrics import (
    aggregate_block_metrics,
    compute_detection_latency,
    compute_oracle_efficiency,
    compute_snr,
    infer_boundaries_from_blocks,
    infer_boundaries_from_novelty,
)


logger = logging.getLogger(__name__)


def _empty_metrics_log():
    return {
        "batch_accuracy_source": [],
        "batch_accuracy_eatta": [],
        "batch_accuracy_cas": [],
        "batch_accuracy_oatta": [],
        "novelty_nt": [],
        "novelty_selected": [],
        "novelty_batch_mean": [],
        "llr_accumulator": [],
        "lambda_values": [],
        "cas_eatta_divergence": [],
        "selected_sample_diff": [],
        "batch_mean_diff": [],
        "corruption_block": [],
        "batch_entropy_mean": [],
        "block_accuracy": {},
        "block_novelty_mean": [],
        "block_novelty_at_transition": [],
        "detection_latency": [],
        "mean_lambda": None,
        "llr_range": [None, None],
        "selection_divergence_rate": None,
        "oracle_efficiency_cas": None,
        "oracle_efficiency_eatta": None,
        "corruption_boundaries": [],
        "oracle_labels_used": [],
    }


def _batch_size(imgs):
    return imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]


def _batch_accuracy(predictions, labels_dev):
    return float((predictions == labels_dev).float().mean().item() * 100.0)


def _batch_block_name(data):
    if len(data) < 3:
        return "unknown"
    blocks = data[2]
    if isinstance(blocks, (list, tuple)):
        return Counter([str(block) for block in blocks]).most_common(1)[0][0]
    if torch.is_tensor(blocks):
        return str(blocks[0].item()) if blocks.numel() else "unknown"
    return str(blocks)


def _batch_has_shift(data):
    if len(data) < 5:
        return False
    shifts = data[4]
    if torch.is_tensor(shifts):
        return bool((shifts > 0).any().item())
    if isinstance(shifts, (list, tuple)):
        return any(int(s) > 0 for s in shifts)
    return bool(shifts)


def _finite_mean(values, scale=1.0):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals) / scale) if vals else None


def _sum_oracle_labels(values):
    total = 0
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            total += int(round(value))
    return total


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """
    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device]):
    """
    Evaluates model accuracy over a data stream.

    If the model returns a tuple (outputs, p_hat) — as EATTA+OATTA does —
    both raw EATTA accuracy and OATTA-refined accuracy are tracked and logged.
    If the model returns a single tensor, the function behaves exactly as before.

    Returns
    -------
    accuracy      : float  - primary accuracy (OATTA-refined if available, else raw)
    domain_dict   : dict   - updated domain breakdown
    num_samples   : int    - total samples processed
    results        : dict   - scalar summaries plus per-batch metrics_log
    """

    num_correct_eatta = 0.   # raw EATTA predictions  (argmax of logits)
    num_correct_oatta = 0.   # OATTA-refined predictions  (argmax of p_hat)
    num_correct_source = 0.
    num_correct_cas = 0.
    num_samples = 0
    is_dual_output = None    # detected on first batch
    metrics_log = _empty_metrics_log()
    has_independent_source_outputs = False

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            labels_dev = labels.to(device)

            # ── Forward pass ──────────────────────────────────────────────
            raw_output = (
                model([img.to(device) for img in imgs])
                if isinstance(imgs, list)
                else model(imgs.to(device), labels_dev)
            )

            # ── Detect output format on first batch ───────────────────────
            # EATTA+OATTA returns (outputs, p_hat) — a tuple of two tensors.
            # Plain EATTA / other methods return a single tensor.
            if is_dual_output is None:
                is_dual_output = isinstance(raw_output, tuple) and len(raw_output) == 2

            batch_aux = getattr(model, "last_batch_metrics", {}) or {}

            if is_dual_output:
                # Unpack: outputs = raw logits, p_hat = OATTA posteriors
                outputs, p_hat = raw_output
                eatta_preds = outputs.argmax(dim=1)   # EATTA-only prediction
                oatta_preds = p_hat.argmax(dim=1)     # EATTA + OATTA prediction
                cas_outputs = batch_aux.get("cas_outputs", outputs)
                source_outputs = batch_aux.get("source_outputs")
                cas_preds = cas_outputs.argmax(dim=1)
                if source_outputs is not None:
                    has_independent_source_outputs = True
                    source_preds = source_outputs.argmax(dim=1)
                else:
                    source_preds = eatta_preds
                # Primary predictions for domain tracking = OATTA-refined
                predictions = oatta_preds
            else:
                # Unchanged path — any method that returns a single tensor
                outputs = raw_output
                eatta_preds = outputs.argmax(dim=1)
                oatta_preds = None
                cas_outputs = batch_aux.get("cas_outputs")
                source_outputs = batch_aux.get("source_outputs")
                cas_preds = cas_outputs.argmax(dim=1) if cas_outputs is not None else eatta_preds
                if source_outputs is not None:
                    has_independent_source_outputs = True
                    source_preds = source_outputs.argmax(dim=1)
                else:
                    source_preds = eatta_preds
                predictions = eatta_preds

            # ── Accuracy accumulation ─────────────────────────────────────
            num_correct_eatta += (eatta_preds == labels_dev).float().sum()
            num_correct_cas += (cas_preds == labels_dev).float().sum()
            num_correct_source += (source_preds == labels_dev).float().sum()
            if oatta_preds is not None:
                num_correct_oatta += (oatta_preds == labels_dev).float().sum()

            metrics_log["batch_accuracy_source"].append(_batch_accuracy(source_preds, labels_dev))
            metrics_log["batch_accuracy_eatta"].append(_batch_accuracy(eatta_preds, labels_dev))
            metrics_log["batch_accuracy_cas"].append(_batch_accuracy(cas_preds, labels_dev))
            metrics_log["batch_accuracy_oatta"].append(
                _batch_accuracy(oatta_preds, labels_dev) if oatta_preds is not None else None
            )
            for key in [
                "novelty_nt",
                "novelty_selected",
                "novelty_batch_mean",
                "llr_accumulator",
                "lambda_values",
                "cas_eatta_divergence",
                "selected_sample_diff",
                "batch_mean_diff",
                "batch_entropy_mean",
                "oracle_labels_used",
            ]:
                metrics_log[key].append(batch_aux.get(key))

            block_name = _batch_block_name(data)
            if i == 0 or _batch_has_shift(data) or (
                metrics_log["corruption_block"] and block_name != metrics_log["corruption_block"][-1]
            ):
                metrics_log["corruption_boundaries"].append(i)
            metrics_log["corruption_block"].append(block_name)

            # ── Domain tracking (unchanged) ───────────────────────────────
            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # ── Progress logging ──────────────────────────────────────────
            num_samples += _batch_size(imgs)

            if print_every > 0 and (i + 1) % print_every == 0:
                eatta_err = 1 - num_correct_eatta / num_samples
                log_msg = f"#batches={i+1:<6} #samples={num_samples:<9} EATTA error={eatta_err:.2%}"
                if is_dual_output:
                    oatta_err = 1 - num_correct_oatta / num_samples
                    log_msg += f"  |  OATTA error={oatta_err:.2%}"
                    log_msg += f"  |  gain={eatta_err - oatta_err:+.2%}"
                logger.info(log_msg)

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

    # ── Final summary ─────────────────────────────────────────────────────
    eatta_accuracy = num_correct_eatta.item() / num_samples
    cas_accuracy = num_correct_cas.item() / num_samples
    source_accuracy = num_correct_source.item() / num_samples

    if not metrics_log["corruption_boundaries"]:
        metrics_log["corruption_boundaries"] = infer_boundaries_from_blocks(metrics_log["corruption_block"])
    if len(metrics_log["corruption_boundaries"]) <= 1:
        inferred = infer_boundaries_from_novelty(metrics_log["novelty_nt"])
        if inferred:
            metrics_log["corruption_boundaries"] = sorted({0, *inferred})

    block_accuracy = aggregate_block_metrics(metrics_log, {"domain_name": domain_name})
    metrics_log["block_accuracy"] = block_accuracy

    novelty = metrics_log["novelty_nt"]
    boundaries = metrics_log["corruption_boundaries"]
    detection_latency = compute_detection_latency(novelty, boundaries)
    metrics_log["detection_latency"] = detection_latency

    block_novelty_mean = []
    block_novelty_at_transition = []
    for idx, boundary in enumerate(boundaries):
        next_boundary = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(novelty)
        vals = [v for v in novelty[boundary:next_boundary] if v is not None and np.isfinite(v)]
        block_novelty_mean.append(float(np.mean(vals)) if vals else None)
        block_novelty_at_transition.append(novelty[boundary] if boundary < len(novelty) else None)
    metrics_log["block_novelty_mean"] = block_novelty_mean
    metrics_log["block_novelty_at_transition"] = block_novelty_at_transition

    lambda_vals = [v for v in metrics_log["lambda_values"] if v is not None and np.isfinite(v)]
    llr_vals = [v for v in metrics_log["llr_accumulator"] if v is not None and np.isfinite(v)]
    div_vals = [v for v in metrics_log["cas_eatta_divergence"] if v is not None]
    metrics_log["mean_lambda"] = float(np.mean(lambda_vals)) if lambda_vals else None
    metrics_log["llr_range"] = [float(np.min(llr_vals)), float(np.max(llr_vals))] if llr_vals else [None, None]
    metrics_log["selection_divergence_rate"] = float(np.mean(div_vals)) if div_vals else None

    snr_ratio, mean_nt_transition, mean_nt_stable = compute_snr(novelty, boundaries)

    if is_dual_output:
        oatta_accuracy = num_correct_oatta.item() / num_samples
        logger.info(
            f"Final accuracy — "
            f"EATTA: {eatta_accuracy:.2%}  |  "
            f"OATTA: {oatta_accuracy:.2%}  |  "
            f"gain: {oatta_accuracy - eatta_accuracy:+.2%}"
        )
        primary_accuracy = oatta_accuracy
    else:
        oatta_accuracy = None
        primary_accuracy = eatta_accuracy

    cas_enabled = getattr(model, "cas_enabled", None)
    oatta_enabled = getattr(model, "oatta_enabled", None)
    method_variant = getattr(model, "method_variant", model.__class__.__name__.lower())
    n_oracle_labels = _sum_oracle_labels(metrics_log["oracle_labels_used"])

    # With the separate-run workflow, non-source runs do not carry an
    # independent source baseline in-process. Leave these per-run efficiencies
    # unset and let merge_separate_run_results recompute them against the real
    # source run.
    source_for_efficiency = source_accuracy if has_independent_source_outputs else None
    metrics_log["oracle_efficiency_cas"] = compute_oracle_efficiency(
        cas_accuracy, source_for_efficiency, n_oracle_labels
    )
    metrics_log["oracle_efficiency_eatta"] = compute_oracle_efficiency(
        eatta_accuracy, source_for_efficiency, n_oracle_labels
    )

    results = {
        "source_error": 1.0 - source_accuracy if source_accuracy is not None else None,
        "source_accuracy": source_accuracy,
        "eatta_accuracy": eatta_accuracy,
        "oatta_accuracy": oatta_accuracy,
        "oatta_gain": (oatta_accuracy - eatta_accuracy) if oatta_accuracy is not None else None,
        "primary_error": 1.0 - primary_accuracy,
        "primary_accuracy": primary_accuracy,
        "run_dir": None,
        "log_file": None,
        "cas_accuracy": cas_accuracy,
        "cas_gain_over_eatta": cas_accuracy - eatta_accuracy,
        "per_block_accuracy": block_accuracy,
        "mean_nt_stable": mean_nt_stable,
        "mean_nt_transition": mean_nt_transition,
        "snr_ratio": snr_ratio,
        "detection_latency_mean": _finite_mean(detection_latency),
        "selection_divergence_rate": metrics_log["selection_divergence_rate"],
        "mean_lambda": metrics_log["mean_lambda"],
        "llr_min": metrics_log["llr_range"][0],
        "llr_max": metrics_log["llr_range"][1],
        "oracle_efficiency_cas": metrics_log["oracle_efficiency_cas"],
        "oracle_efficiency_eatta": metrics_log["oracle_efficiency_eatta"],
        "n_oracle_labels": n_oracle_labels,
        "n_runs": 1,
        "accuracy_std_across_runs": None,
        "cas_enabled": cas_enabled,
        "oatta_enabled": oatta_enabled,
        "method_variant": method_variant,
        "metrics_log": metrics_log,
    }

    return primary_accuracy, domain_dict, num_samples, results
