# EATTA+CAS: Context-Aware Active Test-Time Adaptation

> Extending EATTA with temporal memory via BN Statistics Anchoring and Bayesian Temporal Filtering for robust adaptation to distribution shift at inference time.

**Lahore University of Management Sciences**
Shayan Ayaz Khan (27100290) · Umer Ashraf (27100236)

---

## Architecture

<img width="363" height="640" alt="model_architecture" src="https://github.com/user-attachments/assets/9d4adeb5-e7b3-4fd0-a7b2-7ae3c82a927e" />


The pipeline executes per batch: forward pass → entropy filter → parallel scoring (BN novelty anchor + perturbation score) → CAS multiplicative selection → dual loss BN update → OATTA Bayesian filter → final prediction.

---

## Overview

Standard active TTA methods (e.g. EATTA) select oracle-labelled samples based solely on instantaneous model uncertainty, with no memory of how the visual environment has evolved. We propose **Context-Aware Selection (CAS)**, which adds two temporal context modules:

- **BN Statistics Anchor** — tracks an EMA of pre-normalisation BatchNorm activation statistics to detect corruption-type transitions with 7–11× signal-to-noise ratio and zero detection latency.
- **OATTA Bayesian Filter with LLR Gate** — refines predictions using learned class-transition dynamics, remaining fully transparent (mean λ < 0.02) when no sequential class structure is present.

### Key Results

| Stream | Source | EATTA | CAS | CAS+OATTA |
|--------|--------|-------|-----|-----------|
| Abrupt (mean ± std, 3 seeds) | 58.55% | 75.95 ± 0.07% | **75.99 ± 0.13%** | 76.00 ± 0.09% |
| Driving (mean ± std, 3 seeds) | 57.43% | 73.07 ± 0.04% | **73.14 ± 0.14%** | 73.09 ± 0.16% |

---

## Repository Structure
---
<img width="1456" height="1120" alt="image" src="https://github.com/user-attachments/assets/55f1ea18-8584-40c0-8504-53b5e48ccd92" />

### 3. Dataset

This project uses **ImageNet-100-C** (ImageNet-100 with 15 corruption types × 5 severity levels, 5,000 images each).

Place the dataset on your Google Drive. The evaluation notebook (`imagenet100cEAtta.ipynb`) pulls the dataset and repository directly from Drive at runtime — no local download required if running on Google Colab.

### 4. Build evaluation streams

Run `build_sequential_stream.ipynb` to generate the CSV files defining the stream sample orderings.

Two streams are supported:

**Abrupt Stream** — 6 corruption types at uniform severity 3, 5,000 images per block, 469 total batches. Designed to maximise clean corruption-type transitions for BN anchor evaluation.

| Block | Corruption | Severity |
|-------|-----------|----------|
| 1 | Fog | 3 |
| 2 | Gaussian Noise | 3 |
| 3 | Motion Blur | 3 |
| 4 | Frost | 3 |
| 5 | Contrast | 3 |
| 6 | JPEG Compression | 3 |

**Driving Stream** — 7 corruption types at mixed severities simulating a realistic deployment scenario (dawn → fog → highway → tunnel → rain → snow → glare), 2,000–3,000 images per block, 297 total batches.

| Block | Corruption | Severity | Scenario |
|-------|-----------|----------|----------|
| 1 | Brightness | 2 | Dawn |
| 2 | Fog | 4 | Heavy fog |
| 3 | Motion Blur | 2 | Highway speed |
| 4 | Contrast | 5 | Dark tunnel |
| 5 | Gaussian Noise | 3 | Rain / sensor noise |
| 6 | Snow | 3 | Winter conditions |
| 7 | Brightness | 5 | Strong glare |

Each stream is evaluated under **3 independent random seeds** (42, 123, 456), which shuffle the within-block sample ordering while preserving the corruption block structure. This gives 6 total runs and allows assessment of result stability across different batch compositions.


## Running Experiments

Open `imagenet100cEAtta.ipynb` in Google Colab. The notebook:
1. Mounts Google Drive and loads the dataset and repo
2. Runs all four methods (Source, EATTA, CAS, CAS+OATTA) simultaneously on each batch
3. Saves per-batch accuracy, novelty scores, gate behaviour, and selection diagnostics to a results JSON
4. Calls `save_plots.py` to generate all five diagnostic plots per run

Run the notebook once per stream × seed combination. Evaluated configurations:

| Stream | Seeds |
|--------|-------|
| Abrupt (fog → gaussian noise → motion blur → frost → contrast → jpeg, sev. 3) | 42, 123, 456 |
| Driving (brightness → fog → motion blur → contrast → gaussian noise → snow → brightness, mixed sev.) | 42, 123, 456 |

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 64 | Images per batch |
| Oracle budget | 1 label/batch | Oracle labels requested per batch |
| BN anchor γ | 0.5 | EMA decay for anchor update |
| Anchor warmup | 3 batches | Batches before anchor activates |
| Hooked layers | layer1, layer2 | ResNet-50 layers with BN hooks (23 layers) |
| Statistics dim | 9984 | Dimension of concatenated BN statistics vector |
| LLR window W | 10 | Sliding window for LLR accumulation |
| LLR margin m | 1.0 | Threshold for gate opening |
| LLR temperature τ | 1.0 | Sigmoid sharpness for gate weight |
| Updated params | BN γ, β (layers 1–3) | Only BatchNorm affine params updated |

---

## Output

Each run produces a `results_merged.json` containing:
- Per-batch accuracy for all four methods
- BN anchor novelty scores (`novelty_nt`, `novelty_selected`, `novelty_batch_mean`)
- OATTA gate values (`llr_accumulator`, `lambda_values`)
- CAS vs. EATTA selection divergence
- Per-block accuracy breakdown

Plots are saved automatically by `save_plots.py`:
1. Per-batch accuracy curves
2. BN anchor novelty time series
3. CAS selection behaviour
4. Per-corruption accuracy breakdown
5. OATTA gate behaviour

---

## Citation

```bibtex
@misc{khan2025cas,
  title     = {Context-Aware Active Test-Time Adaptation via
               Batch Normalisation Statistics Anchoring and
               Bayesian Temporal Filtering},
  author    = {Khan, Shayan Ayaz and Ashraf, Umer},
  year      = {2025},
  school    = {Lahore University of Management Sciences},
  note      = {Undergraduate Research Project},
  url       = {https://github.com/Shayan-Ayaz-Khan/DeepLearning-Context-Aware-Selection}
}
```

---

## Acknowledgements

Built on top of [EATTA](https://arxiv.org/abs/2501.04858) (Wang et al., CVPR 2025) and integrates the [OATTA](https://arxiv.org/abs/2601.21012) Bayesian filter (Kim et al., 2026). Corruption benchmarks from [ImageNet-C](https://arxiv.org/abs/1903.12261) (Hendrycks & Dietterich, 2019).
