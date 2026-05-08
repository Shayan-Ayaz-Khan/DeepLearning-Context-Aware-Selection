import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import copy
import random
from collections import OrderedDict
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from models.model import split_up_model
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# OATTAFilter - Recursive Bayesian Temporal Filter with LLR Gate
# Kim et al. (2026) - Algorithm 1 + Algorithm 2
# Operates in output probability space. Does NOT touch model weights.
# LLR gate guarantees worst-case = EATTA accuracy on unstructured streams.
# ============================================================================

class OATTAFilter:
    def __init__(self,
                 num_classes : int,
                 kappa       : float = None,
                 gamma       : float = 0.1,
                 tau         : float = 1.0,
                 use_llr     : bool  = True,
                 llr_window  : int   = 10,
                 llr_margin  : float = 1.0,
                 llr_tau     : float = 1.0,
                 eta         : float = 0.1,
                 eps         : float = 1e-8):

        self.K       = num_classes
        self.gamma   = gamma
        self.tau     = tau
        self.eps     = eps
        self.use_llr = use_llr
        self.W       = llr_window
        self.m       = llr_margin
        self.llr_tau = llr_tau
        self.eta     = eta

        kappa = kappa if kappa is not None else 1.0 / num_classes
 
        self.C      = kappa * torch.eye(num_classes)
        self.A      = self._row_normalise(self.C)
        self.p_prev = torch.full((num_classes,), 1.0 / num_classes)
        self.q_prev = torch.full((num_classes,), 1.0 / num_classes)
        self.pi_bar = torch.full((num_classes,), 1.0 / num_classes)
        self.llr_t  = 0.0
 
        self._batch_count  = 0
        self._lambda_accum = 0.0
        self.last_mean_lambda = 0.0
        self.device = None
 
    def _row_normalise(self, M):
        return M / M.sum(dim=1, keepdim=True).clamp(min=self.eps)

    def _to_device(self, device):
        if self.device != device:
            self.C      = self.C.to(device)
            self.A      = self.A.to(device)
            self.p_prev = self.p_prev.to(device)
            self.q_prev = self.q_prev.to(device)
            self.pi_bar = self.pi_bar.to(device)
            self.device = device
 
    def _entropy(self, q):
        return -(q * (q + self.eps).log()).sum()

    def _filter_single(self, q_t):
        pi_t = self.A.T @ self.p_prev
        p_t  = q_t * pi_t
        p_t  = p_t / p_t.sum().clamp(min=self.eps)
 
        if self.use_llr:
            self.pi_bar = (1.0 - self.eta) * self.pi_bar + self.eta * q_t
            self.pi_bar = self.pi_bar / self.pi_bar.sum().clamp(min=self.eps)
            delta_t = (
                torch.log(torch.dot(q_t, pi_t)        + self.eps)
              - torch.log(torch.dot(q_t, self.pi_bar) + self.eps)
            ).item()
            self.llr_t = (1.0 - 1.0 / self.W) * self.llr_t + (1.0 / self.W) * delta_t
            lambda_t = torch.sigmoid(
                torch.tensor((self.llr_t - self.m) / self.llr_tau,
                             dtype=torch.float32, device=self.device)
            ).item()
            p_hat_t = lambda_t * p_t + (1.0 - lambda_t) * q_t
            p_hat_t = p_hat_t / p_hat_t.sum().clamp(min=self.eps)
        else:
            p_hat_t  = p_t
            lambda_t = 1.0
 
        self._lambda_accum += lambda_t
 
        H_qt   = self._entropy(q_t)
        w_t    = torch.exp(-H_qt / self.tau)
        outer  = torch.outer(self.q_prev, q_t)
        self.C = (1.0 - self.gamma * w_t) * self.C + self.gamma * w_t * outer
        self.A = self._row_normalise(self.C)
 
        self.p_prev = p_hat_t.detach()
        self.q_prev = q_t.detach()
        return p_hat_t

    def filter_batch(self, logits):
        device = logits.device
        self._to_device(device)
        q_batch = F.softmax(logits.detach(), dim=-1)
        p_hat   = torch.zeros_like(q_batch)
        self._lambda_accum = 0.0
        for i in range(q_batch.size(0)):
            p_hat[i] = self._filter_single(q_batch[i])
        self._batch_count += 1
        self.last_mean_lambda = self._lambda_accum / max(q_batch.size(0), 1)
        if self._batch_count % 10 == 0:
            print(f"[OATTA] batch={self._batch_count:>4}  |  "
                  f"mean lambda={self.last_mean_lambda:.3f}  |  "
                  f"A diag={self.A.diag().mean().item():.3f}  |  "
                  f"LLR={self.llr_t:.3f}")
        return p_hat

    def reset(self):
        kappa  = 1.0 / self.K
        device = self.device if self.device is not None else torch.device('cpu')
        self.C      = (kappa * torch.eye(self.K)).to(device)
        self.A      = self._row_normalise(self.C)
        self.p_prev = torch.full((self.K,), 1.0 / self.K, device=device)
        self.q_prev = torch.full((self.K,), 1.0 / self.K, device=device)
        self.pi_bar = torch.full((self.K,), 1.0 / self.K, device=device)
        self.llr_t  = 0.0
        self._batch_count  = 0
        self._lambda_accum = 0.0
        self.last_mean_lambda = 0.0
 
    def get_transition_matrix(self):
        return self.A.detach().cpu()

    def get_llr(self):
        return self.llr_t


# ============================================================================
# BNStatisticsAnchor - EMA anchor over Batch Normalisation statistics
#
# -------------------------------------------
# ResNet-50 features encode semantic content (WHAT the object is) much more
# strongly than corruption type (HOW it looks). A foggy car and a sharp car
# have similar features. Mixed-class random streams therefore show no
# systematic shift in feature space at corruption boundaries.
#
# Batch Normalisation statistics (per-channel mean and variance computed
# over the batch) behave differently:
# - They are computed BEFORE semantic processing, closer to pixel statistics.
# - They are sensitive to low-level distributional properties: blur changes
#   edge statistics, noise raises variance, fog shifts mean intensity.
# - They are INSENSITIVE to which class is present, because BN normalises
#   out semantic variation within a batch.
#
# This means BN statistics shift systematically at corruption boundaries
# even in mixed-class random streams - exactly the signal we need.
#
# IMPLEMENTATION
# ---------------
# hook into selected BN layers (early-to-mid network, where corruption
# signal is strongest before being abstracted away) and record the batch
# mean and variance for each channel. These are concatenated into a single
# statistics vector, then tracked with an EMA anchor.
#
# Novelty = ||s_t - anchor_t||^2  where s_t is the current BN stats vector.
#
# --------------------------------
# - NOTE (Gong et al., NeurIPS 2022): BN statistics drift under temporal
#   correlation, validating their sensitivity to distribution shift.
# - MemBN (Kang et al., ECCV 2024): BN statistics capture distributional
#   context better than output-layer features for TTA.
# - E-TTA (Kubo, 2025): EMA anchors produce bounded, stable trajectories.
# ============================================================================


class BNStatisticsAnchor:
    # Tracks an EMA of Batch Normalisation statistics across batches and
    # computes per-batch novelty scores indicating environmental shift.

    # Unlike feature-vector anchors, BN statistics are sensitive to corruption
    # type but insensitive to class content, making them ideal for detecting
    # corruption boundary crossings in mixed-class streams.

    # Parameters
    # ----------
    # model : nn.Module
    #     The edge model (self.model[1]) to hook into.
    # gamma : float
    #     EMA momentum for the anchor. 0.9 = slow adaptation.
    #     Higher gamma = anchor adapts more slowly = more stable reference.
    # layer_names : list of str or None
    #     BN layer name fragments to hook. If None, hooks all BN layers in
    #     layer1 and layer2 (early layers, most sensitive to corruption).
    # warmup_batches : int
    #     Batches before novelty scores are used. Anchor populates but
    #     returns zeros during warmup.
    # eps : float
    #     Numerical stability constant.


    def __init__(self,
                 model          : nn.Module,
                 gamma          : float = 0.5,
                 layer_names    : list  = None,
                 warmup_batches : int   = 3,
                 eps            : float = 1e-8):
 
        self.gamma          = gamma
        self.warmup_batches = warmup_batches
        self.eps            = eps

        # BN statistics anchor - None until first batch (lazy init)
        self.mu             = None
        self.device         = None
        self._batch_count   = 0
        self._novelty_history = []
        self.last_novelty = 0.0

        # Storage for hooks: maps layer_name -> (batch_mean, batch_var)
        self._bn_stats      = OrderedDict()
        self._bn_per_sample = OrderedDict()  # layer_name -> (B, 2C) per-sample stats
        self._hooks         = []

        # Register forward hooks on selected BN layers
        self._register_hooks(model, layer_names)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self, model, layer_names):
        """
        Register forward hooks on BN layers to capture batch statistics.

        We target early-to-mid layers (layer1, layer2 in ResNet) because:
        - Early layers encode low-level statistics (edges, textures, noise)
          that are directly affected by corruption type.
        - Later layers encode semantic content (objects, parts) and are
          more corruption-invariant, which is unhelpful for shift detection.
        """
        if layer_names is None:
            # Default: hook BN layers in layer1 and layer2 only
            # These capture corruption signal before semantic abstraction.
            layer_names = ['layer1', 'layer2']
 
        for name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            # Only hook layers whose name contains one of our targets
            if not any(target in name for target in layer_names):
                continue

            # Closure to capture layer name for the hook callback
            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    # Using input[0] - PRE-normalisation activation (B, C, H, W).
                    # EATTA updates BN affine parameters (weight, bias) each batch
                    # which drives the OUTPUT back toward zero variance.
                    # The INPUT is the raw conv activation BEFORE BN corrects it
                    # so it retains the raw distributional signal from corruption.
                    with torch.no_grad():
                        x      = input[0].detach()
                        # Batch-level stats (for anchor update + batch novelty)
                        b_mean = x.mean(dim=[0, 2, 3])  # (C,)
                        b_var  = x.var(dim=[0, 2, 3])   # (C,)
                        # Per-sample stats (for per-sample novelty from BN inputs)
                        # Each sample's spatial mean/var per channel
                        s_mean = x.mean(dim=[2, 3])      # (B, C)
                        s_var  = x.var(dim=[2, 3])        # (B, C)
                    self._bn_stats[layer_name] = torch.cat([b_mean, b_var])
                    self._bn_per_sample[layer_name] = torch.cat([s_mean, s_var], dim=1)  # (B, 2C)
                return hook_fn

            handle = module.register_forward_hook(make_hook(name))
            self._hooks.append(handle)

        print(f"[BNAnchor] Registered hooks on {len(self._hooks)} BN layers "
              f"matching: {layer_names}")

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
# 
    def update_and_score(self):
        """
        Collect current BN statistics from hooks, update anchor, return
        batch-level novelty and per-sample novelty scores.

        Must be called AFTER the forward pass that populates hooks.

        Returns
        -------
        batch_novelty : scalar tensor - ||s_t - mu_t||^2 (0.0 during warmup)
        per_sample_novelty : (B,) tensor - per-sample distance from batch
                             centroid in BN input space (zeros during warmup)
        """
        if not self._bn_stats:
            self.last_novelty = 0.0
            self._novelty_history.append(0.0)
            return torch.tensor(0.0), None

        device = next(iter(self._bn_stats.values())).device

        # Concatenate all hooked BN statistics into a single vector s_t
        s_t = torch.cat(list(self._bn_stats.values()))  # (sum of 2C per layer,)

        # Per-sample BN stats: concatenate across layers -> (B, D_total)
        per_sample_stats = torch.cat(list(self._bn_per_sample.values()), dim=1)

        # Lazy initialisation
        if self.mu is None:
            self.mu = s_t.detach().clone()
            self._batch_count += 1
            self.last_novelty = 0.0
            self._novelty_history.append(0.0)
            return torch.tensor(0.0, device=device), torch.zeros(per_sample_stats.size(0), device=device)

        # ── Update anchor ─────────────────────────────────────────────────
        self.mu = self.gamma * self.mu + (1.0 - self.gamma) * s_t.detach()

        self._batch_count += 1

        # ── Batch novelty: squared L2 distance between current stats and anchor
        novelty = ((s_t.detach() - self.mu) ** 2).sum()

        # ── Per-sample novelty: distance from batch centroid in BN space ──
        # L2 normalise so distances reflect directional difference
        ps_norm = F.normalize(per_sample_stats, p=2, dim=1)
        centroid = ps_norm.mean(dim=0, keepdim=True)
        per_sample_nov = ((ps_norm - centroid) ** 2).sum(dim=1)  # (B,)

        # Diagnostic logging
        nov_val = novelty.item()
        self.last_novelty = nov_val
        self._novelty_history.append(nov_val)
        if self._batch_count % 10 == 0:
            recent_mean = sum(self._novelty_history[-10:]) / min(10, len(self._novelty_history))
            print(f"[BN-ANCHOR] batch={self._batch_count:>4}  |  "
                  f"novelty={nov_val:.4f}  |  "
                  f"10-batch avg={recent_mean:.4f}  |  "
                  f"stats_dim={s_t.shape[0]}")

        if self._batch_count <= self.warmup_batches:
            return torch.tensor(0.0, device=device), torch.zeros(per_sample_stats.size(0), device=device)

        return novelty, per_sample_nov

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
# 
    def reset(self):
        """Reset anchor state between stream runs."""
        self.mu             = None
        self._batch_count   = 0
        self._novelty_history = []
        self.last_novelty   = 0.0
        self._bn_stats      = OrderedDict()
        self._bn_per_sample = OrderedDict()

    def remove_hooks(self):
        """Remove all registered hooks. Call on cleanup."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ============================================================================
# EATTA - with OATTA filter + BN Statistics Anchor (CAS Direction 1)
# ============================================================================

@ADAPTATION_REGISTRY.register()
class EATTA(TTAMethod):
#     Effortless Active Test-Time Adaptation (EATTA) augmented with:

#     1. BNStatisticsAnchor - EMA anchor over BN layer statistics providing
#        the Novelty term for Context-Aware Selection (CAS).

#        Why BN statistics instead of feature vectors:
#        BN statistics (per-channel mean/variance) are sensitive to corruption
#        type but insensitive to class content. Mixed-class random streams
#        therefore produce systematic shifts in BN space at corruption
#        boundaries, even though feature-vector distances show only noise.

#     2. OATTAFilter - Recursive Bayesian filter with LLR gate providing
#        output-level temporal refinement (Surprise term).

#     CAS Selection Score
#     -------------------
#         S(x_i) = Perturbation(x_i) - Novelty_batch - Surprise(x_i)
# 
#     Novelty_batch is a scalar computed from BN statistics - it represents
#     how different the current batch's visual statistics are from the recent
#     environmental anchor. It is broadcast to all samples in the batch,
#     effectively up-weighting the entire batch for labeling when a corruption
#     boundary is detected.

#     Changes vs original EATTA
#      -------------------------
#     1.  BNStatisticsAnchor added in __init__, hooks registered on model[1].
#     2.  OATTAFilter added in __init__.
#     3.  loss_calculation: BN novelty scalar computed after forward pass,
#         multiplied into diff before sample selection.
#     4.  forward_and_adapt: returns (outputs, p_hat) instead of outputs.
#     5.  reset(): resets both anchor and OATTA filter state.
#     All weight update logic, buffer, grad debiasing: untouched.

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # ── Original EATTA hyper-parameters (unchanged) ───────────────────
        self.e_margin      = cfg.EATA.MARGIN_E0 * math.log(num_classes)
        self.cls_num_count = [0 for _ in range(num_classes)]
        self.cls_diff      = [0 for _ in range(num_classes)]
        self.oracle_num    = cfg.MODEL.ORACLE_NUM
        self.select_       = select_sample(self.oracle_num, self.device)
        self.noise_std     = 0.01
        self.w1_ema        = 0
        self.w2_ema        = 0
        self.mo            = 0.8
        self.softmax_entropy = Entropy()
        self.use_buffer    = cfg.MODEL.BUFFER
        self.samplebuffer  = SampleBuffer()
        self.buffer_bs     = 32
        self.annotator     = cfg.MODEL.HUMAN_OR_LARGE_MODEL
        arch_name          = cfg.MODEL.EDGE_ARCH
        self.featurizer, self.classifier = split_up_model(
            self.model[1], arch_name, self.dataset_name)
        for param in self.model[0].parameters():
            param.detach_()
        self.cfg = cfg

        cas_cfg = getattr(cfg, 'CAS', None)
        oatta_cfg = getattr(cfg, 'OATTA', None)
        self.cas_enabled = getattr(cas_cfg, 'ENABLED', True) if cas_cfg else True
        self.oatta_enabled = getattr(oatta_cfg, 'ENABLED', True) if oatta_cfg else True
        if self.cas_enabled and self.oatta_enabled:
            self.method_variant = "cas_oatta"
        elif self.cas_enabled:
            self.method_variant = "cas"
        else:
            self.method_variant = "eatta"
        self.last_batch_metrics = {}
# 
        # ── BN Statistics Anchor (CAS Novelty term) ───────────────────────
        # Hooks layer1 + layer2 BN layers in the edge model.
        # gamma=0.5: responsive anchor - effective memory window of 2 batches.
        # Responds within 2-3 batches of a shift while suppressing single-batch noise.
        # warmup=3: first 3 batches populate anchor without affecting selection.
        anchor_gamma  = getattr(cas_cfg, 'ANCHOR_GAMMA',  0.5) if cas_cfg else 0.5
        anchor_warmup = getattr(cas_cfg, 'ANCHOR_WARMUP',   3) if cas_cfg else 3
        anchor_layers = getattr(cas_cfg, 'ANCHOR_LAYERS', None) if cas_cfg else None
        anchor_layers = anchor_layers or None
# 
        self.bn_anchor = BNStatisticsAnchor(
            model          = self.model[1],
            gamma          = anchor_gamma,
            layer_names    = anchor_layers,   # None → defaults to layer1+layer2
            warmup_batches = anchor_warmup,
        )

        # ── OATTA filter (output-level temporal refinement) ───────────────
        self.oatta = OATTAFilter(
            num_classes = num_classes,
            kappa       = 1.0 / num_classes,
            gamma       = getattr(oatta_cfg, 'GAMMA',      0.1)  if oatta_cfg else 0.1,
            tau         = getattr(oatta_cfg, 'TAU',        1.0)  if oatta_cfg else 1.0,
            use_llr     = getattr(oatta_cfg, 'USE_LLR',   True)  if oatta_cfg else True,
            llr_window  = getattr(oatta_cfg, 'LLR_WINDOW',  10)  if oatta_cfg else 10,
            llr_margin  = getattr(oatta_cfg, 'LLR_MARGIN', 1.0)  if oatta_cfg else 1.0,
            llr_tau     = getattr(oatta_cfg, 'LLR_TAU',    1.0)  if oatta_cfg else 1.0,
            eta         = getattr(oatta_cfg, 'ETA',        0.1)  if oatta_cfg else 0.1,
        )

    # ------------------------------------------------------------------
    # loss_calculation - BN novelty injected into CAS selection score
    # ------------------------------------------------------------------
 
    def loss_calculation(self, x, y):
        imgs_test = x[0]
 
        # ── Forward pass - also populates bn_anchor._bn_stats via hooks ───
        features  = self.featurizer(imgs_test)
        outputs   = self.classifier(features)
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
 
        entropys = self.softmax_entropy(outputs)
        ids1     = torch.where(entropys < self.e_margin)[0]
 
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
 
        # ── EATTA perturbation score (unchanged) ──────────────────────────
        noise = torch.randn(features.size(), device=features.device) * self.noise_std
        fea   = features.clone().detach() + noise
        out   = self.classifier(fea)
        py2   = F.softmax(out, dim=-1)[:, y_prime]
        py2   = torch.diag(py2)
        diff  = torch.abs(py - py2)                               # (B,) perturbation score
 
        if self.cas_enabled:
            # ── BN anchor update + per-sample novelty from BN inputs ─────
            bn_novelty_tensor, bn_per_sample_nov = self.bn_anchor.update_and_score()
            bn_novelty = float(bn_novelty_tensor.detach().item())

            if bn_per_sample_nov is not None and bn_per_sample_nov.sum() > 0:
                per_sample_novelty = bn_per_sample_nov
            else:
                # Fallback during warmup: use feature-space distance
                features_flat = features.flatten(1).detach()
                features_norm = F.normalize(features_flat, p=2, dim=1)
                batch_mean_feat = features_norm.mean(dim=0, keepdim=True)
                per_sample_novelty = ((features_norm - batch_mean_feat) ** 2).sum(dim=1)

            # ── bn_scale: linear scaling with wider dynamic range ────────
            # Old: sigmoid compressed range to (1, 2) — only 33% difference
            # New: linear with clamp gives range (1, 5) — 5x difference
            #   Stable periods (bn_novelty ~ 0.07): scale ~ 1.7
            #   Transition spikes (bn_novelty ~ 0.8): scale ~ 5.0
            bn_threshold = 0.2  # tuned to typical stable-period Nt
            bn_scale = 1.0 + min(bn_novelty / bn_threshold, 4.0)

            novelty = per_sample_novelty * bn_scale

            # ── Cross-batch-aware normalisation ──────────────────────────
            # Old: per-batch min-max killed cross-batch signal entirely
            # New: EMA-tracked baseline provides cross-batch reference.
            # Within-batch relative ranking is preserved, but absolute scale
            # reflects whether this batch is genuinely novel vs stable.
            nov_min = novelty.min()
            nov_max = novelty.max()
            if (nov_max - nov_min) > 1e-6:
                # Normalise within batch to [0, 1]
                novelty_within = (novelty - nov_min) / (nov_max - nov_min + 1e-8)
            else:
                novelty_within = torch.ones_like(novelty)

            # Cross-batch boost: multiply within-batch scores by a factor
            # reflecting how novel THIS batch is relative to recent history.
            # EMA of batch-level novelty tracks the baseline.
            if not hasattr(self, '_novelty_ema'):
                self._novelty_ema = bn_novelty if bn_novelty > 0 else 0.01
            else:
                self._novelty_ema = 0.8 * self._novelty_ema + 0.2 * bn_novelty

            # Ratio of current batch novelty to recent average, clamped
            cross_batch_factor = min(bn_novelty / (self._novelty_ema + 1e-8), 3.0)
            cross_batch_factor = max(cross_batch_factor, 0.5)

            # Final novelty: within-batch ranking amplified by cross-batch signal
            novelty_norm = novelty_within * cross_batch_factor

            cas_score = diff * novelty_norm
        else:
            bn_novelty = 0.0
            novelty = torch.zeros_like(diff)
            novelty_norm = torch.ones_like(novelty)
            cas_score = diff

        eatta_selected_idx = torch.argmax(diff)
        cas_selected_idx = torch.argmax(cas_score)
# 
        # ── Sample selection using CAS score ──────────────────────────────
        sorted_indices = torch.argsort(cas_score, descending=True)
        sorted_idx, self.cls_num_count, self.cls_diff = self.select_(
            y_prime, sorted_indices, cas_score,
            self.cls_num_count, self.cls_diff)

        for i in sorted_idx:
            for j in ids1:
                if i == j:
                    mask = ids1 != i
                    ids1 = ids1[mask]
# 
        loss_ent = entropys[ids1].mean(0)
# 
        # ── Supervised loss (unchanged) ───────────────────────────────────
        if self.annotator == 'HUMAN':
            loss_ce = F.cross_entropy(outputs[sorted_idx], y[sorted_idx])
        elif self.annotator == 'LARGE_MODEL':
            with torch.no_grad():
                imgs_cloud    = imgs_test[sorted_idx]
                cloud_outputs = self.cloud_model(imgs_cloud)
                py_c, y_prime_c = F.softmax(cloud_outputs, dim=-1).max(1)
            loss_ce = F.cross_entropy(outputs[sorted_idx], y_prime_c.detach())
        else:
            raise NotImplementedError

        # ── Buffer (unchanged) ────────────────────────────────────────────
        if self.use_buffer:
            oracle_labels = y_prime_c if self.annotator == 'large_model' \
                            else y[sorted_idx]
            samples = copy.deepcopy(imgs_test)[sorted_idx]
            self.samplebuffer.add(samples, oracle_labels)
            buffer_dataset = Buffer(self.samplebuffer.buffer)
            buffer_loader  = DataLoader(buffer_dataset,
                                        batch_size=self.buffer_bs, shuffle=True)
            for imgs, labels in buffer_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                features_o   = self.featurizer(imgs).view(imgs.size(0), -1)
                outputs_o    = self.classifier(features_o)
                loss_buffer  = F.cross_entropy(outputs_o, labels)
            loss_ce = loss_ce + loss_buffer
# 
        # ── Gradient norm debiasing (unchanged) ───────────────────────────
        grad1 = torch.autograd.grad(
            loss_ent,
            list(p for p in self.model[1].parameters() if p.requires_grad),
            retain_graph=True)
        grad1_norm = torch.norm(torch.stack([g.norm() for g in grad1]))
# 
        grad2 = torch.autograd.grad(
            loss_ce,
            list(p for p in self.model[1].parameters() if p.requires_grad),
            retain_graph=True)
        grad2_norm = torch.norm(torch.stack([g.norm() for g in grad2]))
# 
        w1 = 2 * grad2_norm / (grad2_norm + grad1_norm)
        w2 = 2 * grad1_norm / (grad1_norm + grad2_norm)
        w1 = w1.detach().item()
        w2 = w2.detach().item()
# 
        self.w1_ema, self.w2_ema = update_w1_w2(
            w1, w2, self.w1_ema, self.w2_ema, self.mo)

        loss = self.w1_ema * loss_ent + self.w2_ema * loss_ce
        self.last_batch_metrics = {
            "cas_outputs": outputs.detach(),
            "novelty_nt": bn_novelty,
            "novelty_selected": novelty[cas_selected_idx].detach().item(),
            "novelty_batch_mean": novelty.detach().mean().item(),
            "cas_eatta_divergence": int(cas_selected_idx.item() != eatta_selected_idx.item()),
            "selected_sample_diff": diff[cas_selected_idx].detach().item(),
            "batch_mean_diff": diff.detach().mean().item(),
            "batch_entropy_mean": entropys.detach().mean().item(),
            "eatta_selected_idx": int(eatta_selected_idx.item()),
            "cas_selected_idx": int(cas_selected_idx.item()),
            "oracle_labels_used": int(sorted_idx.numel()),
        }
        return outputs, loss

    # ------------------------------------------------------------------
    # forward_and_adapt
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def forward_and_adapt(self, x, y):
#         Run one EATTA-family update.

#         Returns
#         -------
#         outputs : (B, K) raw logits from this run variant
#         p_hat   : (B, K) OATTA posteriors when OATTA is enabled
        outputs, loss = self.loss_calculation(x, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if not self.oatta_enabled:
            return outputs

        with torch.no_grad():
            p_hat = self.oatta.filter_batch(outputs)
        self.last_batch_metrics["llr_accumulator"] = float(self.oatta.get_llr())
        self.last_batch_metrics["lambda_values"] = float(self.oatta.last_mean_lambda)

        return outputs, p_hat

    # ------------------------------------------------------------------
    # Remaining methods - IDENTICAL to original EATTA
    # ------------------------------------------------------------------
# 
    def collect_params(self):
        params = []
        names  = []
        if self.cfg.CORRUPTION.DATASET in {'imagenet_c', 'imagenet100_c'}:
            for nm, m in self.model[1].named_modules():
                if 'layer4' in nm:
                    continue
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                  nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias'] and p.requires_grad:
                            params.append(p)
                            names.append(f"{nm}.{np}")
                if 'layer1' in nm:
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            p.requires_grad_(True)
                            params.append(p)
                            names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        self.model[1].eval()
        self.model[0].eval()
        self.model[1].requires_grad_(False)
        self.model[0].requires_grad_(False)
        for m in self.model[1].modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var  = None
            elif isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.w1_ema        = 0
        self.w2_ema        = 0
        self.cls_num_count = [0 for _ in range(self.num_classes)]
        self.cls_diff      = [0 for _ in range(self.num_classes)]
        self.bn_anchor.reset()
        self.oatta.reset()
        self.last_batch_metrics = {}
        if hasattr(self, '_novelty_ema'):
            del self._novelty_ema


# ============================================================================
# Helper classes - IDENTICAL to original EATTA
# ============================================================================

class select_sample(nn.Module):
    def __init__(self, oracle_num, device):
        self.last_cls   = []
        self.oracle_num = oracle_num
        self.device     = device

    def __call__(self, y_prime, sorted_indices, c_diff, cls_num_count, cls_diff):
        self.mask = torch.zeros(sorted_indices.size(0), dtype=torch.bool).to(self.device)
        if len(self.last_cls) == 0:
            can  = y_prime[sorted_indices][:self.oracle_num]
            diff = c_diff[sorted_indices][:self.oracle_num]
            for idx, i in enumerate(can):
                cls_num_count[i] += 1
                cls_diff[i]      = diff[idx]
                self.mask[idx]   = True
            sorted_idx = sorted_indices[self.mask]
            for i in y_prime[sorted_idx]:
                self.last_cls.append(i.item())
        else:
            can  = y_prime[sorted_indices][:self.oracle_num]
            diff = c_diff[sorted_indices][:self.oracle_num]
            for idx, i in enumerate(can):
                if i.item() not in self.last_cls:
                    cls_num_count[i.item()] += 1
                    cls_diff[i.item()]       = diff[idx]
                    self.mask[idx]           = True
                else:
                    if diff[idx] < cls_diff[i]:
                        can2  = y_prime[sorted_indices][self.oracle_num:]
                        diff2 = c_diff[sorted_indices][self.oracle_num:]
                        for idx_, j in enumerate(can2):
                            if j not in self.last_cls:
                                cls_num_count[j] += 1
                                cls_diff[j]       = diff2[idx_]
                                self.mask[self.oracle_num + idx_] = True
                                break
                    else:
                        cls_num_count[i] += 1
                        cls_diff[i]       = diff[idx]
                        self.mask[idx]    = True
            sorted_idx = sorted_indices[self.mask]
            if len(sorted_idx) >= 1:
                for i in y_prime[sorted_idx]:
                    self.last_cls.append(i.item())
            else:
                sorted_idx = sorted_indices[:self.oracle_num]
            if len(self.last_cls) > 2:
                self.last_cls = self.last_cls[len(self.last_cls) - 1:]
        return sorted_idx, cls_num_count, cls_diff


class SampleBuffer:
    def __init__(self, capacity=300):
        self.capacity = capacity
        self.buffer   = []
# 
    def add(self, samples, labels):
        new_entries = list(zip(samples, labels))
        if len(self.buffer) + len(new_entries) > self.capacity:
            excess      = len(self.buffer) + len(new_entries) - self.capacity
            self.buffer = self.buffer[excess:]
        self.buffer.extend(new_entries)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class Buffer(Dataset):
    def __init__(self, data):
        self.data = data
# 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        return sample, label


@torch.no_grad()
def update_w1_w2(w1, w2, w1_ema, w2_ema, momentum=0.8):
    if w1_ema == 0 and w2_ema == 0:
        return w1, w2
    w1_ema = momentum * w1_ema + (1 - momentum) * w1
    w2_ema = momentum * w2_ema + (1 - momentum) * w2
    return w1_ema, w2_ema