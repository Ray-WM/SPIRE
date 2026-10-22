"""
SPIRE on TSB-AD: phase-conditioned spiking dynamics for time-series anomaly detection.

One causal spiking backbone (SRP residual encoder + spiking TCN + SPF) is trained on the
normal prefix of each series with a causal reconstruction loss. The same trained model is
then scored with one of three evidence sources:

  recon   (E1)  output error      S(t) = ||x(t) - y(t)||^2
  spike   (E2)  internal state    phase-conditioned robust deviation of five internal-state
                                  statistics (residual energy, encoder rate, backbone activity,
                                  burst ratio, silence ratio) from per-phase normal templates
  hybrid  (E3)  late fusion       lambda_spk * z(S_spike) + lambda_rec * z(S_recon) with fixed
                                  median / 1.4826*MAD scalers fitted on the normal training slice

Controls (same residual, architecture, feature operators, templates, fusion and protocol;
only the unit differs):
  --model ann                  matched ReLU network scored with activation templates
  --binary_lif --rate_match    stateless binary unit at every LIF site, threshold calibrated
                               to the firing rate of a LIF reference trained on the same data

Protocol options:
  --train_prefix official      normal prefix from the tr_ field of the file name (TSB-AD protocol)
  --vus_window auto            per-file VUS window from find_length_rank (TSB-AD protocol)
  --save_streams               per-file score streams to out_dir/streams/*.npz (input of
                               analysis/offline_scores.py and analysis/stream_metrics.py)
  --seed N                     global RNG seed for repeated training runs

Examples:
  python run_spire.py --data_dir data/eval_u --subset TSB-AD-U --train_prefix official \
      --score_mode hybrid --save_streams --out_dir results/u_hybrid <hyper-parameters>
  python run_spire.py --data_dir data/eval_m --subset TSB-AD-M --train_prefix official \
      --score_mode recon --out_dir results/m_recon <hyper-parameters>
"""

import os, re, sys, glob, argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class TSBADWindowDataset(Dataset):
    def __init__(self, data, window_size=100, stride=1):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.stride = stride
        self.num_windows = max(0, (len(data) - window_size) // stride + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return torch.from_numpy(self.data[start:start + self.window_size]), start


# ============================================================
# Network builder
# ============================================================

def _measure_lif_rates(network, train_data, window_size, batch_size, device):
    """Return {module_name: mean firing rate} for each LIF-like module."""
    from spire.controls import is_lif_like
    rates, hooks = {}, []
    def mk(name):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(o, torch.Tensor):
                rates.setdefault(name, []).append(o.float().mean().item())
        return hook
    for name, mod in network.named_modules():
        if is_lif_like(mod):
            hooks.append(mod.register_forward_hook(mk(name)))
    network.eval()
    ds = TSBADWindowDataset(train_data, window_size, stride=max(1, window_size // 2))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for windows, _ in loader:
            network(windows.to(device), return_internal=False)
    for h in hooks:
        h.remove()
    return {n: float(np.mean(v)) for n, v in rates.items()}


def _calibrate_binary_thresholds(network, lif_rates, train_data, window_size, batch_size, device):
    """Set each RateMatchedBinaryAct target rate from the matching LIF module,
    then calibrate its threshold on collected training-data inputs."""
    from spire.controls import RateMatchedBinaryAct
    default_rate = float(np.mean(list(lif_rates.values()))) if lif_rates else 0.1
    samples, hooks = {}, []
    def mk(name):
        def pre_hook(_m, inputs):
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                flat = x.detach().reshape(-1)
                if flat.numel() > 200_000:
                    idx = torch.randint(0, flat.numel(), (200_000,), device=flat.device)
                    flat = flat[idx]
                samples.setdefault(name, []).append(flat.cpu())
        return pre_hook
    for name, mod in network.named_modules():
        if isinstance(mod, RateMatchedBinaryAct):
            mod.target_rate = float(lif_rates.get(name, default_rate))
            hooks.append(mod.register_forward_pre_hook(mk(name)))
    network.eval()
    ds = TSBADWindowDataset(train_data, window_size, stride=max(1, window_size // 2))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for windows, _ in loader:
            network(windows.to(device), return_internal=False)
    for h in hooks:
        h.remove()
    modules = dict(network.named_modules())
    for name, chunks in samples.items():
        modules[name].calibrate(torch.cat(chunks))
    n_cal = len(samples)
    matched = sum(1 for n in samples if n in lif_rates)
    print(f"  [rate_match] calibrated {n_cal} binary modules "
          f"({matched} matched to LIF rate, mean target {default_rate:.4f})")


def build_network_astar(input_size, window_size, cfg, binary_lif=False, model="snn"):
    cfg = dict(cfg)
    if model == "ann":
        # Matched ReLU control: the same SRP/SPF/TCN topology with continuous
        # activations, scored with activation templates.
        from spire.backbone_ann import ANNTCN_SRPSPF
        return ANNTCN_SRPSPF(input_size=input_size, max_length=window_size, **cfg)
    from spire.backbone import SpikeTCN_SRPSPF_AD_Star
    network = SpikeTCN_SRPSPF_AD_Star(
        input_size=input_size, max_length=window_size,
        causal=True,  # SPIRE always uses the causal backbone
        **cfg,
    )
    if binary_lif:
        # Stateless binary control: every LIF neuron is replaced by a binary
        # threshold unit with a straight-through estimator (no leak, no
        # integration, no reset).
        from spire.controls import (
            replace_lif_with_binary, count_lif_like_modules, count_binary_modules)
        n_before = count_lif_like_modules(network)
        network = replace_lif_with_binary(network)
        print(f"  [binary_lif] replaced {n_before} LIF-like modules "
              f"({count_binary_modules(network)} BinaryAct now)")
    return network


# ============================================================
# Training (causal reconstruction)
# ============================================================

def train_model(
    network, train_data,
    window_size=100, batch_size=64,
    lr=1e-3, max_epochs=50, patience=10, train_stride=5,
    device="cuda",
):
    """
    Train with causal reconstruction loss (auxiliary).
    loss = MSE(y[t], x[t])
    """
    network = network.to(device)
    network.train()
    dataset = TSBADWindowDataset(train_data, window_size, stride=train_stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_loss, best_state, no_improve = float('inf'), None, 0

    for epoch in range(max_epochs):
        total_loss, n_batches = 0, 0
        for windows, _ in loader:
            windows = windows.to(device)
            y_seq, _ = network(windows, return_internal=False)  # training: no need for internals
            output = y_seq.permute(0, 2, 1)
            loss = loss_fn(output, windows)  # causal reconstruction

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}, best loss: {best_loss:.6f}")
            break

    if best_state is not None:
        network.load_state_dict(best_state)
    return network


# ============================================================
# Phase-conditioned template of the internal state (core of SPIRE)
# ============================================================

class SpikeTemplate:
    """
    Build phase-conditioned normal templates from training data.

    For each detected period p_i and each phase φ = t mod p_i,
    stores the median and MAD of the spike feature vector u(t)
    computed over the training (normal) portion.

    Feature vector u(t):
      [0] |R(t)|_1      - SRP residual energy (mean over D)
      [1] ρ_enc(t)       - encoder firing rate (mean over D)
      [2] ρ_back(t)      - backbone firing rate (mean over D)
      [3] B(t)           - burst ratio: fraction of units with spike_count >= 2 in Ts steps
      [4] Q(t)           - silence ratio: fraction of units with spike_count == 0 in Ts steps
    """

    N_FEATURES = 5

    def __init__(self):
        self.templates = {}  # period -> {phase -> (median[5], mad[5])}
        self.periods = []
        self.weights = None
        self.r = 0.0
        self.enabled = False
        self.global_stats = None  # (median[5], mad[5]) over all train steps

    def fit(self, network, train_data, window_size, batch_size=128, device="cuda"):
        """
        Run inference on training data, collect spike features,
        build phase-conditioned templates.
        """
        network.eval()
        T_train, C = train_data.shape
        dataset = TSBADWindowDataset(train_data, window_size, stride=1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Collect features and router info
        all_features = np.zeros((T_train, self.N_FEATURES), dtype=np.float64)
        feature_count = np.zeros(T_train, dtype=np.float64)
        collected_periods = []
        collected_weights = []
        collected_r = []

        with torch.no_grad():
            for windows, starts in loader:
                windows = windows.to(device)
                _, _, internals = network(windows, return_internal=True)

                # Extract features for each window
                feats = self.extract_features(internals)  # [B, ws, 5]
                feats_np = feats.cpu().numpy()
                starts_np = starts.numpy()

                for i in range(len(starts_np)):
                    s = starts_np[i]
                    all_features[s:s + window_size] += feats_np[i]
                    feature_count[s:s + window_size] += 1

                # Collect router info
                periods, weights, r, enabled = internals['router']
                if enabled and len(periods) > 0:
                    collected_periods.append(periods)
                    collected_weights.append(weights)
                    collected_r.append(r)

        # Average features
        feature_count = np.maximum(feature_count, 1)
        all_features /= feature_count[:, None]

        # Global (non-phase) statistics: fallback for missing/degenerate
        # phases, and the p=1 template when no reliable period is detected.
        g_median = np.median(all_features, axis=0)
        g_mad = np.median(np.abs(all_features - g_median), axis=0)
        self.global_stats = (g_median, np.maximum(g_mad, 1e-8))

        # Determine dominant periods from training
        if collected_periods:
            # Use the most frequent period set
            from collections import Counter
            period_strs = [str(sorted(p)) for p in collected_periods]
            most_common = Counter(period_strs).most_common(1)[0][0]
            idx = period_strs.index(most_common)
            self.periods = collected_periods[idx]
            self.weights = collected_weights[idx]
            self.r = float(np.mean(collected_r))
            self.enabled = True
        else:
            # No reliable period: fit a single global template (p = 1) instead
            # of returning all-zero spike scores on weakly periodic series.
            self.periods = [1]
            self.weights = np.array([1.0], dtype=np.float64)
            self.r = 0.0
            self.enabled = True

        # Build per-phase templates
        if self.enabled and len(self.periods) > 0:
            for p in self.periods:
                p = int(p)
                phase_features = {}
                for phi in range(p):
                    # Gather all time steps with this phase
                    indices = np.arange(phi, T_train, p)
                    if len(indices) < 3:
                        continue
                    phase_data = all_features[indices]  # [n_samples, 5]
                    median = np.median(phase_data, axis=0)
                    mad = np.median(np.abs(phase_data - median), axis=0)
                    phase_features[phi] = (median, mad)
                self.templates[p] = phase_features

        print(f"  SpikeTemplate: periods={self.periods}, r={self.r:.3f}, "
              f"enabled={self.enabled}, "
              f"template_phases={sum(len(v) for v in self.templates.values())}")

    def score(self, features, t_global):
        """
        Compute spike-native anomaly score for a single time step.

        Args:
            features: [5] feature vector for this time step
            t_global: global time index (for computing phase)
        Returns:
            float: spike anomaly score
        """
        if not self.enabled or not self.templates:
            return 0.0

        g_median, g_mad = self._global_or_default()

        total_score = 0.0
        for i, p in enumerate(self.periods):
            p = int(p)
            if p not in self.templates:
                continue
            phi = int(t_global) % p
            stats = self.templates[p].get(phi)
            if stats is None:
                median, mad = g_median, g_mad  # global fallback for missing phase
            else:
                median = stats[0]
                mad = np.where(stats[1] < 1e-8, g_mad, stats[1])
            # Robust z-score: |u - median| / (mad + eps)
            deviation = np.abs(features - median) / (mad + 1e-8)
            # Weight by period importance
            w = float(self.weights[i]) if self.weights is not None else 1.0 / len(self.periods)
            total_score += w * deviation.mean()

        return total_score

    def _global_or_default(self):
        """Global (median, mad) stats, defaulting to (0, 1) if never fitted."""
        if self.global_stats is not None:
            return self.global_stats
        return (np.zeros(self.N_FEATURES, dtype=np.float64),
                np.ones(self.N_FEATURES, dtype=np.float64))

    def score_batch(self, features_array, start_positions, window_size,
                    return_per_feature=False):
        """
        Vectorized scoring for a batch of windows.

        Args:
            features_array: [B, ws, 5]
            start_positions: [B]
            window_size: int
            return_per_feature: if True, return [B, ws, 5] period-weighted
                per-feature deviations; the scalar score equals their mean
                over the last axis (used for the u_t component ablation).
        Returns:
            scores: [B, ws]  (or [B, ws, 5] with return_per_feature=True)
        """
        B = features_array.shape[0]
        per_feature = np.zeros((B, window_size, self.N_FEATURES), dtype=np.float64)

        if not self.enabled or not self.templates:
            return per_feature if return_per_feature else per_feature.mean(axis=-1)

        features_array = np.asarray(features_array, dtype=np.float64)
        g_median, g_mad = self._global_or_default()
        starts = np.asarray(start_positions, dtype=np.int64).reshape(-1)
        offsets = np.arange(window_size, dtype=np.int64)

        for i, p in enumerate(self.periods):
            p = int(p)
            if p not in self.templates:
                continue
            w = float(self.weights[i]) if self.weights is not None else 1.0 / len(self.periods)

            # Build median/mad arrays for all phases of this period.
            # Missing phases and zero-MAD features fall back to global stats
            median_arr = np.tile(g_median, (p, 1))
            mad_arr = np.tile(g_mad, (p, 1))
            for phi in range(p):
                stats = self.templates[p].get(phi)
                if stats is not None:
                    median_arr[phi] = stats[0]
                    mad_arr[phi] = np.where(stats[1] < 1e-8, g_mad, stats[1])

            phases = (starts[:, None] + offsets[None, :]) % p           # [B, ws]
            dev = np.abs(features_array - median_arr[phases]) / (mad_arr[phases] + 1e-8)
            per_feature += w * dev

        return per_feature if return_per_feature else per_feature.mean(axis=-1)

    @staticmethod
    def _extract_features_ann(internals: dict, v_threshold: float = 1.0) -> torch.Tensor:
        """
        Matched 5-dim statistics from the ReLU network's internals (the
        activation-template control): the same statistical operators as the
        LIF feature vector, computed on continuous activations.

        [0] |R| mean          (same as SNN)
        [1] enc activation mean            ~ encoder firing rate
        [2] backbone activity mean         ~ backbone activity mean
        [3] high-activation proportion (>= v_threshold)   ~ burst ratio
        [4] zero-activation proportion (ReLU exact zeros) ~ silence ratio

        Returns: [B, T, 5]
        """
        srp_res = internals["srp_residual"]        # [B,T,H]
        enc_act = internals["enc_act"]             # [B,T,H]
        back_act = internals["back_act"]           # [B,T,ch]

        return torch.stack([
            srp_res.abs().mean(dim=-1),
            enc_act.mean(dim=-1),
            back_act.mean(dim=-1),
            (enc_act >= v_threshold).float().mean(dim=-1),
            (enc_act <= 1e-6).float().mean(dim=-1),
        ], dim=-1)

    @staticmethod
    def extract_features(internals: dict) -> torch.Tensor:
        """Dispatch: SNN spike internals or ANN activation internals -> [B,T,5]."""
        if "enc_act" in internals:
            return SpikeTemplate._extract_features_ann(internals)
        return SpikeTemplate._extract_features(internals)

    @staticmethod
    def _extract_features(internals: dict) -> torch.Tensor:
        """
        Extract the 5-dim spike feature vector from network internals.

        Returns: [B, T, 5]
        """
        srp_res = internals['srp_residual']       # [B, T, D]
        enc_spikes = internals['enc_spikes']       # [B, Ts, T, D]
        back_spikes = internals['back_spikes']     # [B, Ts, T, D_ch]
        enc_rate = internals['enc_rate']            # [B, T, D]
        back_rate = internals['back_rate']          # [B, T, D_ch]

        B, T, D = srp_res.shape
        Ts = enc_spikes.shape[1]

        # [0] SRP residual energy: mean |R(t)| over D
        srp_energy = srp_res.abs().mean(dim=-1)                    # [B, T]

        # [1] Encoder firing rate: mean over D
        enc_rate_scalar = enc_rate.mean(dim=-1)                    # [B, T]

        # [2] Backbone firing rate: mean over D_ch
        back_rate_scalar = back_rate.mean(dim=-1)                  # [B, T]

        # [3] Burst ratio: fraction of units that spike >= 2 times in Ts steps
        enc_spike_counts = enc_spikes.sum(dim=1)                   # [B, T, D]
        burst_ratio = (enc_spike_counts >= 2).float().mean(dim=-1) # [B, T]

        # [4] Silence ratio: fraction of units that spike 0 times
        silence_ratio = (enc_spike_counts == 0).float().mean(dim=-1)  # [B, T]

        features = torch.stack([
            srp_energy, enc_rate_scalar, back_rate_scalar,
            burst_ratio, silence_ratio
        ], dim=-1)  # [B, T, 5]

        return features


# ============================================================
# Fixed hybrid-score scalers
# ============================================================

class HybridScalers:
    """
    Fixed robust scalers for hybrid late fusion.

    Each stream s is normalized as z = (s - median) / (1.4826 * MAD + eps),
    with median/MAD fitted once on the normal training slice. Fixed
    normalization keeps a test point's score independent of the other points
    in the same evaluation batch (unlike the old per-batch min-max).
    """

    MAD_SCALE = 1.4826

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.spk_center, self.spk_scale = 0.0, 1.0
        self.rec_center, self.rec_scale = 0.0, 1.0
        self.fitted = False

    @classmethod
    def _center_scale(cls, stream, eps):
        stream = np.asarray(stream, dtype=np.float64).ravel()
        med = float(np.median(stream))
        mad = float(np.median(np.abs(stream - med)))
        return med, cls.MAD_SCALE * mad + eps

    def fit(self, spike_stream, recon_stream):
        """Fit both scalers on per-timestep score streams from normal data."""
        self.spk_center, self.spk_scale = self._center_scale(spike_stream, self.eps)
        self.rec_center, self.rec_scale = self._center_scale(recon_stream, self.eps)
        self.fitted = True
        return self

    def fuse(self, spike_stream, recon_stream, lambda_spk=0.7, lambda_rec=0.3):
        """Late fusion of z-scored streams: λ_spk * z_spk + λ_rec * z_rec."""
        assert self.fitted, "HybridScalers.fit must be called before fuse"
        z_spk = (np.asarray(spike_stream, dtype=np.float64) - self.spk_center) / self.spk_scale
        z_rec = (np.asarray(recon_stream, dtype=np.float64) - self.rec_center) / self.rec_scale
        return lambda_spk * z_spk + lambda_rec * z_rec


# ============================================================
# Point-level scoring
# ============================================================

def get_point_scores_astar(
    network, data, template: SpikeTemplate,
    window_size=100, batch_size=128,
    score_mode="hybrid", lambda_spk=0.7, lambda_rec=0.3,
    hybrid_norm="fixed", train_end=None,
    return_streams=False,
    device="cuda",
):
    """
    Score every time step.

    score_mode:
      'recon':  S(t) = MSE reconstruction error only (E1, output error)
      'spike':  S(t) = phase-conditioned internal-state deviation only (E2)
      'hybrid': S(t) = lambda_spk * z(S_spike) + lambda_rec * z(S_recon) (E3)

    hybrid_norm (hybrid mode only):
      'fixed':  z(·) via median/1.4826·MAD scalers fitted on scores[:train_end]
                (the normal training slice); the protocol of the paper.
      'minmax': per-batch min-max normalization. Batch-dependent; kept only
                as a reference.
    """
    network.eval()
    T, C = data.shape
    dataset = TSBADWindowDataset(data, window_size, stride=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    need_spike = score_mode in ("spike", "hybrid")
    legacy_minmax = (score_mode == "hybrid" and hybrid_norm == "minmax")

    recon_sum = np.zeros(T, dtype=np.float64)
    spike_sum = np.zeros(T, dtype=np.float64)
    legacy_sum = np.zeros(T, dtype=np.float64)
    featdev_sum = np.zeros((T, SpikeTemplate.N_FEATURES), dtype=np.float64)
    score_count = np.zeros(T, dtype=np.float64)

    with torch.no_grad():
        for windows, starts in loader:
            windows = windows.to(device)

            if need_spike:
                y_seq, _, internals = network(windows, return_internal=True)
            else:
                y_seq, _ = network(windows, return_internal=False)
            output = y_seq.permute(0, 2, 1)
            starts_np = starts.numpy()

            recon_errors = (windows - output).pow(2).mean(dim=-1).cpu().numpy()  # [B, ws]

            spike_scores = None
            feature_devs = None
            if need_spike:
                features = SpikeTemplate.extract_features(internals)
                if return_streams:
                    feature_devs = template.score_batch(
                        features.cpu().numpy(), starts_np, window_size,
                        return_per_feature=True)          # [B, ws, 5]
                    spike_scores = feature_devs.mean(axis=-1)
                else:
                    spike_scores = template.score_batch(
                        features.cpu().numpy(), starts_np, window_size)  # [B, ws]

            combined = None
            if legacy_minmax:
                # Per-batch min-max fusion (batch-dependent; reference only).
                def _norm(arr):
                    mn, mx = arr.min(), arr.max()
                    if mx - mn < 1e-12:
                        return np.zeros_like(arr)
                    return (arr - mn) / (mx - mn)

                combined = lambda_spk * _norm(spike_scores) + lambda_rec * _norm(recon_errors)

            for i in range(len(starts_np)):
                sl = slice(starts_np[i], starts_np[i] + window_size)
                recon_sum[sl] += recon_errors[i]
                if need_spike:
                    spike_sum[sl] += spike_scores[i]
                if feature_devs is not None:
                    featdev_sum[sl] += feature_devs[i]
                if legacy_minmax:
                    legacy_sum[sl] += combined[i]
                score_count[sl] += 1

    def _avg(sum_arr):
        return np.divide(
            sum_arr, score_count,
            out=np.zeros_like(sum_arr),
            where=(score_count > 0),
        )

    recon_stream = _avg(recon_sum)
    spike_stream = _avg(spike_sum) if need_spike else None

    if score_mode == "recon":
        scores = recon_stream
    elif score_mode == "spike":
        scores = spike_stream
    elif legacy_minmax:
        scores = _avg(legacy_sum)
    else:
        # Fixed fusion: fit robust scalers on the normal training slice of the
        # aggregated streams, then fuse pointwise. Because the z-transform is
        # affine, fusing after window-averaging equals averaging fused scores.
        if train_end is None:
            train_end = T // 2
            print(f"  [WARN] hybrid fixed norm: train_end not given, using T//2={train_end}")
        fit_end = min(max(int(train_end), 2), T)
        scalers = HybridScalers().fit(spike_stream[:fit_end], recon_stream[:fit_end])
        scores = scalers.fuse(spike_stream, recon_stream, lambda_spk, lambda_rec)

    # Fill uncovered
    uncovered = (score_count == 0)
    if uncovered.any():
        valid_idx = np.where(~uncovered)[0]
        if len(valid_idx) > 0:
            for idx in np.where(uncovered)[0]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                scores[idx] = scores[nearest]

    if return_streams:
        streams = {
            "recon_stream": recon_stream.astype(np.float32),
            "spike_stream": (spike_stream.astype(np.float32)
                             if spike_stream is not None else None),
            "feature_dev_stream": (
                np.divide(featdev_sum, score_count[:, None],
                          out=np.zeros_like(featdev_sum),
                          where=(score_count[:, None] > 0)).astype(np.float32)
                if need_spike else None),
        }
        return scores.astype(np.float32), streams
    # Map non-finite scores to large finite values (ordering preserved for VUS/AUC).
    return np.nan_to_num(scores, nan=0.0, posinf=1e30, neginf=-1e30).astype(np.float32)


# ============================================================
# Main interface
# ============================================================

def run_SpikeAD_AStar(
    data: np.ndarray,
    # Score
    score_mode: str = "hybrid",
    lambda_spk: float = 0.7,
    lambda_rec: float = 0.3,
    hybrid_norm: str = "fixed",
    # Model
    window_size: int = 100,
    hidden_size: int = 64,
    channel: int = 32,
    num_levels: int = 2,
    kernel_size: int = 8,
    num_steps: int = 4,
    lif_tau: float = 2.5,
    dropout: float = 0.2,
    # SRP/SPF
    use_srp: bool = True,
    use_spf: bool = True,
    alpha: float = 0.5,
    top_k: int = 2,
    r_min: float = 0.1,
    spf_at: int = 0,
    # Training
    train_ratio: float = 0.5,
    train_end: int = None,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 10,
    train_stride: int = 5,
    return_streams: bool = False,
    binary_lif: bool = False,
    rate_match: bool = False,
    model: str = "snn",
    device: str = None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T, C = data.shape

    # Normalize. train_end overrides train_ratio when given (official TSB-AD
    # protocol: the normal prefix length comes from the file name's tr_ field).
    if train_end is None:
        train_end = int(T * train_ratio)
    train_end = max(min(int(train_end), T - 1), window_size + 1)
    train_raw = data[:train_end]
    mean = np.nanmean(train_raw, axis=0, keepdims=True)
    std = np.nanstd(train_raw, axis=0, keepdims=True) + 1e-8
    data_norm = np.nan_to_num(((data - mean) / std).astype(np.float32))
    train_data = data_norm[:train_end]

    # Build network
    network_cfg = dict(
        kernel_size=kernel_size, num_levels=num_levels,
        channel=channel, hidden_size=hidden_size,
        num_steps=num_steps, lif_tau=lif_tau, dropout=dropout,
        use_srp=use_srp, use_spf=use_spf,
        alpha=alpha, top_k=top_k, r_min=r_min,
        spf_at=min(spf_at, num_levels - 1),
    )
    if binary_lif and rate_match:
        # Rate-matched binary control: train a LIF reference to measure per-module
        # firing rates, then build/train/calibrate a binary model at matched rate.
        from spire.controls import replace_lif_with_binary, count_binary_modules
        lif_ref = build_network_astar(C, window_size, network_cfg,
                                      binary_lif=False, model=model).to(device)
        lif_ref = train_model(lif_ref, train_data, window_size=window_size,
                              batch_size=batch_size, lr=lr, max_epochs=max_epochs,
                              patience=patience, train_stride=train_stride, device=device)
        lif_rates = _measure_lif_rates(lif_ref, train_data, window_size,
                                       batch_size * 2, device)
        del lif_ref
        network = build_network_astar(C, window_size, network_cfg,
                                      binary_lif=False, model=model)
        network = replace_lif_with_binary(network, rate_matched=True,
                                          target_rate=0.1).to(device)
        print(f"  [rate_match] {count_binary_modules(network)} rate-matched "
              f"binary modules; LIF global rate "
              f"{np.mean(list(lif_rates.values())) if lif_rates else 0:.4f}")
        network = train_model(network, train_data, window_size=window_size,
                              batch_size=batch_size, lr=lr, max_epochs=max_epochs,
                              patience=patience, train_stride=train_stride, device=device)
        _calibrate_binary_thresholds(network, lif_rates, train_data, window_size,
                                     batch_size * 2, device)
    else:
        network = build_network_astar(C, window_size, network_cfg,
                                      binary_lif=binary_lif, model=model)
        # Step 1: Train with causal reconstruction (auxiliary)
        network = train_model(
            network, train_data,
            window_size=window_size,
            batch_size=batch_size, lr=lr,
            max_epochs=max_epochs, patience=patience,
            train_stride=train_stride,
            device=device,
        )

    # Step 2: Build phase-conditioned spike template from training data
    template = SpikeTemplate()
    if score_mode in ("spike", "hybrid"):
        template.fit(
            network, train_data,
            window_size=window_size,
            batch_size=batch_size * 2,
            device=device,
        )

    # Step 3: Score
    result = get_point_scores_astar(
        network, data_norm, template,
        window_size=window_size,
        batch_size=batch_size * 2,
        score_mode=score_mode,
        lambda_spk=lambda_spk,
        lambda_rec=lambda_rec,
        hybrid_norm=hybrid_norm,
        train_end=train_end,
        return_streams=return_streams,
        device=device,
    )

    if return_streams:
        scores, streams = result
        streams["train_end"] = train_end
        streams["router_r"] = float(template.r)
        streams["router_periods"] = np.asarray(
            [int(p) for p in template.periods], dtype=np.int64)
        return scores, streams
    return result


# ============================================================
# Batch evaluation
# ============================================================

SCORE_LABELS = {
    "recon": "E1: output error (causal reconstruction)",
    "spike": "E2: internal-state template",
    "hybrid": "E3: hybrid (fixed-scaler late fusion)",
}


def evaluate_tsbad(data_dir, out_dir, subset="TSB-AD-U", max_files=None, **kw):
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if max_files:
        csv_files = csv_files[:max_files]

    # 'auto' = official TSB-AD protocol (per-file ACF sliding window); an integer = fixed window.
    vus_window = kw.pop('vus_window', 'auto')

    # Dump per-file score streams (recon/spike) + label to out_dir/streams/,
    # so fusion variants can be evaluated offline without re-training.
    save_streams = kw.pop('save_streams', False)
    if save_streams:
        os.makedirs(os.path.join(out_dir, "streams"), exist_ok=True)

    # 'ratio' = train_ratio prefix (default 0.5);
    # 'official' = per-file tr_ field from the file name (TSB-AD protocol).
    train_prefix = kw.pop('train_prefix', 'ratio')

    score_mode = kw.get('score_mode', 'hybrid')
    mode_str = SCORE_LABELS.get(score_mode, score_mode)
    if score_mode == "hybrid":
        mode_str += f" [norm={kw.get('hybrid_norm', 'fixed')}]"
    print(f"SPIRE score mode: {mode_str}  [vus_window={vus_window}]")

    all_results = []
    for i, csv_path in enumerate(csv_files):
        fname = os.path.basename(csv_path)
        print(f"\n[{i+1}/{len(csv_files)}] {fname}")
        try:
            df = pd.read_csv(csv_path).dropna()
            data = df.iloc[:, :-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            T, C = data.shape
            ws = kw.get('window_size', 100)
            if T < ws * 2:
                print(f"  SKIP (T={T})")
                continue
            if str(vus_window) == "auto":
                sliding_window = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
            else:
                sliding_window = int(vus_window)
            per_file_kw = dict(kw)
            if train_prefix == "official":
                m = re.search(r"_tr_(\d+)_", fname)
                if m is None:
                    print(f"  SKIP: no tr_ field in name for official prefix")
                    continue
                per_file_kw["train_end"] = int(m.group(1))
            t0 = time.time()
            if save_streams:
                scores, streams = run_SpikeAD_AStar(data, return_streams=True, **per_file_kw)
                np.savez_compressed(
                    os.path.join(out_dir, "streams", fname.replace(".csv", ".npz")),
                    scores=scores, label=label,
                    sliding_window=np.int64(sliding_window),
                    **{k: v for k, v in streams.items() if v is not None},
                )
            else:
                scores = run_SpikeAD_AStar(data, **per_file_kw)
            elapsed = time.time() - t0
            # non-finite scores are mapped to large finite values (ordering preserved)
            scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=1e30, neginf=-1e30).astype(np.float32)
            metrics = get_metrics(scores, label, slidingWindow=sliding_window)
            metrics['file'] = fname
            metrics['time'] = elapsed
            metrics['sliding_window'] = sliding_window
            all_results.append(metrics)
            print(f"  VUS-PR={metrics.get('VUS-PR','?'):.4f}  "
                  f"AUC-ROC={metrics.get('AUC-ROC','?'):.4f}  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        rdf = pd.DataFrame(all_results)
        rpath = os.path.join(out_dir, f"results_{subset}.csv")
        rdf.to_csv(rpath, index=False)
        print(f"\n{'='*60}")
        print(f"Results ({len(rdf)} files) - {SCORE_LABELS.get(score_mode, score_mode)}")
        for c in ['VUS-PR','AUC-ROC','AUC-PR','Standard-F1']:
            if c in rdf:
                print(f"  {c}: {rdf[c].mean():.4f}")
        print(f"  Saved: {rpath}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPIRE on TSB-AD")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/spire")
    parser.add_argument("--subset", type=str, default="TSB-AD-U")
    parser.add_argument("--max_files", type=int, default=None)

    # Score mode
    parser.add_argument("--score_mode", type=str, default="hybrid",
                        choices=["recon", "spike", "hybrid"],
                        help="recon = E1 output error; spike = E2 internal-state template; hybrid = E3")
    parser.add_argument("--lambda_spk", type=float, default=0.7)
    parser.add_argument("--lambda_rec", type=float, default=0.3)
    parser.add_argument("--hybrid_norm", type=str, default="fixed",
                        choices=["fixed", "minmax"],
                        help="fixed = train-slice median/MAD scalers (paper protocol); "
                             "minmax = per-batch min-max (reference only)")
    parser.add_argument("--vus_window", type=str, default="auto",
                        help="'auto' = official find_length_rank per file; "
                             "an integer = fixed VUS window")
    parser.add_argument("--save_streams", action="store_true", default=False,
                        help="Save per-file recon/spike score streams + label to "
                             "out_dir/streams/*.npz for offline fusion experiments")
    parser.add_argument("--train_prefix", type=str, default="ratio",
                        choices=["ratio", "official"],
                        help="'ratio' = train_ratio prefix; "
                             "'official' = per-file tr_ field (TSB-AD protocol)")
    parser.add_argument("--binary_lif", action="store_true", default=False,
                        help="LIF-dynamics control: replace LIF neurons with "
                             "stateless binary-threshold STE units")
    parser.add_argument("--rate_match", action="store_true", default=False,
                        help="With --binary_lif: calibrate binary thresholds so "
                             "activation rate matches the LIF firing rate "
                             "(removes the sparsity confound)")
    parser.add_argument("--model", type=str, default="snn", choices=["snn", "ann"],
                        help="'ann' = matched ReLU architecture with "
                             "activation-template scoring (control)")

    # Model
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--channel", type=int, default=16)
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=6)
    parser.add_argument("--lif_tau", type=float, default=1.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_srp", action="store_true", default=True)
    parser.add_argument("--no_srp", dest="use_srp", action="store_false")
    parser.add_argument("--use_spf", action="store_true", default=True)
    parser.add_argument("--no_spf", dest="use_spf", action="store_false")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--r_min", type=float, default=0.05)
    parser.add_argument("--spf_at", type=int, default=2)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None,
                        help="global RNG seed (torch/numpy/random) for repeated training runs; "
                             "None = unseeded")

    args = parser.parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"  [seed] global RNG seed = {args.seed}")
    kw = {k: v for k, v in vars(args).items()
          if k not in ('data_dir', 'out_dir', 'subset', 'max_files', 'seed')}
    evaluate_tsbad(args.data_dir, args.out_dir, args.subset, args.max_files, **kw)
