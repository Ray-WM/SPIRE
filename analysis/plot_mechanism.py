"""Mechanism visualization on one series: train SPIRE on the normal prefix, then plot the raw
signal, the SRP residual energy, the encoder spike raster, the firing rates, the burst and
silence ratios, and the three anomaly scores (E1 output error, E2 internal state, E3 hybrid).

Usage (from the repository root):
  python analysis/plot_mechanism.py --csv data/eval_u/<series>.csv --out mechanism.png \
      --window_size 50 --hidden_size 32 --channel 16 --num_levels 3 --kernel_size 16 \
      --num_steps 6 --lif_tau 1.5 --dropout 0.1 --alpha 0.5 --top_k 2 --r_min 0.05 --spf_at 2 \
      [--train_prefix official] [--range START END] [--seed 0]
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_spire import (TSBADWindowDataset, SpikeTemplate, build_network_astar,  # noqa: E402
                       get_point_scores_astar, train_model)


def internals_over_series(network, data, window_size, batch_size, device):
    """Raw five-dimensional internal-state vector [T, 5] and encoder spike counts [T, D]
    from non-overlapping windows (the last partial window is covered by a window ending at T)."""
    network.eval()
    T = len(data)
    starts = list(range(0, T - window_size + 1, window_size))
    if starts[-1] + window_size < T:
        starts.append(T - window_size)
    feats = np.zeros((T, SpikeTemplate.N_FEATURES), dtype=np.float64)
    counts = None
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch = starts[i:i + batch_size]
            windows = torch.from_numpy(np.stack([data[s:s + window_size] for s in batch])).to(device)
            _, _, internals = network(windows, return_internal=True)
            f = SpikeTemplate.extract_features(internals).cpu().numpy()        # [B, ws, 5]
            spikes = internals["enc_spikes"].sum(dim=1).cpu().numpy()           # [B, ws, D]
            if counts is None:
                counts = np.zeros((T, spikes.shape[-1]), dtype=np.float64)
            for j, s in enumerate(batch):
                feats[s:s + window_size] = f[j]
                counts[s:s + window_size] = spikes[j]
    return feats, counts


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="mechanism.png")
    ap.add_argument("--train_prefix", choices=["official", "ratio"], default="official")
    ap.add_argument("--train_ratio", type=float, default=0.5)
    ap.add_argument("--range", nargs=2, type=int, default=None, help="time range to plot")
    ap.add_argument("--channel_to_plot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lambda_spk", type=float, default=0.7)
    ap.add_argument("--lambda_rec", type=float, default=0.3)
    for name, default in (("window_size", 50), ("hidden_size", 32), ("channel", 16), ("num_levels", 3),
                          ("kernel_size", 16), ("num_steps", 6), ("top_k", 2), ("spf_at", 2),
                          ("batch_size", 64), ("max_epochs", 50), ("patience", 10), ("train_stride", 5)):
        ap.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("lif_tau", 1.5), ("dropout", 0.1), ("alpha", 0.5), ("r_min", 0.05), ("lr", 1e-3)):
        ap.add_argument(f"--{name}", type=float, default=default)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    T, C = data.shape
    fname = os.path.basename(args.csv)
    if args.train_prefix == "official":
        m = re.search(r"_tr_(\d+)_", fname)
        train_end = int(m.group(1)) if m else int(T * args.train_ratio)
    else:
        train_end = int(T * args.train_ratio)
    train_end = max(min(train_end, T - 1), args.window_size + 1)
    mean = np.nanmean(data[:train_end], axis=0, keepdims=True)
    std = np.nanstd(data[:train_end], axis=0, keepdims=True) + 1e-8
    data_norm = np.nan_to_num(((data - mean) / std).astype(np.float32))
    train_data = data_norm[:train_end]

    cfg = dict(kernel_size=args.kernel_size, num_levels=args.num_levels, channel=args.channel,
               hidden_size=args.hidden_size, num_steps=args.num_steps, lif_tau=args.lif_tau,
               dropout=args.dropout, use_srp=True, use_spf=True, alpha=args.alpha, top_k=args.top_k,
               r_min=args.r_min, spf_at=min(args.spf_at, args.num_levels - 1))
    network = build_network_astar(C, args.window_size, cfg)
    network = train_model(network, train_data, window_size=args.window_size, batch_size=args.batch_size,
                          lr=args.lr, max_epochs=args.max_epochs, patience=args.patience,
                          train_stride=args.train_stride, device=device)
    template = SpikeTemplate()
    template.fit(network, train_data, window_size=args.window_size, batch_size=args.batch_size * 2, device=device)
    hybrid, streams = get_point_scores_astar(network, data_norm, template, window_size=args.window_size,
                                             batch_size=args.batch_size * 2, score_mode="hybrid",
                                             lambda_spk=args.lambda_spk, lambda_rec=args.lambda_rec,
                                             train_end=train_end, return_streams=True, device=device)
    feats, counts = internals_over_series(network, data_norm, args.window_size, args.batch_size, device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = (0, T) if args.range is None else (max(0, args.range[0]), min(T, args.range[1]))
    t = np.arange(a, b)
    fig, axes = plt.subplots(6, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.6, 1, 1, 1.2]})

    def shade(ax):
        lab = label[a:b]
        edges = np.diff(np.concatenate([[0], lab, [0]]))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(a + s, a + e, color="tab:red", alpha=0.15, lw=0)
        if a < train_end < b:
            ax.axvline(train_end, color="gray", ls="--", lw=0.8)

    ax = axes[0]
    ax.plot(t, data[a:b, args.channel_to_plot], color="k", lw=0.7)
    ax.set_title(f"{fname}: raw signal (channel {args.channel_to_plot}); shaded = labelled anomaly, dashed = end of the normal prefix", fontsize=9)
    ax = axes[1]
    ax.plot(t, feats[a:b, 0], color="tab:blue", lw=0.8)
    ax.set_ylabel("|R(t)|", fontsize=8)
    ax.set_title("SRP residual energy", fontsize=9)
    ax = axes[2]
    ax.imshow(counts[a:b].T, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest",
              extent=[a, b, 0, counts.shape[1]])
    ax.set_ylabel("encoder unit", fontsize=8)
    ax.set_title("encoder spike raster (spike count over the micro-steps)", fontsize=9)
    ax = axes[3]
    ax.plot(t, feats[a:b, 1], color="tab:blue", lw=0.8, label="encoder firing rate")
    ax.plot(t, feats[a:b, 2], color="tab:orange", lw=0.8, label="backbone activity mean")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("firing rates", fontsize=9)
    ax = axes[4]
    ax.plot(t, feats[a:b, 3], color="tab:blue", lw=0.8, label="burst ratio B(t)")
    ax.plot(t, feats[a:b, 4], color="tab:orange", lw=0.8, label="silence ratio Q(t)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("burst and silence ratios", fontsize=9)
    ax = axes[5]
    ax.plot(t, minmax(streams["recon_stream"])[a:b], color="tab:gray", lw=0.8, label="S_err (E1)")
    ax.plot(t, minmax(streams["spike_stream"])[a:b], color="tab:blue", lw=0.8, label="S_spk (E2)")
    ax.plot(t, minmax(hybrid)[a:b], color="tab:red", lw=0.9, label="S_hyb (E3)")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.set_title("anomaly scores (min-max scaled for display)", fontsize=9)
    ax.set_xlabel("time step")
    for ax in axes:
        shade(ax)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"router periods={list(template.periods)} r={template.r:.3f}; saved {args.out}")


if __name__ == "__main__":
    main()
