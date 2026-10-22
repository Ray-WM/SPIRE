"""
Operational protocols of the SPIRE backbone on TSB-AD under output-error scoring.

Three protocols in one script (the backbone, the SRP encoder and the SPF block are those of
spire/protocol_backbone.py; the internal spike states are not read here):

  prediction     (A2)  causal TCN + shifted target:   y[t] = f(x[0:t]) predicts x[t+1]
  reconstruction (A3)  non-causal TCN + same target:  y[t] = f(x[0:T]) reconstructs x[t]
  causal_recon   (A1)  causal TCN + same target:      y[t] = f(x[0:t]) reconstructs x[t]
                       (the protocol used by SPIRE; default)

  pred_warmup applies only to prediction mode: skip the first W positions of each window,
  where the model has insufficient causal history for SRP to work.

The --no_srp / --no_spf flags give the component ablation of the paper (E1 ladder:
no SRP/SPF, SRP only, SPF only, SRP + SPF) under the A1 protocol.

Known limitation: in the prediction and causal_recon modes the SRP encoder's Conv1d still
uses symmetric padding (about one step of future context) and the period router runs an FFT
on the whole window; only the TCN backbone is strictly causal.
"""

import os, re, sys, glob, argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

VALID_MODES = ("prediction", "reconstruction", "causal_recon")


# ============================================================
# Helpers
# ============================================================

def _is_causal_backbone(ad_mode: str) -> bool:
    """Whether the TCN backbone should use causal (left-padded) convolutions."""
    return ad_mode in ("prediction", "causal_recon")


def _is_shifted_target(ad_mode: str) -> bool:
    """Whether training uses shifted target (y[t] -> x[t+1])."""
    return ad_mode == "prediction"


# ============================================================
# Dataset
# ============================================================

class TSBADWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 100, stride: int = 1):
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

def build_network(input_size: int, window_size: int, ad_mode: str, cfg: dict) -> nn.Module:
    cfg = dict(cfg)  # shallow copy
    causal = _is_causal_backbone(ad_mode)
    from spire.protocol_backbone import SpikeTCN_SRPSPF_AD
    return SpikeTCN_SRPSPF_AD(
        input_size=input_size, max_length=window_size,
        causal=causal, **cfg,
    )


# ============================================================
# Training
# ============================================================

def train_model(
    network, train_data,
    window_size=100, batch_size=64,
    lr=1e-3, max_epochs=50, patience=10, train_stride=5,
    ad_mode="prediction", pred_warmup=0,
    device="cuda",
):
    """
    prediction:    loss = MSE( y[W : T-1],  x[W+1 : T] )   - shifted, with warmup
    reconstruction: loss = MSE( y,  x )
    causal_recon:  loss = MSE( y,  x )                      - same as recon, causal backbone
    """
    shifted = _is_shifted_target(ad_mode)

    if shifted:
        assert window_size >= 2, f"window_size must be >= 2, got {window_size}"
        max_warmup = window_size - 2
        if pred_warmup > max_warmup:
            print(f"  [WARN] pred_warmup={pred_warmup} clamped to {max_warmup}")
            pred_warmup = max_warmup

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
            windows = windows.to(device)            # [B, T, C]
            y_seq, _ = network(windows)             # [B, C, T]
            output = y_seq.permute(0, 2, 1)         # [B, T, C]

            if shifted:
                W = pred_warmup
                loss = loss_fn(output[:, W:-1, :], windows[:, W + 1:, :])
            else:
                loss = loss_fn(output, windows)

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
# Point-level anomaly scores
# ============================================================

def get_point_scores(
    network, data,
    window_size=100, batch_size=128,
    ad_mode="prediction", pred_warmup=0,
    device="cuda",
) -> np.ndarray:
    """
    prediction:
      e[j] = (x[s+j+1] - y[s+j])^2   j = W .. ws-2
      -> score_sum[s+W+1 : s+ws]
      Positions 0..W filled with nearest valid score.

    reconstruction / causal_recon:
      e[j] = (x[s+j] - y[s+j])^2      j = 0 .. ws-1
      -> score_sum[s : s+ws]
    """
    shifted = _is_shifted_target(ad_mode)

    if shifted:
        max_warmup = window_size - 2
        pred_warmup = min(pred_warmup, max_warmup)

    network.eval()
    T, C = data.shape
    dataset = TSBADWindowDataset(data, window_size, stride=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    score_sum = np.zeros(T, dtype=np.float64)
    score_count = np.zeros(T, dtype=np.float64)

    with torch.no_grad():
        for windows, starts in loader:
            windows = windows.to(device)
            y_seq, _ = network(windows)
            output = y_seq.permute(0, 2, 1)

            if shifted:
                all_errors = (windows[:, 1:, :] - output[:, :-1, :]).pow(2).mean(dim=-1)
                W = pred_warmup
                valid_errors = all_errors[:, W:].cpu().numpy()
                starts_np = starts.numpy()
                for i in range(len(starts_np)):
                    s = starts_np[i]
                    score_sum[s + W + 1: s + window_size] += valid_errors[i]
                    score_count[s + W + 1: s + window_size] += 1
            else:
                errors = (windows - output).pow(2).mean(dim=-1).cpu().numpy()
                starts_np = starts.numpy()
                for i in range(len(starts_np)):
                    s = starts_np[i]
                    score_sum[s: s + window_size] += errors[i]
                    score_count[s: s + window_size] += 1

    # Aggregate
    scores = np.divide(
        score_sum, score_count,
        out=np.zeros_like(score_sum),
        where=(score_count > 0),
    )

    # Fill uncovered positions
    if shifted:
        first_covered = pred_warmup + 1
        if first_covered < T:
            scores[:first_covered] = scores[first_covered]

    uncovered = (score_count == 0)
    if uncovered.any():
        valid_idx = np.where(~uncovered)[0]
        if len(valid_idx) > 0:
            for idx in np.where(uncovered)[0]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                scores[idx] = scores[nearest]

    return scores.astype(np.float32)


# ============================================================
# Main interface
# ============================================================

def run_SpikeAD(
    data: np.ndarray,
    # AD mode
    ad_mode: str = "prediction",
    pred_warmup: int = 0,
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
    device: str = None,
) -> np.ndarray:
    assert ad_mode in VALID_MODES, \
        f"ad_mode must be one of {VALID_MODES}, got '{ad_mode}'"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T, C = data.shape

    # Normalize using the training portion. train_end (per-file tr_ field,
    # official TSB-AD protocol) overrides the train_ratio prefix when given.
    if train_end is None:
        train_end = int(T * train_ratio)
    train_end = int(train_end)
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
    network = build_network(C, window_size, ad_mode, network_cfg)

    # Effective warmup (only for prediction mode)
    effective_warmup = pred_warmup if ad_mode == "prediction" else 0

    # Train
    network = train_model(
        network, train_data,
        window_size=window_size,
        batch_size=batch_size, lr=lr,
        max_epochs=max_epochs, patience=patience,
        train_stride=train_stride,
        ad_mode=ad_mode,
        pred_warmup=effective_warmup,
        device=device,
    )

    # Score
    scores = get_point_scores(
        network, data_norm,
        window_size=window_size,
        batch_size=batch_size * 2,
        ad_mode=ad_mode,
        pred_warmup=effective_warmup,
        device=device,
    )

    return scores


# ============================================================
# Batch evaluation
# ============================================================

MODE_LABELS = {
    "prediction": "Prediction (shifted target, causal backbone)",
    "reconstruction": "Reconstruction (same target, non-causal backbone)",
    "causal_recon": "Causal Reconstruction (same target, causal backbone)",
}


def evaluate_tsbad(data_dir, out_dir, subset="TSB-AD-U", max_files=None,
                   vus_window="auto", **model_kwargs):
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if max_files:
        csv_files = csv_files[:max_files]

    ad_mode = model_kwargs.get('ad_mode', 'prediction')
    warmup = model_kwargs.get('pred_warmup', 0)
    # 'official' = per-file tr_ field from the file name (TSB-AD protocol);
    # 'ratio' = train_ratio prefix (default 0.5).
    train_prefix = model_kwargs.pop('train_prefix', 'ratio')
    mode_str = MODE_LABELS.get(ad_mode, ad_mode)
    if ad_mode == "prediction" and warmup > 0:
        mode_str += f" [warmup={warmup}]"
    print(f"AD Mode: {mode_str}  [train_prefix={train_prefix}]")

    all_results = []
    for i, csv_path in enumerate(csv_files):
        fname = os.path.basename(csv_path)
        print(f"\n[{i+1}/{len(csv_files)}] Processing: {fname}") 
        try:
            df = pd.read_csv(csv_path).dropna()
            data = df.iloc[:, :-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            T, C = data.shape
            print(f"  Shape: T={T}, C={C}, anomaly_ratio={label.mean():.4f}")
            ws = model_kwargs.get('window_size', 100)
            if T < ws * 2:
                print(f"  SKIP: too short (T={T} < 2*ws={2*ws})")
                continue
            # Official TSB-AD protocol: per-file ACF sliding window for VUS;
            # an integer vus_window gives a fixed window.
            if str(vus_window) == "auto":
                sliding_window = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
            else:
                sliding_window = int(vus_window)
            per_file_kw = dict(model_kwargs)
            if train_prefix == "official":
                m = re.search(r"_tr_(\d+)_", fname)
                if m is None:
                    print(f"  SKIP: no tr_ field in name for official prefix")
                    continue
                per_file_kw["train_end"] = int(m.group(1))
            t0 = time.time()
            scores = run_SpikeAD(data, **per_file_kw)
            elapsed = time.time() - t0
            metrics = get_metrics(scores, label, slidingWindow=sliding_window)
            metrics['file'] = fname
            metrics['time'] = elapsed
            metrics['T'] = T
            metrics['C'] = C
            metrics['sliding_window'] = sliding_window
            all_results.append(metrics)
            print(f"  VUS-PR: {metrics.get('VUS-PR', 'N/A')}, "
                  f"AUC-ROC: {metrics.get('AUC-ROC', 'N/A')}, "
                  f"Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(out_dir, f"results_{subset}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results Summary ({len(all_results)} files) - {mode_str}")
        print(f"{'='*60}")
        for col in results_df.select_dtypes(include=[np.number]).columns:
            if col not in ['T', 'C', 'time', 'sliding_window']:
                print(f"  {col}: mean={results_df[col].mean():.4f}, "
                      f"std={results_df[col].std():.4f}")
        print(f"\n  Avg time: {results_df['time'].mean():.1f}s")
        print(f"  Results saved to: {results_path}")
    return all_results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Operational protocols of the SPIRE backbone on TSB-AD")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/protocols")
    parser.add_argument("--subset", type=str, default="TSB-AD-U",
                        choices=["TSB-AD-U", "TSB-AD-M"])
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--vus_window", type=str, default="auto",
                        help="'auto' = official find_length_rank per file; "
                             "an integer = fixed VUS window")

    # AD mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--prediction", dest="ad_mode", action="store_const", const="prediction",
        help="A2: causal TCN + shifted target (y[t] -> x[t+1])")
    mode_group.add_argument(
        "--reconstruction", dest="ad_mode", action="store_const", const="reconstruction",
        help="A3: non-causal TCN + same target (y[t] -> x[t])")
    mode_group.add_argument(
        "--causal_recon", dest="ad_mode", action="store_const", const="causal_recon",
        help="A1: causal TCN + same target (y[t] -> x[t]); the protocol of SPIRE")
    parser.set_defaults(ad_mode="causal_recon")

    parser.add_argument("--pred_warmup", type=int, default=0,
                        help="(Prediction only) Skip first W positions per window. "
                             "Recommended: window_size // 2. Ignored for other modes.")

    # Model
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--lif_tau", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.2)

    # SRP/SPF
    parser.add_argument("--use_srp", action="store_true", default=True)
    parser.add_argument("--no_srp", dest="use_srp", action="store_false")
    parser.add_argument("--use_spf", action="store_true", default=True)
    parser.add_argument("--no_spf", dest="use_spf", action="store_false")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--r_min", type=float, default=0.1)
    parser.add_argument("--spf_at", type=int, default=0)

    # Training
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--train_prefix", type=str, default="ratio",
                        choices=["ratio", "official"],
                        help="'official' = per-file tr_ field (TSB-AD protocol); "
                             "'ratio' = train_ratio prefix")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_stride", type=int, default=5)

    args = parser.parse_args()
    model_kwargs = dict(
        ad_mode=args.ad_mode,
        pred_warmup=args.pred_warmup,
        window_size=args.window_size, hidden_size=args.hidden_size,
        channel=args.channel, num_levels=args.num_levels,
        kernel_size=args.kernel_size, num_steps=args.num_steps,
        lif_tau=args.lif_tau, dropout=args.dropout,
        use_srp=args.use_srp, use_spf=args.use_spf,
        alpha=args.alpha, top_k=args.top_k, r_min=args.r_min,
        spf_at=args.spf_at,
        train_ratio=args.train_ratio, lr=args.lr,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, train_stride=args.train_stride,
        train_prefix=args.train_prefix,
    )
    evaluate_tsbad(data_dir=args.data_dir, out_dir=args.out_dir,
                   subset=args.subset, max_files=args.max_files,
                   vus_window=args.vus_window, **model_kwargs)
