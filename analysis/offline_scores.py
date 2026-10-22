"""Offline evidence-source analysis from saved score streams (no retraining).

Input: the streams written by `run_spire.py --save_streams` (out_dir/streams/*.npz: recon_stream,
feature_dev_stream [T, 5], label, train_end, sliding_window, router_r). Output: one CSV row per
series with VUS-PR of

- E1 (output error), E2 (five-dimensional internal state), E3 (fixed-scaler hybrid, lambda_spk = 0.7)
- the decomposition of the internal-state vector: residual energy only, the 4-D remainder, the
  3-D spike sub-block (encoder rate, burst, silence), the backbone activity mean alone, and the
  fusion of each with E1
- single-feature and leave-one-out subsets
- scaler types (median/MAD, mean/std, median/IQR) and the lambda_spk grid

Feature dims: 0 = residual energy |R|, 1 = encoder firing rate, 2 = backbone activity mean,
3 = burst ratio, 4 = silence ratio.

Usage:
  python analysis/offline_scores.py <streams_dir> <out_csv> [n_proc]
  OFFLINE_CORE=1 restricts the output to E1/E2/E3 and the four main feature sets.
"""

import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORE_ONLY = os.environ.get("OFFLINE_CORE") == "1"

FEATURE_SETS = {
    "R_only": [0], "4D": [1, 2, 3, 4], "3D_spike": [1, 3, 4], "cont_back": [2],
    "5D": [0, 1, 2, 3, 4],
    "f_encrate": [1], "f_backact": [2], "f_burst": [3], "f_silence": [4],
    "loo_R": [1, 2, 3, 4], "loo_encrate": [0, 2, 3, 4], "loo_backact": [0, 1, 3, 4],
    "loo_burst": [0, 1, 2, 4], "loo_silence": [0, 1, 2, 3],
}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCALERS = ("mad", "std", "iqr")
if CORE_ONLY:
    # Main subset only; the sensitivity variants are skipped.
    FEATURE_SETS = {k: FEATURE_SETS[k] for k in ("R_only", "4D", "3D_spike", "5D")}
    LAMBDAS = []
    SCALERS = ()


def _center_scale(train_vals, kind):
    v = np.asarray(train_vals, dtype=np.float64)
    if kind == "mad":
        c = np.median(v)
        s = 1.4826 * np.median(np.abs(v - c))
    elif kind == "std":
        c = float(v.mean())
        s = float(v.std())
    else:  # iqr
        c = np.median(v)
        s = float(np.percentile(v, 75) - np.percentile(v, 25))
    return float(c), float(s) + 1e-8


def _fuse(spk, rec, train_end, lam, kind="mad"):
    cs, ss = _center_scale(spk[:train_end], kind)
    cr, sr = _center_scale(rec[:train_end], kind)
    return lam * (spk - cs) / ss + (1.0 - lam) * (rec - cr) / sr


def process_file(path):
    from TSB_AD.evaluation.metrics import get_metrics

    d = np.load(path)
    label = d["label"].astype(int)
    rec = d["recon_stream"].astype(np.float64)
    fd = d["feature_dev_stream"].astype(np.float64)  # [T, 5]
    te = int(d["train_end"])
    sw = int(d["sliding_window"])
    r_rel = float(d["router_r"])

    def vus(score):
        try:
            return float(get_metrics(score.astype(np.float32), label,
                                     slidingWindow=sw)["VUS-PR"])
        except Exception:
            return float("nan")

    row = {"file": os.path.basename(path).replace(".npz", ".csv"),
           "router_r": r_rel, "train_end": te}
    row["E1"] = vus(rec)
    for name, dims in FEATURE_SETS.items():
        stream = fd[:, dims].mean(axis=1)
        row[name] = vus(stream)
        if name in ("R_only", "4D", "3D_spike", "5D"):
            row[f"E1+{name}"] = vus(_fuse(stream, rec, te, 0.7))
    row["E2"] = row["5D"]
    row["E3"] = row["E1+5D"]
    spk5 = fd.mean(axis=1)
    for lam in LAMBDAS:
        row[f"lam{lam}"] = vus(_fuse(spk5, rec, te, lam))
    for kind in SCALERS:
        row[f"scaler_{kind}"] = vus(_fuse(spk5, rec, te, 0.7, kind))
    return row


def main():
    streams_dir, out_csv = sys.argv[1], sys.argv[2]
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    files = sorted(glob.glob(os.path.join(streams_dir, "*.npz")))

    # Resume: skip files already present in the output CSV.
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    todo = [p for p in files
            if os.path.basename(p).replace(".npz", ".csv") not in done]
    print(f"{streams_dir}: {len(files)} files ({len(done)} cached), "
          f"{n_proc} procs, core={CORE_ONLY}")
    if not todo:
        return

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not done
    with Pool(n_proc) as pool, open(out_csv, "a", newline="") as f:
        writer = None
        for row in pool.imap_unordered(process_file, todo):
            if row is None:
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
