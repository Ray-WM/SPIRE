"""Per-file VUS-PR / AUC-PR / AUC-ROC of E1 (output error), E2 (five-dimensional internal state)
and E3 (fixed-scaler hybrid, lambda_spk = 0.7) from saved score streams (out_dir/streams/*.npz of
`run_spire.py --save_streams`). Used for the repeated training runs and the full metric report.
Usage: python analysis/stream_metrics.py <streams_dir> <out_csv> [n_proc]
Same fusion rule as analysis/offline_scores.py (median / 1.4826*MAD scalers on the training slice)."""
import csv, glob, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _center_scale(v):
    v = np.asarray(v, dtype=np.float64)
    c = np.median(v)
    s = 1.4826 * np.median(np.abs(v - c))
    return float(c), float(s) + 1e-8


def _fuse(spk, rec, te, lam=0.7):
    cs, ss = _center_scale(spk[:te])
    cr, sr = _center_scale(rec[:te])
    return lam * (spk - cs) / ss + (1.0 - lam) * (rec - cr) / sr


def process_file(path):
    from TSB_AD.evaluation.metrics import get_metrics
    d = np.load(path)
    label = d["label"].astype(int)
    rec = d["recon_stream"].astype(np.float64)
    fd = d["feature_dev_stream"].astype(np.float64)
    te = int(d["train_end"]); sw = int(d["sliding_window"])
    spk = fd.mean(axis=1)
    row = {"file": os.path.basename(path).replace(".npz", ".csv"), "router_r": float(d["router_r"]), "train_end": te}
    for name, s in (("E1", rec), ("E2", spk), ("E3", _fuse(spk, rec, te, 0.7))):
        try:
            m = get_metrics(s.astype(np.float32), label, slidingWindow=sw)
            for k in ("VUS-PR", "AUC-PR", "AUC-ROC"):
                row[f"{name}_{k}"] = float(m[k])
        except Exception:
            for k in ("VUS-PR", "AUC-PR", "AUC-ROC"):
                row[f"{name}_{k}"] = float("nan")
    return row


def main():
    streams_dir, out_csv = sys.argv[1], sys.argv[2]
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    files = sorted(glob.glob(os.path.join(streams_dir, "*.npz")))
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    todo = [p for p in files if os.path.basename(p).replace(".npz", ".csv") not in done]
    print(f"{streams_dir}: {len(files)} files ({len(done)} cached), {n_proc} procs")
    if not todo:
        return
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with Pool(n_proc) as pool, open(out_csv, "a", newline="") as f:
        writer = None
        for row in pool.imap_unordered(process_file, todo):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not done:
                    writer.writeheader()
            writer.writerow(row); f.flush()
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
