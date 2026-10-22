"""Summaries behind the paper's tables, from the per-series CSVs.

  python analysis/summarize.py --offline results/main/u_offline_run1.csv --r_min 0.05
      means of every column of an analysis/offline_scores.py table, paired bootstrap differences
      (E3 vs E1, 5D vs residual energy only, 5D vs E1, E1+5D vs E1+R, 3-D spike block vs E1),
      equal-count terciles of the router reliability, per-domain means, and the fallback count

  python analysis/summarize.py --compare results/main/u_hybrid_run1.csv results/controls/u_relu_template_run1.csv
      paired bootstrap difference of VUS-PR (first minus second) between two result CSVs of
      run_spire.py / run_protocols.py, matched by file name

  python analysis/summarize.py --seeds results/main/u_hybrid_run1.csv results/seeds/u_hybrid_run2.csv results/seeds/u_hybrid_run3.csv
      mean and standard deviation of the per-split mean of every metric column over training runs

Bootstrap: 10,000 resamples of the series, percentile 95% interval, Pr(delta > 0), and the
win / tie / loss counts (exact ties). The generator seed is fixed (0).
"""
import argparse
import csv
import numpy as np

rng = np.random.default_rng(0)


def load_rows(path):
    return list(csv.DictReader(open(path)))


def load_res(path, col="VUS-PR"):
    return {r["file"]: float(r[col]) for r in load_rows(path)}


def boot(a, b, n=10000):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    N = len(d)
    idx = rng.integers(0, N, size=(n, N))
    m = d[idx].mean(1)
    win = int((d > 0).sum())
    loss = int((d < 0).sum())
    tie = N - win - loss
    return d.mean(), float((m > 0).mean()), np.percentile(m, 2.5), np.percentile(m, 97.5), win, tie, loss


def fmt_boot(a_name, b_name, a, b):
    dm, pr, lo, hi, w, t, l = boot(a, b)
    return (f"  {a_name} - {b_name}: mean={dm:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
            f"Pr(delta>0)={pr:.3f}  win/tie/loss={w}/{t}/{l}")


def offline(path, r_min):
    rows = load_rows(path)
    cols = [c for c in rows[0] if c not in ("file", "router_r", "train_end")]
    V = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    print(f"===== {path} (n={len(rows)})")
    print("means:")
    for c in cols:
        print(f"  {c:14s} {np.nanmean(V[c]):.4f}")
    print("paired differences:")
    for a, b in [("E3", "E1"), ("5D", "R_only"), ("5D", "E1"), ("E1+5D", "E1+R_only"), ("3D_spike", "E1")]:
        if a in V and b in V:
            print(fmt_boot(a, b, V[a], V[b]))
    r = np.array([float(x["router_r"]) for x in rows])
    order = np.argsort(r, kind="stable")
    k = len(r) // 3
    print("router-reliability terciles (equal count):")
    for name, sl in (("low", order[:k]), ("mid", order[k:2 * k]), ("high", order[2 * k:])):
        if len(sl) == 0:
            continue
        print(f"  {name:4s} n={len(sl):3d} r in [{r[sl].min():.3f}, {r[sl].max():.3f}]  "
              f"E1={V['E1'][sl].mean():.3f}  E3={V['E3'][sl].mean():.3f}  "
              f"E3-E1={V['E3'][sl].mean() - V['E1'][sl].mean():+.3f}  E2={V['5D'][sl].mean():.3f}")
    dom = {}
    for i, x in enumerate(rows):
        dom.setdefault(x["file"].split("_")[1], []).append(i)
    print("per-domain (E3, E1, n), sorted by E3:")
    for d, ix in sorted(dom.items(), key=lambda kv: -V["E3"][kv[1]].mean()):
        ix = np.array(ix)
        print(f"  {d:14s} E3={V['E3'][ix].mean():.3f}  E1={V['E1'][ix].mean():.3f}  n={len(ix)}")
    print(f"series with router_r < r_min ({r_min}): {(r < r_min).sum()}; router_r == 0 (global fallback): {(r == 0).sum()}")


def compare(a_path, b_path, col):
    a, b = load_res(a_path, col), load_res(b_path, col)
    keys = sorted(set(a) & set(b))
    av, bv = [a[k] for k in keys], [b[k] for k in keys]
    print(f"===== {col}: {a_path} ({np.mean(av):.4f}) vs {b_path} ({np.mean(bv):.4f}), n={len(keys)}")
    print(fmt_boot("first", "second", av, bv))


def seeds(paths):
    tables = [load_rows(p) for p in paths]
    cols = [c for c in tables[0][0] if c not in ("file", "time", "sliding_window", "T", "C", "router_r", "train_end")]
    print(f"===== {len(paths)} runs")
    for c in cols:
        try:
            means = [np.nanmean([float(r[c]) for r in t]) for t in tables]
        except ValueError:
            continue
        print(f"  {c:14s} " + "  ".join(f"{m:.4f}" for m in means) +
              f"   mean={np.mean(means):.4f}  sd={np.std(means, ddof=1) if len(means) > 1 else 0.0:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", type=str, help="an analysis/offline_scores.py table")
    ap.add_argument("--r_min", type=float, default=0.05, help="router threshold of the split (0.05 for U, 0.15 for M)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two result CSVs (A minus B)")
    ap.add_argument("--col", type=str, default="VUS-PR", help="metric column for --compare")
    ap.add_argument("--seeds", nargs="+", help="result CSVs of repeated training runs")
    args = ap.parse_args()
    if args.offline:
        offline(args.offline, args.r_min)
    if args.compare:
        compare(args.compare[0], args.compare[1], args.col)
    if args.seeds:
        seeds(args.seeds)
