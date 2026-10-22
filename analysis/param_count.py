"""Parameter counts of SPIRE (LIF) and of the matched ReLU control under the configurations of the paper.
Usage: python analysis/param_count.py   (from the repository root)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_spire import build_network_astar  # noqa: E402

PU = dict(hidden_size=32, channel=16, num_levels=3, kernel_size=16, num_steps=6, lif_tau=1.5, dropout=0.1,
          use_srp=True, use_spf=True, alpha=0.5, top_k=2, r_min=0.05, spf_at=2)
PM = dict(hidden_size=64, channel=16, num_levels=3, kernel_size=8, num_steps=4, lif_tau=2.0, dropout=0.2,
          use_srp=True, use_spf=True, alpha=0.5, top_k=2, r_min=0.15, spf_at=2)


def count(net):
    return sum(p.numel() for p in net.parameters())


if __name__ == "__main__":
    # TSB-AD-U is univariate; the multivariate channel counts span the TSB-AD-M Eval split.
    for name, cfg, Cs in (("TSB-AD-U", PU, [1]), ("TSB-AD-M", PM, [1, 2, 4, 8, 19, 29, 38, 55, 248])):
        for C in Cs:
            row = []
            for model in ("snn", "ann"):
                net = build_network_astar(C, 50, dict(cfg), model=model)
                row.append(f"{model}={count(net)}")
            print(name, f"C={C}", *row)
