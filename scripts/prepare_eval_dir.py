"""Create a folder of symbolic links to the series of an official TSB-AD file list.

  python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-U --list File_List/TSB-AD-U-Eva.csv --out data/eval_u
  python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-M --list File_List/TSB-AD-M-Eva.csv --out data/eval_m
  python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-U --list File_List/TSB-AD-U-Tuning.csv --out data/tune_u

The runners take the folder as --data_dir and read every CSV in it; the normal prefix of each
series is the tr_ field of its file name (--train_prefix official).
"""
import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True, help="unzipped TSB-AD-U or TSB-AD-M folder")
ap.add_argument("--list", required=True, help="one of the File_List/*.csv lists")
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
n = missing = 0
with open(args.list) as f:
    for row in csv.DictReader(f):
        fname = row.get("file_name") or row.get("filename") or ""
        if not fname:
            continue
        src = os.path.abspath(os.path.join(args.data_dir, fname))
        dst = os.path.join(args.out, fname)
        if not os.path.exists(src):
            missing += 1
            continue
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        n += 1
print(f"{n} series linked into {args.out}" + (f" ({missing} listed files not found in {args.data_dir})" if missing else ""))
