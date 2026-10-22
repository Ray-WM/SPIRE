#!/bin/bash
# Fusion weight selected on the official tuning split (Table 1, last row; Appendix J):
# hybrid run with saved streams on the tuning series, then the lambda grid offline.
#   bash scripts/run_tuning.sh {u|m} [gpu]
source "$(dirname "$0")/hparams.sh" "$@"
python run_spire.py --data_dir $TUNE --subset $SUB --train_prefix official --score_mode hybrid --save_streams \
    --out_dir outputs/${split}_tuning $P
python analysis/offline_scores.py outputs/${split}_tuning/streams outputs/${split}_tuning_offline.csv 8
python analysis/summarize.py --offline outputs/${split}_tuning_offline.csv --r_min $([ "$split" = m ] && echo 0.15 || echo 0.05)
