#!/bin/bash
# Main result (Tables 1 and 2): one training per series with the hybrid score and saved streams,
# then E1 / E2 / E3 and the evidence decomposition offline from the same streams.
#   bash scripts/run_main.sh {u|m} [gpu]
source "$(dirname "$0")/hparams.sh" "$@"
python run_spire.py --data_dir $DATA --subset $SUB --train_prefix official --score_mode hybrid --save_streams \
    --out_dir outputs/${split}_hybrid $P
python analysis/offline_scores.py outputs/${split}_hybrid/streams outputs/${split}_offline.csv 8
python analysis/stream_metrics.py outputs/${split}_hybrid/streams outputs/${split}_metrics.csv 8
python analysis/summarize.py --offline outputs/${split}_offline.csv --r_min $([ "$split" = m ] && echo 0.15 || echo 0.05)
