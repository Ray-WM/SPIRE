#!/bin/bash
# Operational protocols under output-error scoring (Table 6): A2 one-step prediction (warm-up 25)
# and A3 non-causal reconstruction. A1 is the E1 column of the main run.
#   bash scripts/run_protocols.sh {u|m} [gpu]
source "$(dirname "$0")/hparams.sh" "$@"
python run_protocols.py --data_dir $DATA --subset $SUB --train_prefix official --prediction --pred_warmup 25 \
    --out_dir outputs/${split}_a2_prediction $P
python run_protocols.py --data_dir $DATA --subset $SUB --train_prefix official --reconstruction \
    --out_dir outputs/${split}_a3_reconstruction $P
