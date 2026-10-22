#!/bin/bash
# Repeated training runs with a fixed seed (Table 2, mean and standard deviation; Appendix K for the ReLU control):
#   bash scripts/run_seeds.sh {u|m} [gpu] [seed] [snn|ann]
source "$(dirname "$0")/hparams.sh" "$@"
seed=${3:-2}; model=${4:-snn}
python run_spire.py --data_dir $DATA --subset $SUB --train_prefix official --score_mode hybrid --save_streams \
    --model $model --seed $seed --out_dir outputs/${split}_${model}_seed${seed} $P
python analysis/stream_metrics.py outputs/${split}_${model}_seed${seed}/streams outputs/${split}_${model}_seed${seed}_metrics.csv 8
