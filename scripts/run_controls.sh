#!/bin/bash
# Neuron-isolation controls (Table 3) and the architecture without SRP/SPF under hybrid scoring (Table 5):
#   bash scripts/run_controls.sh {u|m} [gpu]
source "$(dirname "$0")/hparams.sh" "$@"
python run_spire.py --data_dir $DATA --subset $SUB --train_prefix official --score_mode hybrid --model ann \
    --out_dir outputs/${split}_relu_template $P
python run_spire.py --data_dir $DATA --subset $SUB --train_prefix official --score_mode hybrid --binary_lif --rate_match \
    --out_dir outputs/${split}_binary_ratematch $P
python run_spire.py --data_dir $DATA --subset $SUB --train_prefix official --score_mode hybrid --no_srp --no_spf \
    --out_dir outputs/${split}_no_srp_spf_hybrid $P
for c in relu_template binary_ratematch no_srp_spf_hybrid; do
    python analysis/summarize.py --compare outputs/${split}_hybrid/results_$SUB.csv outputs/${split}_$c/results_$SUB.csv
done
