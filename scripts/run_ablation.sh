#!/bin/bash
# Component ablation under the A1 protocol with output-error scoring (Table 5, first block):
# no SRP/SPF, SRP only, SPF only. The full model is the E1 column of the main run.
#   bash scripts/run_ablation.sh {u|m} [gpu]
source "$(dirname "$0")/hparams.sh" "$@"
python run_protocols.py --data_dir $DATA --subset $SUB --train_prefix official --causal_recon --no_srp --no_spf \
    --out_dir outputs/${split}_e1_no_srp_spf $P
python run_protocols.py --data_dir $DATA --subset $SUB --train_prefix official --causal_recon --use_srp --no_spf \
    --out_dir outputs/${split}_e1_srp_only $P
python run_protocols.py --data_dir $DATA --subset $SUB --train_prefix official --causal_recon --no_srp --use_spf \
    --out_dir outputs/${split}_e1_spf_only $P
