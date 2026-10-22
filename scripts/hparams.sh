#!/bin/bash
# Hyper-parameters of the paper (one set per split). Sourced by the other scripts.
PU="--window_size 50 --hidden_size 32 --channel 16 --num_levels 3 --kernel_size 16 --num_steps 6 --lif_tau 1.5 --dropout 0.1 --alpha 0.5 --top_k 2 --r_min 0.05 --spf_at 2 --lr 1e-3 --train_stride 5 --max_epochs 50 --patience 10"
PM="--window_size 50 --hidden_size 64 --channel 16 --num_levels 3 --kernel_size 8 --num_steps 4 --lif_tau 2.0 --dropout 0.2 --alpha 0.5 --top_k 2 --r_min 0.15 --spf_at 2 --lr 1e-3 --train_stride 5 --max_epochs 50 --patience 10"
split=${1:?usage: $0 {u|m} [gpu]}
export CUDA_VISIBLE_DEVICES=${2:-0}
if [ "$split" = m ]; then P="$PM"; DATA=data/eval_m; SUB=TSB-AD-M; TUNE=data/tune_m; else P="$PU"; DATA=data/eval_u; SUB=TSB-AD-U; TUNE=data/tune_u; fi
cd "$(dirname "$0")/.."
mkdir -p outputs
