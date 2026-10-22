# SPIRE: Spikes as Detectors

Code and per-series results for the paper

> **Spikes as Detectors: Phase-Conditioned Spiking Dynamics for Time-Series Anomaly Detection.**
> Miao Wang, Gangyi Ding, Yunlin Lei, Yu Zhang, Jingfeng Chen, Xu Yang. NeurIPS 2026.

SPIRE asks whether a detector's own spiking dynamics can serve as anomaly evidence. A causal
spiking backbone (a period-aware residual encoder, a spiking temporal convolutional network and
a period-folding block) is trained on the normal prefix of each series with a reconstruction
loss. The same trained model is then scored from one of three evidence sources:

| score mode | evidence | what is scored |
|---|---|---|
| `recon` (E1) | output error | squared reconstruction error of the current step |
| `spike` (E2) | internal state | phase-conditioned robust deviation of five internal-state statistics (residual energy, encoder firing rate, backbone activity mean, burst ratio, silence ratio) from per-phase normal templates |
| `hybrid` (E3) | both | late fusion of the two z-scored streams with fixed median / MAD scalers fitted on the normal prefix |

Two controls isolate the neuron under an otherwise identical pipeline: a matched ReLU network
scored with activation templates (`--model ann`) and a rate-matched stateless binary unit at every
LIF site (`--binary_lif --rate_match`). A second runner compares the operational protocols
(causal current-step, one-step prediction, non-causal reconstruction) under output-error scoring.

Official TSB-AD protocol, VUS-PR, one training per series:

| | TSB-AD-U | TSB-AD-M |
|---|---|---|
| E1 output error | 0.409 | 0.336 |
| E2 internal state | 0.288 | 0.342 |
| E3 hybrid (lambda_spk = 0.7) | 0.394 | 0.355 |
| matched ReLU control (hybrid) | 0.362 | 0.340 |
| rate-matched binary control (hybrid) | 0.340 | 0.344 |

## Layout

```
run_spire.py          SPIRE on TSB-AD: training, templates, the three score modes, the two controls
run_protocols.py      operational protocols A1 / A2 / A3 and the SRP / SPF component ablation (output error)
spire/                the model: srp_spf.py (period router, SRP encoder, SPF block, LIF builders),
                      backbone.py (SPIRE backbone with internal-state access), protocol_backbone.py
                      (causal / non-causal switch), backbone_ann.py (matched ReLU control), controls.py
analysis/             offline_scores.py (E1 / E2 / E3, feature decomposition, lambda and scaler grids
                      from saved streams), stream_metrics.py (VUS-PR / AUC per stream), summarize.py
                      (paired bootstrap, reliability terciles, per-domain means, seed statistics),
                      param_count.py, plot_mechanism.py (the mechanism figure on one series)
scripts/              prepare_eval_dir.py and the launch scripts of every experiment in the paper
File_List/            the official TSB-AD evaluation and tuning file lists
results/              per-series CSVs behind the paper's tables (see below)
tests/                smoke tests on synthetic tensors (CPU, no data)
```

## Install

```bash
conda create -n spire python=3.10
conda activate spire
pip install -r requirements.txt
pip install git+https://github.com/TheDatumOrg/TSB-AD.git   # VUS-PR and the per-series window
python -m pytest tests -q
```

The spiking units come from SpikingJelly (`spikingjelly==0.0.0.0.14`); the code was developed
with PyTorch 2.x and runs on a single GPU (a full pass over one split takes a few GPU-hours).

## Data

No data is included. Download the official archives of the TSB-AD benchmark
(https://github.com/TheDatumOrg/TSB-AD): https://www.thedatum.org/datasets/TSB-AD-U.zip
and https://www.thedatum.org/datasets/TSB-AD-M.zip, unzip them to `data/TSB-AD-U/` and
`data/TSB-AD-M/`, and link the official evaluation and tuning series:

```bash
python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-U --list File_List/TSB-AD-U-Eva.csv    --out data/eval_u
python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-M --list File_List/TSB-AD-M-Eva.csv    --out data/eval_m
python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-U --list File_List/TSB-AD-U-Tuning.csv --out data/tune_u
python scripts/prepare_eval_dir.py --data_dir data/TSB-AD-M --list File_List/TSB-AD-M-Tuning.csv --out data/tune_m
```

The runners follow the official protocol: the normal prefix of each series is the `tr_` field of
its file name (`--train_prefix official`) and the VUS window is computed per series with
`find_length_rank` (`--vus_window auto`).

## Quick start

```bash
# hybrid score on TSB-AD-U with the paper's hyper-parameters, streams saved for the offline analyses
python run_spire.py --data_dir data/eval_u --subset TSB-AD-U --train_prefix official \
    --score_mode hybrid --save_streams --out_dir outputs/u_hybrid \
    --window_size 50 --hidden_size 32 --channel 16 --num_levels 3 --kernel_size 16 --num_steps 6 \
    --lif_tau 1.5 --dropout 0.1 --alpha 0.5 --top_k 2 --r_min 0.05 --spf_at 2 \
    --lr 1e-3 --train_stride 5 --max_epochs 50 --patience 10

# E1 / E2 / E3 and the decomposition of the internal-state vector from the saved streams
python analysis/offline_scores.py outputs/u_hybrid/streams outputs/u_offline.csv 8
python analysis/summarize.py --offline outputs/u_offline.csv --r_min 0.05
```

`--max_files 5` runs a quick check on five series. The hyper-parameters of the two splits are in
`scripts/hparams.sh` (TSB-AD-U: hidden 32, kernel 16, 6 micro-steps, tau 1.5, dropout 0.1,
r_min 0.05; TSB-AD-M: hidden 64, kernel 8, 4 micro-steps, tau 2.0, dropout 0.2, r_min 0.15; both:
window 50, 16 channels, 3 levels, alpha 0.5, top-2 periods, SPF at level 2, lr 1e-3, stride 5,
50 epochs, patience 10).

## Reproducing the paper

Each script takes the split (`u` or `m`) and an optional GPU index and writes to `outputs/`.

| paper | script | result files in `results/` |
|---|---|---|
| Table 1 (main), Table 2 (evidence sources), Table 7 (full metrics), the per-domain, reliability-tercile, feature and sensitivity tables | `scripts/run_main.sh` | `main/{u,m}_hybrid_run1.csv`, `main/{u,m}_offline_run1.csv` |
| Table 2 mean and s.d. over three training runs | `scripts/run_seeds.sh {u,m} 0 2`, `... 3` | `seeds/{u,m}_hybrid_run{2,3}.csv`, `seeds/{u,m}_metrics_run{2,3}.csv` |
| Table 3 (neuron controls), Table 5 second block (hybrid without SRP/SPF) | `scripts/run_controls.sh` | `controls/{u,m}_relu_template_run{1,2,3}.csv`, `controls/{u,m}_binary_ratematch.csv`, `controls/{u,m}_no_srp_spf_hybrid.csv` |
| Table 5 first block (SRP / SPF ladder under output error) | `scripts/run_ablation.sh` | `ablation/{u,m}_e1_{no_srp_spf,srp_only,spf_only}.csv` |
| Table 6 (operational protocols) | `scripts/run_protocols.sh` | `protocols/{u,m}_a{2,3}_*.csv` |
| Table 1 last row and the tuning-split table (fusion weight selected on the official tuning split) | `scripts/run_tuning.sh` | `tuning/{u,m}_tuning_hybrid.csv`, `tuning/{u,m}_tuning_offline.csv` |
| parameter table | `python analysis/param_count.py` | |
| mechanism figure | `python analysis/plot_mechanism.py --csv data/eval_u/<series>.csv --out mechanism.png` | |

`analysis/summarize.py --compare A.csv B.csv` reproduces the paired bootstrap intervals and the
win / tie / loss counts of the paper from any two per-series result files, for example

```bash
python analysis/summarize.py --compare results/main/m_hybrid_run1.csv results/controls/m_relu_template_run1.csv
python analysis/summarize.py --seeds results/main/u_hybrid_run1.csv results/seeds/u_hybrid_run2.csv results/seeds/u_hybrid_run3.csv
```

The per-series files have one row per evaluation series (350 for TSB-AD-U, 180 for TSB-AD-M)
with the TSB-AD metrics, the series name, the wall time and the VUS window; the `*_offline_*.csv`
files add the E1 / E2 / E3 scores, the feature subsets, the lambda grid and the scaler variants
computed from the saved streams of the same training run.

## Acknowledgements

The backbone modules were developed inside a fork of SeqSNN (https://github.com/microsoft/SeqSNN);
the spiking units use SpikingJelly (https://github.com/fangwei123456/spikingjelly); the benchmark,
the file lists and the metrics are those of TSB-AD (https://github.com/TheDatumOrg/TSB-AD).

## License

MIT, see `LICENSE`.
