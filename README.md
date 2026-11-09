# Head Collapse in Recursive Self-Training for Sequential Recommendation

Reproduction code for the ESWA submission **"Head Collapse in Recursive
Self-Training for Sequential Recommendation: Greedy Decoding and the Fidelity
Gate"**.

The paper diagnoses a *head collapse* in [RSIR](https://github.com/USTC-StarTeam/RSIR)
(Recursive Self-Improving Recommendation, [Zhang et al., ICML 2026](https://proceedings.mlr.press/v306/zhang26gf.html)):
one self-training cycle concentrates generated sequences onto a few popular
items. The collapse is **jointly caused by greedy decoding and the rank-based
fidelity gate** (top-*k*, *k* = 20). **Relaxing the gate** restores coverage
without hurting accuracy; temperature sampling restores diversity at an accuracy
cost.

## Contents

This repository is **not standalone**: the scripts import the RSIR framework and
run from the root of [USTC-StarTeam/RSIR](https://github.com/USTC-StarTeam/RSIR).

```
exp/            one script per experiment; *.sh are the server launch queues
patches/        the single patch applied to the RSIR framework (see Setup)
make_figs.py    regenerates the paper's three figures from the reported numbers
```

## Setup

1. Clone [USTC-StarTeam/RSIR](https://github.com/USTC-StarTeam/RSIR) and follow
   its setup (PyTorch, `wandb`, `numpy`, `pandas`, `scipy`).
2. Copy `exp/` and `patches/` into the RSIR repo root, and place `make_figs.py`
   at the root.
3. Apply the synthetic-data down-weighting patch:

```bash
git apply patches/basemodel_synthetic_weight.patch
```

This makes `training_step` scale the per-sample loss of synthetic rows by
`synthetic_weight`. Upstream RSIR does not implement this down-weighting; without
the patch, `--synthetic_weight 0.5` silently runs with uniform weights. Apart
from this patch, the framework is unchanged RSIR code (commit `3b5cc9f`).

## Running an experiment

All results average over seeds `0`--`4` (mean ± std, sample std `ddof=1`). Each
script takes `--dataset {amazon-sport,yelp}`, `--seed`, `--device`:

```bash
cd /path/to/RSIR
python exp/ablation2x2.py --dataset amazon-sport --seed 0 --device 0
```

| Paper result | Script |
|---|---|
| Single-round collapse, greedy / temperature (Table 2) | `exp/ablation2x2.py` (cells `greedy_fid` / `temp2_fid`), or `exp/run_temp5.py` |
| Selector variants (Table 3) | `exp/afd_pipeline.py --greedy` |
| 2x2 fidelity ablation (Table 4) | `exp/ablation2x2.py` |
| Rank-cutoff sweep (Table 5) | `exp/sweep_k.py` |
| Temperature frontier (Figure 2) | `exp/ablation2x2.py` (T sweep, fidelity on) |
| Adaptive / down-weighting (Table 7) | `exp/afd_pipeline.py --adaptive` / `--synthetic_weight 0.5` |
| Decoding + popularity penalty (Table 8) | `exp/sweep_pen.py` |
| Equal-size control (Sec. 5.2) | `exp/subsample_control.py`, `exp/subsample_control_5seed.py` |
| local_prob=0 control (Sec. 5.1) | `exp/local_prob0_control.py` |
| Recommendation-list diversity (2x2 rerun) | `exp/rec_div.py` |
| Recursion compounds collapse (Table 9, Figure 3) | `exp/iter_drift.py` |

`iter_drift.py` reads the RSIR iteration files
(`dataset/{ds}/{dom}/train_1th.pth ... train_8th.pth`) shipped in the RSIR repo.
Figures are regenerated with `python make_figs.py`. The `*.sh` files record the
exact launch commands used on our 2x RTX 3090 server (seeds, devices, ordering)
and contain server-specific paths; they are for reference, not portable runners.

## Data

Raw/preprocessed datasets, the RSIR iteration files, generated `.pkl` candidate
pools, and training checkpoints are large or regenerable and are not included.
Datasets come from the RSIR repo; the scripts regenerate everything else. Full
per-seed numbers and stds are reported in the paper.

## Citation

> Luankang Zhang, Hao Wang, Zhongzhou Liu, et al. *Can Recommender Systems
> Teach Themselves? A Recursive Self-Improving Framework with Fidelity
> Control.* ICML 2026, PMLR 306.
> [https://proceedings.mlr.press/v306/zhang26gf.html](https://proceedings.mlr.press/v306/zhang26gf.html)
