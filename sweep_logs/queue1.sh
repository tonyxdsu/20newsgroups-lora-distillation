#!/bin/bash
# Phase 1: the RoBERTa paper's fine-tuning recipe (Liu et al. 2019, Appendix C), one seed each, dev only.
cd "$(dirname "$0")/.."
PY=~/venvs/cs423/bin/python
R="--epochs 10 --wd 0.1 --warmup 0.06 --beta2 0.98 --eps 1e-6"
$PY sweep.py --name r_lr2e-5 --lr 2e-5 $R > sweep_logs/r_lr2e-5.log 2>&1
$PY sweep.py --name r_lr2e-5_headtail --lr 2e-5 $R --trunc headtail > sweep_logs/r_lr2e-5_headtail.log 2>&1
$PY sweep.py --name r_lr1e-5 --lr 1e-5 $R > sweep_logs/r_lr1e-5.log 2>&1
$PY sweep.py --name r_lr3e-5 --lr 3e-5 $R > sweep_logs/r_lr3e-5.log 2>&1
