#!/bin/bash
# Phase 2: two more seeds of the chosen setup (for the ensemble), then the updated notebook.
cd "$(dirname "$0")/.."
PY=~/venvs/cs423/bin/python
R="--lr 2e-5 --epochs 10 --wd 0.1 --warmup 0.06 --beta2 0.98 --eps 1e-6"
for s in 1 2; do
  $PY sweep.py --name r_lr2e-5 $R --seed $s > sweep_logs/r_lr2e-5_seed$s.log 2>&1
done
~/venvs/cs423/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cs423 \
  --ExecutePreprocessor.timeout=-1 "Fine-tuned Roberta.ipynb" > sweep_logs/notebook.log 2>&1
