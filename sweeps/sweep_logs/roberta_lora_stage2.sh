#!/bin/bash
# LoRA RoBERTa at the chosen learning rate ($1): seeds 1 and 2 (for the test comparison with full fine-tuning), then
# the rank check, r = 4 and 64 with alpha 32, seed 423, dev only.
cd "$(dirname "$0")/.."
LR=$1
for seed in 1 2; do
  ~/venvs/cs423/bin/python sweep_logs/run_lora_roberta.py --lr $LR --seed $seed >> sweep_logs/run_lora_roberta.err 2>&1
done
for r in 4 64; do
  ~/venvs/cs423/bin/python sweep_logs/run_lora_roberta.py --lr $LR --r $r --seed 423 >> sweep_logs/run_lora_roberta.err 2>&1
done
