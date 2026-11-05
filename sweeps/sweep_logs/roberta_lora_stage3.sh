#!/bin/bash
# LoRA RoBERTa rank check, part 2: seeds 1 and 2 for r = 64 and r = 4 (lr 5e-4, alpha 32), so each rank has 3 seeds.
cd "$(dirname "$0")/.."
for r in 64 4; do
  for seed in 1 2; do
    ~/venvs/cs423/bin/python sweep_logs/run_lora_roberta.py --lr 5e-4 --r $r --seed $seed >> sweep_logs/run_lora_roberta.err 2>&1
  done
done
