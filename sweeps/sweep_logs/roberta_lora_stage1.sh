#!/bin/bash
# LoRA RoBERTa learning-rate sweep (LoRA Roberta.ipynb, "How to run it"): 1e-4 and 5e-4 next to the existing 2e-4,
# seed 423, dev only. Each run's notebook output goes to sweep_logs/lora_lr*_seed423.log.
cd "$(dirname "$0")/.."
for lr in 1e-4 5e-4; do
  ~/venvs/cs423/bin/python sweep_logs/run_lora_roberta.py --lr $lr --seed 423 >> sweep_logs/run_lora_roberta.err 2>&1
done
