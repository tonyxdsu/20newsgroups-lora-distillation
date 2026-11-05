#!/bin/bash
# Qwen LoRA learning-rate sweep, stage 1: the notebook's 1e-4 halved and doubled, seed 423, dev only.
source "$(dirname "$0")/qwen_lora_common.sh"
run qwen_lora_lr2e-4 423 --lr 2e-4
run qwen_lora_lr5e-5 423 --lr 5e-5
