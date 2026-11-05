#!/bin/bash
# Qwen LoRA at the chosen learning rate, 1e-4 (stage 1 wasn't promising): seeds 1 and 2, next to the notebook's 423.
source "$(dirname "$0")/qwen_lora_common.sh"
run qwen_lora_lr1e-4 1 --lr 1e-4
run qwen_lora_lr1e-4 2 --lr 1e-4
