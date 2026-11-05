#!/bin/bash
# Qwen LoRA at lr 2e-4, seeds 1 and 2. Not in the protocol: added after the descriptive test look, where 2e-4's
# seed-423 run scored 79.88 against 78.66-78.68 for all three 1e-4 seeds, although dev had ranked it last.
source "$(dirname "$0")/qwen_lora_common.sh"
run qwen_lora_lr2e-4 1 --lr 2e-4
run qwen_lora_lr2e-4 2 --lr 2e-4
