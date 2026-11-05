# Sourced by the qwen_lora_stage*.sh queues. run NAME SEED FLAGS...: one sweep_qwen_lora.py run, logged to
# sweep_logs/NAME_seedSEED.log. If the GPU runs out of memory (other programs share it), it retries once with
# pieces of 1,024 tokens: the same gradients, just more pieces per batch.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=~/venvs/cs423/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WEIGHTS=/tmp/claude-1000/-mnt-c-Users-tonyx-OneDrive-School-cs423/d8e1a644-f92d-4529-b8e4-6f3fcefaca6d/scratchpad
run() {
  local name=$1 seed=$2; shift 2
  local log=sweep_logs/${name}_seed$seed.log
  $PY -u sweep_qwen_lora.py --name $name --seed $seed --save $WEIGHTS/${name}_seed$seed.pt "$@" > $log 2>&1 && return
  grep -q "out of memory" $log || return 1
  mv $log ${log%.log}_oom.log
  $PY -u sweep_qwen_lora.py --name $name --seed $seed --save $WEIGHTS/${name}_seed$seed.pt "$@" \
    --tokens-per-piece 1024 > $log 2>&1
}
