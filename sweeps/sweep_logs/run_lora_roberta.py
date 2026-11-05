"""Runs LoRA Roberta.ipynb with its settings cell (and optionally the rank) replaced, without changing the notebook.
As when it's run by hand, the run's dev history goes to sweep.jsonl and its probabilities to probs/; its printed
output goes to sweep_logs/NAME_seedSEED.log.

    ~/venvs/cs423/bin/python sweep_logs/run_lora_roberta.py --lr 1e-4 --seed 423           (name lora_lr1e-4)
    ~/venvs/cs423/bin/python sweep_logs/run_lora_roberta.py --lr 2e-4 --r 64 --seed 423    (name lora_r64_lr2e-4)

The rank changes with alpha fixed at 32, the LoRA paper's convention (Section 4.1): alpha / r then shrinks as r grows.
"""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent.parent   # sweeps/
NOTEBOOK = HERE.parent / "lora" / "LoRA Roberta.ipynb"

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, required=True)
p.add_argument("--seed", type=int, default=423)
p.add_argument("--r", type=int, default=16)
args = p.parse_args()

lr = f"{args.lr:.0e}".replace("e-0", "e-")   # the notebook's naming: 2e-4
name = f"lora_lr{lr}" if args.r == 16 else f"lora_r{args.r}_lr{lr}"

nb = nbformat.read(NOTEBOOK, as_version=4)
settings, model_cell = nb.cells[3], nb.cells[9]
assert settings.source.startswith("## Settings for this run")
settings.source = f'LR = {args.lr}\nSEED = {args.seed}\nSCORE_TEST = False\nNAME = "{name}"\nprint(NAME, "seed", SEED)'
old = "R, ALPHA, LORA_DROPOUT = 16, 32, 0.05"
assert old in model_cell.source
model_cell.source = model_cell.source.replace(old, f"R, ALPHA, LORA_DROPOUT = {args.r}, 32, 0.05")

try:
    NotebookClient(nb, kernel_name="cs423", timeout=None, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
finally:
    text = [o.get("text", "") or "".join(o.get("traceback", [])) for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", [])]
    (HERE / "sweep_logs" / f"{name}_seed{args.seed}.log").write_text("".join(text))
