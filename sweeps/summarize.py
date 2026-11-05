"""Print sweep.jsonl as a table of dev results. With --final, also score the named runs on test (every seed found,
at the epoch --select picks) and their ensemble (mean of the runs' softmax probabilities in probs/). Only use
--final once the setup is chosen on dev.

    ~/venvs/cs423/bin/python summarize.py
    ~/venvs/cs423/bin/python summarize.py --final r_lr2e-5 --select acc
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--final", nargs="*", default=[])
p.add_argument("--select", choices=["loss", "acc", "last"], default="acc")
args = p.parse_args()

runs = [json.loads(line) for line in (HERE / "sweep.jsonl").read_text().splitlines()]

print("Dev accuracy after each epoch; best dev accuracy and lowest dev loss, with their epochs")
for r in runs:
    h = r["history"]
    best_loss = min(h, key=lambda e: e["dev_loss"])
    best_acc = max(h, key=lambda e: e["dev_acc"])   # first of ties here; sweep.py keeps the last of ties
    print(f"{r['name']:20s} seed {r['cfg']['seed']:<4d} acc {best_acc['dev_acc']:5.2f} (ep {best_acc['epoch']:2d})  "
          f"loss {best_loss['dev_loss']:.3f} (ep {best_loss['epoch']:2d})  {h[-1]['seconds']:4d} s")
    print(f"{'':20s} " + " ".join(f"{e['dev_acc']:5.2f}" for e in h))

if args.final:
    s = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    y_test = np.array([t for d, t in zip(s.data, s.target) if len(d) > 0])
    chosen = [r for r in runs if r["name"] in args.final]
    print(f"\nTest, epoch picked by dev {args.select}:")
    probs = []
    for r in chosen:
        probs.append(np.load(HERE / "probs" / f"{r['name']}_seed{r['cfg']['seed']}_{args.select}_test.npy"))
        pred = probs[-1].argmax(1)
        print(f"  {r['name']:20s} seed {r['cfg']['seed']:<4d} acc {(pred == y_test).mean() * 100:.2f}  "
              f"macro-F1 {metrics.f1_score(y_test, pred, average='macro') * 100:.2f}")
    accs = [(q.argmax(1) == y_test).mean() * 100 for q in probs]
    pred = np.mean(probs, axis=0).argmax(1)
    print(f"  mean of {len(probs)}: {np.mean(accs):.2f} (std {np.std(accs, ddof=1):.2f})" if len(probs) > 1 else "")
    print(f"  ensemble of {len(probs)}: acc {(pred == y_test).mean() * 100:.2f}  "
          f"macro-F1 {metrics.f1_score(y_test, pred, average='macro') * 100:.2f}")
