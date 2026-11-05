"""Cross-validate the qwen3_4b_2048 search's best MLP configs (best overall, best with 1 hidden layer) on other
feature files: same folds, protocol and stacked trainer as search_mean.py, 3 seeds each. CV on train only.

    ~/venvs/cs423/bin/python -u transfer.py [--ref] [--group-by-thread] <name> <features.npz> [<more.npz> ...]

--ref runs RoBERTa <s>'s best config instead (the search's reference point, so it doesn't depend on how far the
search has got). --group-by-thread keeps each thread in one fold, as in search_mean.py. Several feature files are
concatenated (one row per message, side by side). Results go to <name>/trials.jsonl, so final_test.py and
summarize.py can read them like a search's. Re-running skips finished runs.
"""
import argparse
import json

import numpy as np
from sklearn.datasets import fetch_20newsgroups

import search_mean as sm
from final_test import best_config

parser = argparse.ArgumentParser()
parser.add_argument("--ref", action="store_true")
parser.add_argument("--group-by-thread", action="store_true")
parser.add_argument("name")
parser.add_argument("files", nargs="+")
args = parser.parse_args()
X = np.hstack([np.load(f)["X_train"] for f in args.files])
s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
y = np.array([label for doc, label in zip(s.data, s.target) if len(doc) > 0])
assert len(X) == len(y)

(sm.HERE / args.name).mkdir(exist_ok=True)
sm.TRIALS = sm.HERE / args.name / "trials.jsonl"
sm.DEADLINE = float("inf")
for line in (sm.TRIALS.read_text().splitlines() if sm.TRIALS.exists() else []):
    r = json.loads(line)
    sm.RESULTS[(sm.key(r["cfg"]), r["seed"])] = r

groups = np.unique(sm.thread_subjects(), return_inverse=True)[1] if args.group_by_thread else None
sm.log(f"{' + '.join(args.files)}: {X.shape}, {'thread-grouped' if args.group_by_thread else 'random'} folds")
cfgs = ([sm.ROBERTA_S_BEST] if args.ref else
        [best_config("qwen3_4b_2048")[1], best_config("qwen3_4b_2048", depth=1)[1]])
sm.run_configs([(c, seed) for c in cfgs for seed in (0, 1, 2)], sm.FoldData(X, y, groups), "ref" if args.ref else "transfer")
for c in cfgs:
    sm.log(f"{sm.score(c):.2f}  {sm.describe(c)}")
