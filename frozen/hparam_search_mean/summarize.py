"""Print a summary of the mean-pooling searches: run counts, the best 3-seed configs, and the one-change-at-a-time
sweeps around the final center (how much CV accuracy moves when one setting changes).

    ~/venvs/cs423/bin/python summarize.py [search folder ...]      (from this folder; default: roberta modernbert)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from search_mean import SPACE, describe, key

HERE = Path(__file__).resolve().parent

for enc in sys.argv[1:] or ("roberta", "modernbert"):
    rows = [json.loads(line) for line in (HERE / enc / "trials.jsonl").read_text().splitlines()]
    seeds = defaultdict(dict)
    for r in rows:
        seeds[key(r["cfg"])][r["seed"]] = r
    three = {k: (np.mean([s[i]["mean_dev_acc"] for i in (0, 1, 2)]), s[0]["cfg"])
             for k, s in seeds.items() if all(i in s for i in (0, 1, 2))}
    ranked = sorted(three.values(), key=lambda t: -t[0])
    print(f"\n=== {enc}: {len(rows)} runs, {len(seeds)} configs, {len(three)} with 3 seeds. "
          f"Phases: {dict(Counter(r['phase'] for r in rows))}")
    print("Best 3-seed configs:")
    for acc, cfg in ranked[:5]:
        print(f"  {acc:.2f}  {describe(cfg)}")
    # The sweep table is centered on the config with the most one-change neighbors with 3 seeds, i.e. the last
    # sweep's center (the very best config may have won in the last sweep, so nothing was swept around it; and on a
    # flat top the center can drop out of the top 5)
    space = {n: sorted(set(vs) | {r["cfg"][n] for r in rows}) for n, vs in SPACE.items()}  # incl. --wider values
    neighbors = lambda cfg: sum(key({**cfg, n: v}) in three for n, vs in space.items() for v in vs if v != cfg[n])
    best_acc, center = max(ranked, key=lambda t: (neighbors(t[1]), t[0]))
    print(f"One change at a time from {describe(center)} ({best_acc:.2f}); CV change, 3 seeds each:")
    for name, values in space.items():
        cells = []
        for v in values:
            k = key({**center, name: v})
            if k in three:
                cells.append(f"{v:g}: {three[k][0] - best_acc:+.2f}" if v != center[name] else f"[{v:g}]")
        if len(cells) > 1:
            print(f"  {name:16s} " + "   ".join(cells))
