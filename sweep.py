"""Fine-tuning sweep for RoBERTa-base on 20 Newsgroups: the same data, thread-grouped dev split and model
(Hugging Face's classification head) as Fine-tuned Roberta.ipynb, with the training settings as flags.

    ~/venvs/cs423/bin/python sweep.py --name base                      (dev only; appends to sweep.jsonl)
    ~/venvs/cs423/bin/python sweep.py --name final --seed 1 --test     (also scores test; once the setup is fixed)

Differences from the notebook, both for speed: posts are tokenized once up front, and training batches are
grouped by length (as fairseq, RoBERTa's own codebase, and HF's group_by_length do), so a batch of short posts
isn't padded to 512. Random batches pad to 2.6x the tokens. Evaluation sorts by length, which changes nothing
but the speed.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
p.add_argument("--beta2", type=float, default=0.999)
p.add_argument("--eps", type=float, default=1e-8)
p.add_argument("--llrd", type=float, default=1.0, help="layer-wise lr decay: layer k gets lr * llrd^(12-k)")
p.add_argument("--trunc", choices=["head", "headtail"], default="head",
               help="long posts: first 510 tokens, or first 128 + last 382 (Sun et al. 2019)")
p.add_argument("--seed", type=int, default=423)
p.add_argument("--select", choices=["loss", "acc", "last"], default="loss", help="which epoch --test uses")
p.add_argument("--test", action="store_true")
p.add_argument("--out", default="sweep.jsonl")
args = p.parse_args()

device = "cuda"
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)


# ---- Data: exactly the notebook's posts and split ----
def load(subset, **kw):
    s = fetch_20newsgroups(subset=subset, **kw)
    keep = [i for i, d in enumerate(s.data) if len(d) > 0]
    return [s.data[i] for i in keep], np.array(s.target)[keep], [s.filenames[i] for i in keep]

X_train, y_train, files_train = load("train", remove=("headers", "footers", "quotes"))
X_test, y_test, _ = load("test", remove=("headers", "footers", "quotes"))

with_headers = fetch_20newsgroups(subset="train")
subject_of = {f: re.search(r"^Subject:(.*)$", doc, re.M).group(1)
              for f, doc in zip(with_headers.filenames, with_headers.data)}
threads = [re.sub(r"^(re:\s*)+", "", subject_of[f].strip(), flags=re.IGNORECASE).lower() for f in files_train]
thread_ids = np.unique(threads, return_inverse=True)[1]
splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=423)
train_idx, dev_idx = next(splitter.split(X_train, y_train, groups=thread_ids))
assert (len(train_idx), len(dev_idx)) == (9987, 1109)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.model_max_length = 10**6   # we truncate ourselves; silences the "longer than 512" warning


def encode(texts):
    """<s> + up to 510 tokens + </s>. "head" matches the notebook's truncation=True, max_length=512."""
    out = []
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        if len(ids) > 510:
            ids = ids[:510] if args.trunc == "head" else ids[:128] + ids[-382:]
        out.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])
    return out

ids_train = encode(X_train)
ids_test = encode(X_test)
ids_tr = [ids_train[i] for i in train_idx]
ids_dev = [ids_train[i] for i in dev_idx]
y_tr, y_dev = y_train[train_idx], y_train[dev_idx]


def pad(seqs):
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), tokenizer.pad_token_id)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return input_ids.to(device), mask.to(device)


def train_batches(lengths, batch):
    """Shuffle, cut into megabatches of 50 batches, sort each by length, cut into batches, shuffle the batches."""
    order = rng.permutation(len(lengths))
    batches = []
    for s in range(0, len(order), 50 * batch):
        mega = order[s:s + 50 * batch]
        mega = mega[np.argsort(-lengths[mega], kind="stable")]
        batches += [mega[i:i + batch] for i in range(0, len(mega), batch)]
    return [batches[i] for i in rng.permutation(len(batches))]


# ---- Model: the notebook's RobertaClassifier ----
class RobertaClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(768, 768), nn.Tanh(), nn.Dropout(0.1), nn.Linear(768, n_classes))

    def forward(self, input_ids, attention_mask):
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(hidden[:, 0]))

model = RobertaClassifier(20).to(device)


def param_groups():
    """One group per depth so --llrd can give lower layers smaller learning rates (head and layer 12: lr)."""
    depth_of = {}
    for name, w in model.named_parameters():
        if name.startswith("roberta.embeddings"):
            d = 0
        elif name.startswith("roberta.encoder.layer."):
            d = int(name.split(".")[3]) + 1
        else:
            d = 12
        depth_of.setdefault(d, []).append(w)
    return [{"params": ws, "lr": args.lr * args.llrd ** (12 - d)} for d, ws in sorted(depth_of.items())]

optimizer = torch.optim.AdamW(param_groups(), lr=args.lr, weight_decay=args.wd,
                              betas=(0.9, args.beta2), eps=args.eps)
steps_per_epoch = -(-len(ids_tr) // args.batch)
total_steps = args.epochs * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
criterion = nn.CrossEntropyLoss()


def predict(seqs, batch=64):
    """Logits (N, 20) in the original order; batches of similar length for speed."""
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    out = torch.empty((len(seqs), 20))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            out[idx] = model(*pad([seqs[j] for j in idx])).float().cpu()
    return out


def score(logits, y):
    return criterion(logits, torch.tensor(y)).item(), (logits.argmax(1).numpy() == y).mean()


# ---- Train ----
lengths_tr = np.array([len(s) for s in ids_tr])
history, states = [], {}
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    model.train()
    total = 0.0
    for b in train_batches(lengths_tr, args.batch):
        input_ids, mask = pad([ids_tr[i] for i in b])
        labels = torch.tensor(y_tr[b], device=device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, mask)
        loss = criterion(logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(b)
    dev_loss, dev_acc = score(predict(ids_dev), y_dev)
    history.append({"epoch": epoch, "train_loss": round(total / len(ids_tr), 4), "dev_loss": round(dev_loss, 4),
                    "dev_acc": round(dev_acc * 100, 2), "seconds": round(time.perf_counter() - start)})
    print(json.dumps(history[-1]), flush=True)
    # keep the weights of the epochs that "loss" and "acc" would pick so far (on the CPU; "last" is the model)
    for rule, better in (("loss", history[-1]["dev_loss"] <= min(h["dev_loss"] for h in history)),
                         ("acc", history[-1]["dev_acc"] >= max(h["dev_acc"] for h in history))):
        if better:
            states[rule] = (epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

result = {"name": args.name, "cfg": {k: v for k, v in vars(args).items() if k not in ("name", "test", "out")},
          "history": history, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}

# Save dev and test probabilities for each rule's epoch, so the best setup's run can join the final ensemble
# without retraining. Test accuracy is only computed with --test, once the setup is chosen on dev.
states["last"] = (args.epochs, None)
probs_dir = HERE / "probs"
probs_dir.mkdir(exist_ok=True)
done = {}   # epoch -> (dev probabilities, test logits), so an epoch two rules share is scored once
for rule in ("last", "loss", "acc"):   # "last" first, while the model still holds the last epoch's weights
    epoch, state = states[rule]
    if epoch not in done:
        if state is not None:
            model.load_state_dict(state)
        done[epoch] = (predict(ids_dev).softmax(1).numpy(), predict(ids_test))
    dev_probs, test_logits = done[epoch]
    np.save(probs_dir / f"{args.name}_seed{args.seed}_{rule}_dev.npy", dev_probs)
    np.save(probs_dir / f"{args.name}_seed{args.seed}_{rule}_test.npy", test_logits.softmax(1).numpy())
    if args.test and rule == args.select:
        test_pred = test_logits.argmax(1).numpy()
        result.update(epoch_used=epoch, test_acc=round((test_pred == y_test).mean() * 100, 2),
                      test_macro_f1=round(metrics.f1_score(y_test, test_pred, average="macro") * 100, 2))
        print(f"test (epoch {epoch}): acc {result['test_acc']}, macro-F1 {result['test_macro_f1']}")

with open(HERE / args.out, "a") as f:
    f.write(json.dumps(result) + "\n")
