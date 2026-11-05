"""LoRA fine-tuning sweep for Qwen3-Embedding-4B on 20 Newsgroups: the data, thread-grouped dev split, model and
training loop of LoRA Qwen3-Embedding-4B.ipynb, with the learning rate and seed as flags.

    ~/venvs/cs423/bin/python sweep_qwen_lora.py --name qwen_lora_lr2e-4 --lr 2e-4   (dev only; appends to sweep.jsonl)
    ~/venvs/cs423/bin/python summarize.py --final qwen_lora_lr2e-4 --select last    (test; once the setup is fixed)

As in the notebook, the weights kept are the last epoch's. Their dev and test probabilities go to probs/, but test
accuracy is only computed by summarize.py --final.

The notebook's cells between building the model and training draw random numbers too (the toy LoRA layer and the
demo batch), and its batch-printing cell used up one epoch of batch order. Both are replayed here, so --lr 1e-4
--seed 423 repeats the notebook's run up to GPU rounding (batch loss at step 50: 2.237 here, 2.239 in the notebook),
and every learning rate at one seed starts from the same A's and head and sees the same batches and dropout masks.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--seed", type=int, default=423)
p.add_argument("--tokens-per-piece", type=int, default=2048, help="halve it if the GPU runs out of memory")
p.add_argument("--smoke", action="store_true", help="stop after 50 steps; score 100 dev and test posts")
p.add_argument("--out", default="sweep.jsonl")
p.add_argument("--probs", default="probs")
p.add_argument("--save", help="where to save the trainable weights (A's, B's and the head)")
args = p.parse_args()

device = "cuda"
MODEL = "Qwen/Qwen3-Embedding-4B"
INSTRUCTION = "Instruct: Classify the given Usenet newsgroup post by the newsgroup it was posted to\nQuery:"
MAX_LENGTH = 512
BATCH_SIZE = 16
R, ALPHA, LORA_DROPOUT = 16, 32, 0.05


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

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize(texts):
    return tokenizer([INSTRUCTION + t for t in texts], truncation=True, max_length=MAX_LENGTH)["input_ids"]

ids_tr = tokenize([X_train[i] for i in train_idx])
ids_dev = tokenize([X_train[i] for i in dev_idx])
ids_test = tokenize(X_test)
y_tr, y_dev = y_train[train_idx], y_train[dev_idx]
lengths_tr = np.array([len(s) for s in ids_tr])
if args.smoke:
    ids_dev, y_dev, ids_test, y_test = ids_dev[:100], y_dev[:100], ids_test[:100], y_test[:100]


# ---- Batches: the notebook's length-grouped batches, split into pieces of at most --tokens-per-piece tokens ----
rng = np.random.default_rng(args.seed)


def train_batches():
    order = rng.permutation(len(ids_tr))
    batches = []
    for start in range(0, len(order), 50 * BATCH_SIZE):
        chunk = order[start:start + 50 * BATCH_SIZE]
        chunk = chunk[np.argsort(lengths_tr[chunk])]
        batches += [chunk[i:i + BATCH_SIZE] for i in range(0, len(chunk), BATCH_SIZE)]
    return [batches[i] for i in rng.permutation(len(batches))]


def pieces(batch, lengths):
    per_piece = max(1, args.tokens_per_piece // lengths[batch].max())
    return [batch[i:i + per_piece] for i in range(0, len(batch), per_piece)]


def pad(seqs):
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), tokenizer.pad_token_id)
    attention_mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        attention_mask[i, :len(s)] = 1
    return input_ids.to(device), attention_mask.to(device)

train_batches()   # the notebook's cell that printed example batches drew one epoch's order


# ---- Model: the notebook's LoRALinear and QwenClassifier ----
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.A = nn.Linear(base.in_features, r, bias=False, device=base.weight.device)
        self.B = nn.Linear(r, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.dropout(x))) * self.scale


class QwenClassifier(nn.Module):
    def __init__(self, qwen, n_classes):
        super().__init__()
        self.qwen = qwen
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(qwen.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.qwen(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(dim=1) - 1   # each post's <|endoftext|>, the last real token (right padding)
        return self.head(self.dropout(hidden[torch.arange(len(last), device=hidden.device), last]))

torch.manual_seed(args.seed)
qwen = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
qwen.requires_grad_(False)
for layer in qwen.layers:
    for module, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                          (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
        for name in names:
            setattr(module, name, LoRALinear(getattr(module, name), R, ALPHA, LORA_DROPOUT))
qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = QwenClassifier(qwen, 20).to(device)
trainable = [w for w in model.parameters() if w.requires_grad]
criterion = nn.CrossEntropyLoss()

# The notebook's random draws between here and training: the toy layer (CPU), a random input to layer 1's q_proj
# (GPU), and the dropout masks of a forward and backward pass on the first 4 training posts (GPU).
toy = LoRALinear(nn.Linear(6, 4, bias=False), r=2, alpha=4, dropout=0.0)
torch.randn(6)
with torch.no_grad():
    toy.B.weight.normal_()
torch.randn(3, qwen.config.hidden_size, device=device)
model.train()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(*pad(ids_tr[:4]))
criterion(logits.float(), torch.tensor(y_tr[:4], device=device)).backward()
model.zero_grad(set_to_none=True)

optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
steps_per_epoch = math.ceil(len(ids_tr) / BATCH_SIZE)
total_steps = args.epochs * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                            num_training_steps=total_steps)


def predict(seqs):
    """Logits (N, 20) in the original order, 16 posts of similar length at a time, in the training pieces."""
    model.eval()
    lengths = np.array([len(s) for s in seqs])
    order = np.argsort(lengths)
    logits = torch.empty((len(seqs), 20))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, len(order), 16):
            for idx in pieces(order[i:i + 16], lengths):
                logits[idx] = model(*pad([seqs[j] for j in idx])).float().cpu()
    return logits


# ---- Train ----
history = []
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(train_batches(), start=1):
        optimizer.zero_grad()
        batch_loss = 0.0
        for piece in pieces(batch, lengths_tr):
            input_ids, attention_mask = pad([ids_tr[i] for i in piece])
            labels = torch.tensor(y_tr[piece], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
            loss = criterion(logits.float(), labels) * len(piece) / len(batch)
            loss.backward()
            batch_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += batch_loss * len(batch)
        if step % 50 == 0:
            print(f"  epoch {epoch}, step {step}/{steps_per_epoch}: loss {batch_loss:.3f}, "
                  f"learning rate {scheduler.get_last_lr()[0]:.1e}  ({(time.perf_counter() - start) / 60:.1f} min)",
                  flush=True)
        if args.smoke and step == 50:
            break
    dev_logits = predict(ids_dev)
    dev_loss = criterion(dev_logits, torch.tensor(y_dev)).item()
    history.append({"epoch": epoch, "train_loss": round(train_loss / len(ids_tr), 4), "dev_loss": round(dev_loss, 4),
                    "dev_acc": round((dev_logits.argmax(1).numpy() == y_dev).mean() * 100, 2),
                    "seconds": round(time.perf_counter() - start)})
    print(json.dumps(history[-1]), flush=True)
    if args.smoke:
        break

result = {"name": args.name,
          "cfg": {"model": "Qwen3-Embedding-4B", "lora_r": R, "lora_alpha": ALPHA, "lora_dropout": LORA_DROPOUT,
                  "lr": args.lr, "epochs": args.epochs, "wd": 0.01, "warmup": 0.1, "max_length": MAX_LENGTH,
                  "tokens_per_piece": args.tokens_per_piece, "seed": args.seed, "select": "last"},
          "history": history, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
          "epoch_used": history[-1]["epoch"]}

# Dev and test probabilities of the last epoch, under the names summarize.py --final --select last reads
probs_dir = HERE / args.probs
probs_dir.mkdir(exist_ok=True)
np.save(probs_dir / f"{args.name}_seed{args.seed}_last_dev.npy", dev_logits.softmax(1).numpy())
np.save(probs_dir / f"{args.name}_seed{args.seed}_last_test.npy", predict(ids_test).softmax(1).numpy())
if args.save:
    torch.save({name: w.detach().cpu() for name, w in model.named_parameters() if w.requires_grad}, args.save)
print(f"done in {(time.perf_counter() - start) / 60:.1f} min, peak GPU memory {result['peak_gb']} GB", flush=True)

with open(HERE / args.out, "a") as f:
    f.write(json.dumps(result) + "\n")
