# Fine-tuning results (20 Newsgroups)

Every run: batch 16, dev = 10% of the threads (1,109 posts), test = 7,370 posts (1 point = 74 posts).
"BERT recipe" = lr 2e-5, 3 epochs, weight decay 0.01, 10% warmup, epoch with the lowest dev loss.
"RoBERTa recipe" (Liu et al. 2019, Table 10) = lr 2e-5, 10 epochs, weight decay 0.1, 6% warmup, Adam β₂ 0.98 and
ε 1e-6, epoch with the highest dev accuracy.

## Test results

| Setup | Runs | Test acc | Test macro-F1 |
|---|---|---|---|
| Linear head, BERT recipe (notebook) | 3 runs | 71.56, 72.43, 72.84 | 71.13 (the 72.84 run) |
| HF head, BERT recipe (notebook) | 1 | 71.26 | 69.08 |
| HF head, RoBERTa recipe (`sweep.py`) | 3 seeds | 72.84, 73.11, 73.45: mean 73.13 (std 0.31) | 71.75, 71.69, 72.16 |
| HF head, RoBERTa recipe, ensemble of those 3 | 1 | **73.93** | 72.71 |
| HF head, RoBERTa recipe (notebook) | 1 | 71.37 (kept epoch 3) | 69.22 |

All 4 RoBERTa-recipe runs: mean 72.69 (std 0.92). All 4 BERT-recipe runs (both heads): mean 72.02. So the recipe
gains under a point on average, about the size of the seed noise. Picking the best of 10 noisy dev epochs can land
on an early epoch (the notebook run kept epoch 3, before the learning rate had decayed).

For comparison: frozen RoBERTa + MLP 68.33, frozen Qwen3-Embedding-4B + MLP ensemble 78.97, published RoBERTa-large
fine-tune 74.1 (`../FrozenResults.md`).

## How the RoBERTa recipe was chosen (`sweep.py`, seed 423, dev only)

| Setup | Best dev acc (epoch) | Last-epoch dev acc | Lowest dev loss (epoch) |
|---|---|---|---|
| BERT recipe | 76.92 (3) | 76.92 | 0.831 (3) |
| RoBERTa recipe, lr 1e-5 | 77.19 (9) | 77.01 | 0.821 (3) |
| **RoBERTa recipe, lr 2e-5** | **78.00 (8)** | 77.28 | **0.818 (2)** |
| RoBERTa recipe, lr 3e-5 | 77.82 (3) | 76.28 | 0.837 (2) |
| lr 2e-5 + head+tail truncation (Sun et al. 2019) | 78.18 (10) | 78.18 | 0.828 (2) |

- Test was only read after the setup was fixed (`summarize.py --final r_lr2e-5`).
- Head+tail truncation (first 128 + last 382 tokens): no gain on the 131 dev posts it changes (120 correct vs 121
  with the first 510 tokens). Its lead came from the short posts, whose input is identical, so it was seed noise.
- Over 10 epochs dev loss bottoms out at epoch 2 or 3 and then rises (to 1.46 at epoch 10) while dev accuracy
  keeps improving, so the epoch is chosen by dev accuracy.
- `sweep.py` groups training batches by length: twice as fast, same results (BERT recipe: 76.92 dev vs the
  notebook's 77.01).

## Qwen3-Embedding-4B with LoRA (`LoRA Qwen3-Embedding-4B.ipynb`)

LoRA rank 16, α 32, dropout 0.05 on all 7 projection matrices of all 36 layers (33M trainable weights, 0.8% of the
model), a linear head on the last token's vector, 512 tokens with the "Usenet newsgroup" instruction. lr 1e-4;
otherwise the BERT recipe, but keeping the last epoch (chosen in advance, after the sweep above).

| Run | Dev acc by epoch | Dev loss by epoch | Test acc | Test macro-F1 |
|---|---|---|---|---|
| seed 423 | 80.70, 80.79, **82.24** | **0.631**, 0.711, 0.897 | **78.68** | 77.58 |

- About the same as the frozen model: frozen Qwen3-Embedding-4B + tuned LR 78.00, + tuned MLP (10-seed ensemble)
  78.97 (`../FrozenResults.md`). Fine-tuning gained RoBERTa about 5 points over LR on its frozen vectors; Qwen, under 1.
- Sada et al.'s LoRA Qwen3-8B (general chat model, 4-bit, LoRA on attention only, 256 tokens): 77.75 on the same posts.
- Dev loss was lowest after epoch 1 while dev accuracy rose through epoch 3, the same pattern as RoBERTa's.
- 49 minutes of training (16.4 per epoch), 9.6 GB peak on the 5070 Ti.

A full fine-tune of the same model, for an 80 GB GPU on Colab: `Fine-tuned Qwen3-Embedding-4B Colab.ipynb` (not run
yet).

## Baseball vs. hockey (2 classes, A2's original task)

`Fine-tuned Roberta 2 Class.ipynb`: the same notebook and RoBERTa recipe, 1,053 training posts, 118 dev posts, 777 test posts (1 point ≈ 8 posts).

| Setup | Runs | Test acc | Test macro-F1 |
|---|---|---|---|
| HF head, RoBERTa recipe (kept epoch 7, dev 93.22) | 1 | 93.18 | 93.17 |

For comparison (`../FrozenResults.md`): frozen RoBERTa `<s>` + LR 91.63, + MLP 92.92 (one seed), frozen Qwen3-Embedding-4B + LR 95.24.
Fine-tuning cut frozen RoBERTa + MLP's error by about 4% here, against about 14% on 20 classes.
