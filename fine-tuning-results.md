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
| seed 423 (the notebook) | 80.70, 80.79, **82.24** | **0.631**, 0.711, 0.897 | 78.68 | 77.58 |
| seed 1 (`sweep_qwen_lora.py`) | 81.88, 80.70, **82.60** | **0.610**, 0.696, 0.974 | 78.66 | 77.43 |
| seed 2 (`sweep_qwen_lora.py`) | 80.70, 81.06, **81.33** | **0.685**, 0.727, 0.935 | 78.67 | 77.61 |
| mean of 3 / ensemble | | | **78.67 / 79.73** | 77.54 / 78.58 |

- About the same as the frozen model, by the rule fixed before seeds 1 and 2 ran (step 5 under "Learning-rate
  sweep"): the mean is 0.67 above frozen Qwen3-Embedding-4B + tuned LR (78.00), and the ensemble 0.76 above the
  tuned MLP's 10-seed ensemble (78.97; 298 vs. 242 posts right only in one, McNemar p = 0.018). Both are under the
  1-point bar, though the ensemble's lead is unlikely to be chance. Fine-tuning gained RoBERTa about 5 points over LR
  on its frozen vectors; Qwen, under 1.
- The ensemble leads because the LoRA seeds disagree, not because each model is better. One LoRA run (78.67) scores a
  bit below one frozen MLP (78.80 ± 0.22). But averaging 3 LoRA runs gains 1.06, while averaging 10 MLPs on the same
  frozen vectors gains 0.17. LoRA RoBERTa shows the same pattern (conjecture 2 below).
- The three test scores agree to 2 posts (5,797–5,799 right) by luck. Each pair of seeds disagrees on 12.4% of the
  test posts, with about 280 right only in each. On dev the seeds spread 1.3 points, and at lr 2e-4 they spread 1.2
  on test.
- Sada et al.'s LoRA Qwen3-8B (general chat model, 4-bit, LoRA on attention only, 256 tokens): 77.75 on the same posts.
- Dev loss was lowest after epoch 1 while dev accuracy rose through epoch 3, in all 3 seeds: the same pattern as
  RoBERTa's.
- 49 minutes of training (16.4 per epoch), 9.6 GB peak on the 5070 Ti.

A full fine-tune of the same model, for an 80 GB GPU on Colab: `Fine-tuned Qwen3-Embedding-4B Colab.ipynb` (not run
yet).

### Learning-rate sweep (`sweep_qwen_lora.py`, 2026-11-02)

The run above used lr 1e-4 untuned. `sweep_qwen_lora.py` is the notebook's code with the learning rate and seed as
flags. At seed 423 it replays the notebook's random draws, so every learning rate starts from the same A's and head
and sees the same batches and dropout masks. At lr 1e-4 it repeats the notebook's run: batch loss at step 50 2.237,
vs. the notebook's 2.239. The notebook's run is in `sweep.jsonl` as `qwen_lora_lr1e-4`, seed 423, with its cached
test probabilities. The learning rate comes first because it's the setting LoRA is most sensitive to. Rank barely
mattered in the LoRA paper (Section 7.2) or in QLoRA (Section 4), and this run already fit 98% of its own training
posts (`../../Distillation-lessons.md`).

Protocol, fixed before the runs:
1. **Dev only, seed 423:** lr 5e-5 and 2e-4, around 1e-4's last-epoch dev 82.24. Compare last-epoch dev accuracy,
   since the last epoch is the one the recipe keeps.
2. **Look further** if an end of the range beats the best so far by 0.5 dev points or more: halve or double again,
   at most twice. For scale, the last-epoch dev accuracy of each 3-seed RoBERTa setup in `sweep.jsonl` has a std of
   0.1–0.4.
3. **Choose** the best last-epoch dev accuracy. Within 0.5 points of 1e-4 counts as a tie and keeps 1e-4.
4. **Seeds 1 and 2** at the chosen lr. If that isn't 1e-4, seeds 1 and 2 at 1e-4 too, so both have 3 seeds.
5. **Test** with `summarize.py --final NAME --select last`. Same rule as for distillation: a gain of the chosen lr
   over 1e-4 under 1 point (3-seed means) is no effect; 1 point or more with McNemar p < 0.05 (3-seed ensembles)
   means tuning helps. LoRA against frozen Qwen gets the same rule: 3-seed mean against frozen + LR (78.00), and
   ensemble against the frozen MLP ensemble (78.97).

If stage 1 doesn't look promising (neither end 0.5 above 82.24), the planned LoRA RoBERTa sweep goes first
(`LoRA Roberta.ipynb`, "How to run it": lr 1e-4 and 5e-4 on dev next to 2e-4's, then seeds 1 and 2; its decision
rule is under conjecture 2 below), run by `sweep_logs/run_lora_roberta.py`. Step 4 follows. If time is left, the
RoBERTa runs also get a rank check: r = 4 and 64 at the chosen lr, α fixed at 32, seed 423, dev only.

**Stage 1 (dev, seed 423): not promising.** A 4× range of learning rates moved last-epoch dev accuracy by 0.6 points.

| lr | Dev acc by epoch | Last-epoch dev loss | Last-epoch train loss |
|---|---|---|---|
| 5e-5 | 80.16, 80.79, 82.15 | 0.795 | 0.220 |
| **1e-4** (the notebook's run) | 80.70, 80.79, **82.24** | 0.897 | 0.146 |
| 2e-4 | 81.70, 80.79, 81.61 | 1.076 | 0.107 |

- Neither end beats 82.24 by 0.5, so 1e-4 stays (5e-5 is within 0.5: a tie).
- A higher learning rate fit the training posts more closely (lower train loss) and made the model more
  overconfident on dev (higher dev loss) without making it more accurate.
- 52.5 minutes per run, 9.6 GB peak.

**Steps 4–5: 1e-4 ties the frozen model.** Seeds 1 and 2 and the test scores are in the table above.

**Then a descriptive test look at the other two learning rates** (seed 423, after 1e-4 was chosen). The protocol
allowed it for reporting, not for choosing:

| lr | Last-epoch dev acc | Test acc | Test macro-F1 |
|---|---|---|---|
| 5e-5 | 82.15 | 78.63 | 77.34 |
| 1e-4, 3 seeds | 82.24, 82.60, 81.33 | 78.68, 78.66, 78.67 | 77.58, 77.43, 77.61 |
| 2e-4 | 81.61 | **79.88** | 78.89 |

- 2e-4 beats each of the three 1e-4 runs by 1.2 points (McNemar p ≤ 0.0003 each), and 5e-5 ties 1e-4 (p = 0.88).
  Dev had ranked 2e-4 last.
- Dev couldn't settle it. At 1e-4 alone, the seed moves last-epoch dev by 1.3 points (std 0.65), more than the 0.63
  that put 2e-4 below 1e-4. Step 2's 0.5-point bar assumed RoBERTa's dev noise (std 0.1–0.4). Test has 6.6× as many
  posts, and the three 1e-4 seeds agreed to 0.02 there, so the 1.2-point gap looked real.
- **Follow-up, not in the protocol:** seeds 1 and 2 at 2e-4, chosen after this look. Seed 423's 79.88 is the score
  that triggered them, so it's biased upward. Seeds 1 and 2 are fresh draws, so they decide. Rule, fixed before they
  ran: compare each with the 1e-4 run of the same seed, which had the same starting A's, batches and dropout masks.
  If both score 1 point or more higher with McNemar p < 0.05, the learning rate matters here and dev couldn't see it.
  If both are under 1 point higher, 79.88 was mostly one lucky run. Anything else is mixed. Either way, 1e-4 stays
  the protocol's answer for LoRA vs. frozen, and any 2e-4 result is labeled as chosen with test.

**Follow-up result: one lucky run.** Seeds 1 and 2 at 2e-4 scored 0.35 and 0.01 above their 1e-4 twins, both under
1 point, so by the rule seed 423's 79.88 was mostly luck.

| lr 2e-4 | Dev acc by epoch | Last-epoch dev loss | Test acc | vs. 1e-4, same seed |
|---|---|---|---|---|
| seed 423 (prompted the follow-up) | 81.70, 80.79, 81.61 | 1.076 | 79.88 | +1.19 (p = 0.0001) |
| seed 1 | 80.52, 80.52, 82.06 | 1.084 | 79.01 | +0.35 (p = 0.29) |
| seed 2 | 79.26, 80.70, 80.70 | 1.100 | 78.68 | +0.01 (p = 1.0) |
| mean of 3 / ensemble | 81.46 (last epoch) | | 79.19 / 80.56 | +0.52 / +0.83 (p = 0.003) |

- **McNemar's test between two single runs doesn't account for the seed.** It asks whether these two trained models
  differ by more than the choice of test posts explains, not whether other seeds would give the same gap. Seed 423's
  +1.19 had p = 0.0001 and didn't repeat. To compare settings, the unit is the training run: means over seeds, as the
  step-5 rule uses.
- **On test, the seed moves a run as much as the gap that looked real.** The 1e-4 seeds' 0.02 agreement was luck: at
  2e-4 the seeds spread 1.2 points (std 0.62).
- Even with seed 423, 2e-4 leads 1e-4 by 0.52 in the mean and 0.83 in the ensemble (p = 0.003), both under the
  1-point bar. The two fresh seeds alone lead by 0.18.
- A higher learning rate fit the training posts more tightly (last-epoch train loss 0.10 vs. 0.14 at 1e-4, 3-seed
  means) and was more overconfident on dev (loss 1.09 vs. 0.94), with no reliable accuracy change either way.

**LoRA RoBERTa, learning rate (dev, seed 423).** Best-epoch dev accuracy, the notebook's selection rule, and last
epoch:

| lr | Best dev acc (epoch) | Last-epoch dev acc |
|---|---|---|
| 1e-4 | 77.55 (8) | 77.10 |
| 2e-4 (run earlier) | 77.73 (5) | 77.55 |
| **5e-4** | **78.18** (7) | 77.82 |
| 1e-3 | 77.37 (10) | 77.37 |

5e-4 won, 0.45 points above 2e-4, and both columns rose with the learning rate. Since the best value sat at the edge
of the grid, one more step (1e-3) was added before choosing; the notebook's rule then picks among all four. This
step wasn't in the notebook's protocol. It was decided after seeing the three runs above, and before 1e-3 ran.
1e-3 was unstable early (dev 69.43, 66.91, 68.98 in epochs 1–3) and ended lower, so **5e-4 is the choice**.
That's 25× full fine-tuning's best (2e-5), against the ~10× Schulman et al. found for LLMs. With our α / r = 2,
it's roughly 1e-3 at the LoRA paper's α / r = 1, about twice the paper's 4e-4–5e-4 for RoBERTa-base.

Seeds 1 and 2 at 5e-4: best dev 78.36 and 76.92, so the seed alone moves best-epoch dev by 1.4 points.

Rank check at 5e-4 (α 32, seed 423), best dev: r = 4 78.54, r = 16 78.18, r = 64 78.90. r = 64 clears the 0.5
bar and r = 4 doesn't, but both differences are inside the seed spread at r = 16. Rule, fixed before the runs: both
ranks get seeds 1 and 2, and the three ranks are compared on test. A 3-seed mean within 1 point of r = 16's means
rank doesn't matter here.

**Result: rank doesn't matter here.** Seeds in the order 423, 1, 2:

| Rank | LoRA weights | Best dev acc | Test acc | Test mean | Test ensemble |
|---|---|---|---|---|---|
| 4 | 0.66M | 78.54, 77.28, 77.91 | 72.70, 73.39, 72.16 | 72.75 | 74.48 |
| **16** | 2.65M | 78.18, 78.36, 76.92 | 73.23, 73.34, 73.31 | 73.29 | **75.07** |
| 64 | 10.6M | 78.90, 78.18, 78.09 | 73.96, 73.27, 72.99 | **73.41** | 74.91 |

- r = 4 is 0.54 below r = 16 and r = 64 is 0.12 above, both inside 1 point. A 16× range of ranks moved the mean by
  0.66, about the seed std within one rank (0.06–0.62). The ensembles differ by at most 0.60 (r = 16 vs. 4, p = 0.04),
  also under 1 point.
- Peak memory 5.1 GB (r = 4 and 16) and 5.3 GB (r = 64), and about the same time.
- LoRA's extra seed diversity doesn't depend on the rank: at every rank, a pair of seeds disagrees on about 17% of
  the test posts, vs. 14.0% for full fine-tuning. r = 64's ensemble beats full fine-tuning's too (74.91 vs. 73.93,
  p = 0.002), and r = 4's leads without significance (74.48, p = 0.1).
- So Qwen gets no rank runs: at 53 minutes each, they were worth it only if rank mattered here. The LoRA paper
  (Section 7.2) and QLoRA (Section 4) found the same for their models.

**What the sweep found** (17 runs, about 7 hours):
- **No setting moved a 3-seed test mean by 1 point.** That covers Qwen's learning rate (1e-4 vs. 2e-4, plus one run
  at 5e-5) and RoBERTa's rank (4 to 64). RoBERTa's learning rate (1e-4 to 1e-3) moved best dev less than the seed did
  (0.8 vs. 1.4 points).
- **The seed moved test as much as any setting:** 1.2 points within one Qwen setting (2e-4) and within one RoBERTa
  rank (r = 4).
- **Averaging 3 seeds gained more than any setting:** 1.1 and 1.4 points for Qwen, 1.5 to 1.8 for RoBERTa, for 3
  times the training.
- **So the conclusions don't change:** LoRA Qwen ties the frozen model (78.67 vs. 78.00; ensembles 79.73 vs. 78.97),
  and LoRA RoBERTa ties full fine-tuning (73.29 vs. 73.13).

## Conjectures to check after the full fine-tune

Written before running `Fine-tuned Qwen3-Embedding-4B Colab.ipynb`. LoRA (78.68) tied the frozen model (78.00 with LR,
78.97 with the MLP ensemble). Two explanations fit:
- **Fine-tuning of any kind adds little here** (conjecture 1).
- **LoRA in particular fell short,** and training every weight would help (conjecture 2 would be wrong).

Only the full fine-tune tells these apart. And if conjecture 1 holds, Qwen can't test conjecture 2: LoRA and full
fine-tuning would tie because neither helps. So conjecture 2 also gets a test on RoBERTa, where full fine-tuning
clearly helps (`LoRA Roberta.ipynb`).

**Decision rule** (proposed; fix it before the run, not after). Full fine-tune minus LoRA, on test:
- under 1 point: no difference (about one run's seed noise: std 0.92 over the 4 RoBERTa-recipe runs);
- 2 points or more, with McNemar p < 0.05: LoRA held Qwen back;
- in between: inconclusive, more seeds needed.

### 1. Frozen Qwen3-Embedding-4B is already about as good as fine-tuning can make it

Maybe on this dataset, maybe for classification in general: Qwen3-Embedding may be very good at classification without
any fine-tuning.

Why it could be true:
- **Qwen3-Embedding was already trained on classification.** Its first training stage used about 150M synthetic text
  pairs covering retrieval, bitext mining, classification and semantic similarity. The second used about 12M of the
  best synthetic pairs plus about 7M labeled ones (MS MARCO, NQ, HotpotQA, NLI, ...), and then several checkpoints were
  merged (Zhang et al. 2025, Sections 3.2–3.3 and Table 6). The instruction then tells it which task to summarize the
  text for. RoBERTa's `<s>` was never trained to summarize anything. That would explain why fine-tuning gained RoBERTa
  about 5 points over frozen + LR (67.23 → 72.69) but Qwen under 1 (78.00 → 78.68).
- **It may have been trained on this very dataset.** Unknown either way: see "Was Qwen trained on 20 Newsgroups?"
  below.
- **The errors left may be out of reach on these inputs.** They fall between groups that really overlap (religion,
  politics, PC hardware) and on short posts (41% of posts of up to 64 tokens were wrong in CV), and the test posts
  are later in time than the training ones (`../FrozenResults.md`, `../../Lessons.md`).
- **Kumar et al. 2022 (ICLR), "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution":**
  with good pre-trained features and a shifted test distribution, fine-tuning every weight beat a linear probe
  in-distribution (about +2%, averaged over 10 datasets) but lost to it out-of-distribution (about −7%). Their
  explanation: while a randomly initialized head is still learning, its gradients distort the pre-trained features.
  Their fix, LP-FT, trains the head on frozen features first and then fine-tunes everything. Our later-in-time test
  set is a mild shift of that kind, and LoRA's dev accuracy kept climbing (80.70, 80.79, 82.24) while its test
  accuracy ended next to frozen.

What the full fine-tune would show:
- **Supports it:** full fine-tuning also lands within about 1 point of frozen + LR (78.00). That wouldn't say *why*
  frozen is already that good, though (see the next section).
- **Refutes it, for this dataset:** full fine-tuning clearly beats both frozen and LoRA (decision rule above).
- **Kumar et al.'s pattern:** dev well above LoRA's 82.24, but test unchanged. Then the gain is real on posts from the
  training period and doesn't carry over to later ones, and LP-FT is the next experiment.
- **Error overlap:** if frozen, LoRA and full fine-tuning mostly miss the same test posts, the posts are hard rather
  than the models weak. Per-post predictions: the frozen ensemble's are in
  `frozen/hparam_search_mean/final_test_qwen3_4b_2048_instrB_threads_ensemble_pred.npy`, and the Colab notebook saves the
  full fine-tune's probabilities. The LoRA run saved only its weights (`qwen3-embedding-4b-lora.pt`), so its test
  predictions need recomputing: build the model as in the notebook, `load_state_dict(..., strict=False)`, and about 3
  minutes of test inference.
- **"Classification in general"** needs more datasets (e.g. SST-5 or TREC, `../../Further Work.md`). One dataset can
  only support "on 20 Newsgroups".

#### Was Qwen trained on 20 Newsgroups? Unknown

- **It isn't in the documented training data, but that record is incomplete.** The report's labeled training sets are
  retrieval, QA and NLI sets (Table 6: "MS MARCO, NQ, HotpotQA, NLI, Dureader, T2-Ranking, SimCLUE, MIRACL, MLDR,
  Mr.TyDi, Multi-CPR, CodeSearchNet .etc"). MTEB's own record of Qwen3-Embedding's training data lists 11 retrieval
  and reranking sets, and no classification or clustering ones (`training_data` in mteb's
  `models/model_implementations/qwen3_models.py`). But the report's list ends in "etc.", and the training data isn't
  public (`public_training_data=None`).
- **Qwen was evaluated on a 20 Newsgroups task.** MTEB's `TwentyNewsgroupsClustering` clusters subject lines only
  (the headers we remove) and has only a test split. Qwen published an instruction for it ("Identify the topic or
  theme of the given news articles", the prompt the new instruction beat). Evaluating on it isn't training on it.
- **Other embedding models do train on it.** MTEB lists `TwentyNewsgroupsClustering` in the training data of NVIDIA's
  NV-Embed, next to many other MTEB classification and clustering sets. So it's a known way to score higher on MTEB,
  and "etc." could hide it.
- **The posts themselves are probably in Qwen3's pre-training data.** Qwen3-Embedding starts from the Qwen3 base
  model (report, Section 2), pre-trained on about 36T tokens, and 20 Newsgroups is copied all over the web (GitHub,
  Kaggle, Hugging Face, scikit-learn's download). The ~150M synthetic pairs were also generated from documents in that
  pre-training corpus (Section 3.3 says so for retrieval). Reading the posts isn't learning their labels, but the
  labels are close by. The "bydate" files have no `Newsgroups:` header (0 of 18,846 posts), yet every copy stores each
  post in a folder named after its newsgroup (`20news-bydate-test/rec.sport.hockey/54367`), and the original Usenet
  posts carried the header.

**If the full fine-tune ties frozen, is contamination the likely reason?** It's possible, but the tie alone can't
single it out:
- **Every explanation predicts the tie.** Qwen learned this dataset's labels in training, or it learned topic
  classification in general (the synthetic classification pairs), or the remaining errors are out of any model's
  reach. A result that all three predict can't tell them apart.
- **The RoBERTa contrast doesn't point at contamination either.** RoBERTa was pre-trained only to fill in masked
  words, so its `<s>` vector was never trained to summarize a post. Fine-tuning trains that for the first time, so a
  big gain is expected without any contamination. Qwen's last token was already trained to summarize, so a small gain
  is expected without contamination too. RoBERTa also had more room: even fine-tuned (73), it stayed below frozen Qwen
  (78–79).

What would single it out:
- **A dataset Qwen can't have seen.** The same frozen-vs.-fine-tuned comparison on topic-classification data written
  after Qwen3-Embedding's release (June 2025), e.g. recent posts from forums that match some of the 20 topics. If
  fine-tuning helps there but not on 20 Newsgroups, contamination becomes the likely reason. If it doesn't help there
  either, Qwen is just good at topic classification.
- **Not a membership test.** Min-K% Prob (Shi et al. 2024, ICLR) guesses whether a text was in an LLM's pre-training
  data from its least likely words, and could be run on the Qwen3-4B base model. But Duan et al. 2024 (COLM) found
  such tests barely beat random guessing on real pre-training data. Their apparent successes came from members and
  non-members written at different times, which is unavoidable when 1993 posts are compared with anything else. A
  positive result would also show only that Qwen read the posts, not that it learned their labels.

**Either way, the write-up should say it can't be ruled out.** If Qwen was trained on 20 Newsgroups with labels,
frozen Qwen vs. RoBERTa isn't a like-for-like comparison of what each model learned in general.

### 2. LoRA is about as good as full fine-tuning

For:
- **Hu et al. 2022 (LoRA, ICLR), Table 2.** GLUE average: RoBERTa-base 86.4 fully fine-tuned vs. 87.2 with LoRA (0.3M
  trainable weights), RoBERTa-large 88.9 vs. 89.0, DeBERTa-XXL 91.1 vs. 91.3, with LoRA rank 8 on the query and value
  matrices only. The comparison leans LoRA's way, though: the full fine-tuning numbers are copied from earlier papers,
  while LoRA got its own sweep over learning rate, epochs and batch size (25–80 epochs for RoBERTa-base, Table 9), the
  best epoch of each run, and the median of 5 seeds.
- **Dettmers et al. 2023 (QLoRA, NeurIPS), Table 3.** RoBERTa-large on GLUE: 88.6 fully fine-tuned vs. 88.8 with LoRA.
  T5 from 80M to 11B weights on Super-NaturalInstructions: within 1.3 RougeL points either way. Section 4: LoRA on only
  the query and value matrices didn't match full fine-tuning of LLaMA 7B; "LoRA on all linear transformer block layers
  are required to match full finetuning performance", and the rank r didn't matter.
- **Schulman et al. 2025, "LoRA Without Regret"** (Thinking Machines blog, not peer-reviewed). On small-to-medium
  instruction and reasoning datasets, LoRA matched full fine-tuning when applied to all layers. The MLP layers matter
  most, and attention-only LoRA underperformed. The best LoRA learning rate was consistently about 10× full
  fine-tuning's. LoRA fell behind with large batches, and once the dataset exceeded its capacity.

Against:
- **Biderman et al. 2024, "LoRA Learns Less and Forgets Less" (TMLR).** Code and math, with instruction fine-tuning
  (about 100K pairs) and continued pre-training (20B tokens): at typical ranks, LoRA did substantially worse than full
  fine-tuning. Full fine-tuning's weight changes had 10–100× the rank of typical LoRA settings. In exchange, LoRA
  forgot less outside the target domain.
- **Shuttleworth et al. 2024, "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (arXiv).** Even at equal
  accuracy, the two learn different weights. LoRA adds "intruder dimensions", new top singular vectors that full
  fine-tuning doesn't create, and these cause forgetting and worse continual learning. So equal test accuracy doesn't
  mean the same model.

Read together: LoRA tied where the task was classification (GLUE) or the dataset small, and fell behind on large
datasets in hard domains (code, math). 20 Newsgroups is a small classification task (10k posts), so the papers predict
a tie. My Qwen LoRA was set up the way the "for" papers recommend: all 7 matrices, and 1e-4, 10× the full fine-tune's
1e-5. So if full fine-tuning clearly wins, the next suspects are my other LoRA settings (rank 16, 3 epochs), before
LoRA itself. Sada et al.'s Qwen3-8B LoRA (77.75) was on the attention matrices only, the setup that both QLoRA and
Schulman et al. found falls short.

Tests:
- **Qwen:** the full fine-tune against LoRA's 3-seed mean (78.67), with the decision rule above. Informative only if conjecture 1
  fails.
- **RoBERTa (`LoRA Roberta.ipynb`, run 2026-11-02; the sweep is above, under "Learning-rate sweep"):** the cleaner
  test, because full fine-tuning helps there: frozen + MLP 68.33, fully fine-tuned 72.69 (all 4 runs) or 73.13 (the
  3 `sweep.py` seeds). If LoRA's B matrices stayed at 0, it would be frozen RoBERTa with a trained head, about 68, so
  LoRA should land between about 68 and 73. It gets the same recipe and tuning effort as the full fine-tune (3
  learning rates on dev, then 3 seeds), scored with `summarize.py --final`. Proposed rule: LoRA's 3-seed mean within
  1 point of 73.13 is a tie, more than 2 points below means worse, and in between needs more seeds.

  **Result: a tie, so conjecture 2 holds for RoBERTa.**
  - Setup: r 16, lr 5e-4, chosen on dev from 1e-4 to 1e-3.
  - Test 73.23, 73.34, 73.31: mean **73.29** (std 0.06), vs. the full fine-tune's 73.13 (+0.16).
  - It trains 2.6% of the weights (3.26M, 0.61M of them the head).
  - No saving on cost at this size: 5.1 GB peak vs. 5.2, and about the same time per epoch. At 16 posts × 512
    tokens the activations dominate memory, not the weights' gradients and AdamW state.
  - **Ensembles: LoRA's is better,** 75.07 vs. 73.93 (+1.14; 321 vs. 237 posts right only in one, McNemar
    p = 0.0004). Averaging gained LoRA 1.8 points over its mean, but full fine-tuning only 0.8.
  - LoRA's seeds disagree more: 16.9% of test posts per pair vs. 14.0%, and at least one of the 3 seeds is right on
    80.4% vs. 79.0%. A possible reason, not tested: each seed starts from different random A's, which point the
    first updates in different directions, while full fine-tuning starts every seed from the same weights with only
    the head random.
  - The ensemble comparison was the notebook's secondary check. The decision rule is about the means.

### 3. Reading the results
- **Three seeds now (2026-11-02), and still a tie.** LoRA averages 78.67 against frozen + LR's 78.00 (+0.67). Its
  3-seed ensemble scores 79.73 against the frozen MLP's 10-seed ensemble (78.97; +0.76, p = 0.018). Both are under
  the 1-point bar. Tuning the learning rate didn't change this, and rank didn't matter on RoBERTa ("What the sweep
  found" above). The frozen rows read 2,048 tokens and LoRA 512, though going from 512 to 2,048 changed the frozen
  model by only about 0.1.
- **The puzzle:** fine-tuning gained RoBERTa about 5 points over frozen + LR, and Qwen under 1.
- **Dev runs above test:** 3.4 points for Qwen with LoRA (3-seed means, 82.06 vs. 78.67), 5–6 for RoBERTa. Dev comes from the
  training period, test from later. Compare the full fine-tune's dev accuracy per epoch with LoRA's (80.70, 80.79,
  82.24) before looking at test.

## Qwen3-Embedding-8B: the Colab recipe doesn't fit

- **The arithmetic.** The 4B notebook's recipe costs 16 bytes per trained weight (32-bit weight, gradient, AdamW's two
  averages). Qwen3-Embedding-8B has 7.57B weights (hidden size 4,096, 36 layers): 121 GB before activations. Colab's
  largest GPUs are the 80 GB H100 and A100, and possibly a 96 GB "G4" (RTX PRO 6000; secondary sources only).
- **On one 80 GB GPU,** with a changed optimizer:
  - 8-bit AdamW (bitsandbytes; Dettmers et al. 2022, ICLR: 8-bit optimizer states "maintain 32-bit performance" on
    tasks including GLUE fine-tuning): 10 bytes per weight, about 76 GB before activations. Too tight.
  - Plus freezing the 621M-weight embedding table: about 72 GB plus activations, an estimated 77 GB peak out of about
    84 usable. Tight, and not measured, but the likeliest setup for Colab.
  - bfloat16 weights with stochastic rounding (torchao's `AdamW8bit(bf16_stochastic_round=True)`): about 45 GB, but
    further from the recipe.
- **With more hardware, the recipe runs unchanged:** FSDP or DeepSpeed ZeRO-3 across GPUs (121 GB split among them),
  CPU offload of the optimizer (needs about 120 GB of system RAM, and is slower), or one B200 (180 GB) from a GPU cloud
  that rents single cards.
- **Any optimizer change confounds 4B vs. 8B.** For a clean model-size comparison, rerun the 4B with the same
  optimizer.
- **What companies mostly do instead:** LoRA or QLoRA. LoRA on the 8B in bfloat16 needs about 20 GB.

## Distilling the LoRA Qwen into RoBERTa-base (`Distilled Roberta LoRA Qwen3-Embedding-4B.ipynb`)

Hinton-style knowledge distillation (Hinton et al. 2015):
- **Loss:** the RoBERTa recipe above, with α · CE(labels) + (1 − α) · T² · KL(teacher ‖ student) at temperature T, and
  α = 0.5.
- **Teacher:** the LoRA Qwen3-Embedding-4B above (78.68). Rebuilt from `qwen3-embedding-4b-lora.pt`, it scored the
  same 82.24 on dev.
- **Transfer set:** the 9,987 training posts. The teacher's logits on them, and on dev and test, were computed once and
  cached in `qwen3-embedding-4b-lora-logits.npz`.
- **Temperature, chosen on dev (seed 423):** best-epoch dev accuracy 78.09 (T = 1), 78.36 (T = 2) and 78.27 (T = 4),
  against the plain fine-tune's 78.00, 77.82 and 77.73. T = 2 was kept.

| Setup | Runs | Test acc | Test macro-F1 |
|---|---|---|---|
| Distilled, T = 2 | 3 seeds | 72.94, 73.58, 72.93: mean 73.15 (std 0.37) | 71.80, 72.40, 71.90 |
| Distilled, ensemble of those 3 | 1 | 74.02 | 72.90 |
| Plain fine-tune (RoBERTa recipe, `sweep.py`, same seeds) | 3 seeds | 72.84, 73.11, 73.45: mean 73.13 (std 0.31) | 71.75, 71.69, 72.16 |
| Plain fine-tune, ensemble of those 3 | 1 | 73.93 | 72.71 |

- **No effect: +0.02 points on the 3-seed mean, 0% of the 5.55-point gap to the teacher.**
  - The ensembles differ by 0.09 points: 218 posts are right only for the distilled one and 212 only for the plain one
    (McNemar p = 0.81).
  - The decision rule, fixed before the runs, calls a gain under 1 point no clear effect.
- **Dev overstated it.** Distillation was ahead on dev by 0.24 points at the best epoch (mean of 3 seeds) and about 0.7
  at the last epoch. On test it was level. Dev comes from the training period, test from later.
- **The student barely became more like the teacher.** It picked the teacher's class on 78.05% of the test posts,
  against 77.34% for the plain fine-tune (on dev, 82–83% for both).
- **Likely why: the student only saw the teacher on the teacher's own training posts.** There the teacher is 98.18% right
  and sure of it: its top class averages 0.975 at T = 1 and 0.924 at T = 2. So its soft labels add little to the hard
  ones.
- **That's the low end of the published results.**
  - In Patient KD (Sun et al. 2019, Table 1), vanilla distillation of BERT-base into a 6-layer BERT on the same
    labeled data closed between −6% and 43% of the gap on GLUE tasks (−0.2 points on MNLI).
  - The larger shares came with augmented transfer sets. Tang et al. 2019 used one; TinyBERT's ablation without
    augmentation lost 2 to 21 dev points.
  - Other large shares came from matching intermediate layers. That doesn't carry over to a teacher with a different
    architecture and tokenizer.
- **Cost:** the teacher's pass once (9.0 min, 8.5 GB), then about 8 min per student run (5.2 GB), the same as plain
  fine-tuning.

### An honest teacher: frozen Qwen + LR, out of fold

Why the LoRA teacher failed:
- **Its soft labels were the hard labels.** It memorized its own training posts, the transfer set: 98.18% right, with
  a median 0.006% of the probability off the true class.
- **It is overconfident.** Its confidence exceeds its accuracy by 11–13 points; temperature 2.25 calibrates it.
- **Its edge is on posts the transfer set doesn't cover:** 74% of the net gap is in posts under 100 words, and 89% in
  the test posts least like any training post. Diagnosis and literature: `../../Distillation-lessons.md`.

The new teacher:
- **What it is:** the cached frozen Qwen3-Embedding-4B vectors (2,048 tokens, instruction B), standardized, with LR
  at C = 0.001.
- **Out of fold:** each training post's soft label comes from an LR fitted on the other 9 thread-grouped folds. That's
  79.3% right, with a median 10.8% of the probability off the true class, and calibrated.
- **Dev and test labels:** from an LR fitted on all 9,987 posts (dev 81.79, test 77.72).
- **Settings:** T = 1 (fixed in advance, since the teacher is calibrated) and α = 0.5; the notebook's
  `TEACHER = "frozen_lr_oof"`.

| Setup | Runs | Test acc | Test macro-F1 |
|---|---|---|---|
| Distilled from frozen Qwen + LR, out of fold | 3 seeds | 74.49, 74.90, 74.91: **mean 74.77** (std 0.24) | 73.14, 73.56, 73.60 |
| Distilled from frozen Qwen + LR, ensemble of those 3 | 1 | **75.29** | 73.96 |

- **Distillation helps:**
  - **+1.64 over plain fine-tuning** on the 3-seed mean (73.13), and every distilled seed beats every plain seed.
  - **Ensembles: +1.36** (75.29 vs. 73.93). 303 posts are right only for the distilled one and 203 only for the plain
    one (McNemar p = 1e-5).
  - **Gap closed:** 36% of the gap to this teacher (77.72), and 30% of the original gap to the LoRA Qwen (78.68).
- **It beats the published RoBERTa-large fine-tune** (74.1) with RoBERTa-base.
- **The less accurate teacher taught better.** It scores 77.72 on test against the LoRA teacher's 78.68, yet its soft
  labels carry real uncertainty. That's consistent with Cho & Hariharan 2019 ("more accurate teachers often don't make
  good teachers"). But it doesn't isolate accuracy: the two teachers also differ in memorization, calibration and
  input length (`../../Distillation-lessons.md`, "Why LoRA Qwen taught worse").
- **Signs that the student learned from the teacher, not just from softer labels:**
  - It picked the teacher's class on 82.2% of the test posts, against 78.2% for the plain fine-tune.
  - Its dev loss kept falling to about 0.70, where the other runs' dev loss rose to about 1.4.
- **Controls.** Each repeats the run above (α = 0.5, T = 1, the same 3 seeds) with one change:

  | Setup | Test acc (3 seeds) | Mean (gain) | Macro-F1 | Ensemble |
  |---|---|---|---|---|
  | Label smoothing, no teacher (ε = 0.307: the teacher's average softness, without its information) | 72.81, 73.15, 73.31 | 73.09 (−0.04) | 71.82 | 74.19 |
  | The same LR teacher, in-sample (labeling the posts it was fitted on) | 74.37, 75.05, 74.53 | 74.65 (+1.52) | 73.34 | 75.36 |

  - **Softer labels alone did nothing.** Label smoothing vs. plain: 212 vs. 193 posts, McNemar p = 0.37. The
    out-of-fold teacher beat label smoothing: 265 vs. 184, p = 0.0002. So the gain is the information in the teacher's
    soft labels, not regularization (contrast Yuan et al. 2020).
  - **In-sample did as well as out-of-fold:** 82 vs. 87 posts, p = 0.76. This LR is strongly regularized and only
    88.0% right on its own training posts, so it hadn't memorized them, unlike LoRA Qwen (98.2%).
  - **Agreement with frozen Qwen + LR on test:** plain 78.15%, label smoothing 78.68%, LR in-sample 81.49%,
    out-of-fold 82.20%.
  - **A stronger teacher: frozen Qwen + MLP** (the tuned MLP, out of fold). It came out underconfident, about 30
    points below its accuracy, because it was trained with label smoothing 0.3. So it was temperature-scaled on dev
    first (logits ÷ 0.53).
    - **The teacher:** 0.4–0.5 points more accurate than the LR (test 78.22 vs. 77.72). The two agree on 93.8% of the
      test posts.
    - **The student:** test 73.76, 74.87, 74.33, mean 74.32 (+1.19); macro-F1 72.74; ensemble 75.03.
    - **No better than the LR teacher's student:** 102 vs. 121 posts, p = 0.23.
  - **Still confounded between LoRA Qwen and frozen Qwen + LR:** memorization and calibration vs. what each reads
    (fine-tuned at 512 tokens vs. frozen at 2,048).
- **Next, if pursued:**
  - teacher-labeled new inputs (crops of posts, and the parts of long posts beyond 512 tokens) with this teacher;
  - distilling Qwen's 2,560-d embedding into `<s>`.

## Baseball vs. hockey (2 classes, A2's original task)

`Fine-tuned Roberta 2 Class.ipynb`: the same notebook and RoBERTa recipe, 1,053 training posts, 118 dev posts, 777 test posts (1 point ≈ 8 posts).

| Setup | Runs | Test acc | Test macro-F1 |
|---|---|---|---|
| HF head, RoBERTa recipe (kept epoch 7, dev 93.22) | 1 | 93.18 | 93.17 |

For comparison (`../FrozenResults.md`): frozen RoBERTa `<s>` + LR 91.63, + MLP 92.92 (one seed), frozen Qwen3-Embedding-4B + LR 95.24.
Fine-tuning cut frozen RoBERTa + MLP's error by about 4% here, against about 14% on 20 classes.

## Conclusions

What the results so far and the six papers in `../../papers/` (compared in `../FrozenResults.md`) say about fine-tuning
an encoder for classification, next to a frozen embedding model. Accuracies are on our 7,370 test posts (headers,
footers and quotes removed) unless marked "full posts". Conjecture 1, and conjecture 2 for Qwen, stay open until the
full fine-tune runs (conjecture 2 held for RoBERTa); only the part of point 3 about fine-tuning Qwen depends on them.

| Approach | Encoder weights trained | Test acc | Cost (ours: 5070 Ti) |
|---|---|---|---|
| Naive Bayes on word counts, assignment defaults | no encoder | 64.08 | seconds on CPU |
| Complement NB on TF-IDF bigrams ([Sada et al.](https://arxiv.org/abs/2609.13185)) | no encoder | ≈72.7 (point 2) | seconds on CPU (theirs) |
| Frozen RoBERTa-base `<s>` + LR / + tuned MLP | none | 67.23 / 68.33 | 4–6 min to encode, ~2 s per LR fit |
| Fine-tuned RoBERTa-base: mean of 3 seeds / their ensemble | all 125M | 73.13 / 73.93 | 8 min per run, 5.2 GB |
| LoRA RoBERTa-base (r 16, lr 5e-4): mean of 3 seeds / ensemble | 2.65M LoRA weights | 73.29 / 75.07 | 8 min per run, 5.1 GB |
| RoBERTa-base distilled from LoRA Qwen3-Embedding-4B (T = 2): mean of 3 seeds / ensemble | all 125M | 73.15 / 74.02 | 9 min teacher pass once, then 8 min per run |
| RoBERTa-base distilled from frozen Qwen3-Embedding-4B + LR, out of fold: mean of 3 seeds / ensemble | all 125M | **74.77 / 75.29** | 1 min of LR fits on cached vectors, then 8 min per run |
| Frozen Qwen3-Embedding-0.6B + LR | none | 73.42 | 7 min to encode |
| LoRA Qwen3-8B chat model (Sada et al.) | LoRA, attention only | 77.75 | not run here |
| Frozen Qwen3-Embedding-4B + LR / + MLP, 10-seed ensemble | none | 78.00 / 78.97 | encode once: 18 min at 512 tokens, more at 2,048 |
| LoRA Qwen3-Embedding-4B (lr 1e-4): mean of 3 seeds / ensemble | 33M LoRA weights | 78.67 / 79.73 | 49 min per run, 9.6 GB |

**1. Fine-tuning RoBERTa gains about 5 points by fixing its frozen vectors.**
- Frozen + LR 67.23, + tuned MLP 68.33 → fine-tuned 72.69 (all 4 RoBERTa-recipe runs) to 73.13 (the 3 `sweep.py`
  seeds). On baseball vs. hockey, 92.92 → 93.18: frozen features already get an easy task nearly right, so there's
  less to fix.
- LoRA gets the same gain while training 2.6% of the weights: 73.29 (3 seeds) vs. 73.13 fully fine-tuned, a tie
  (conjecture 2). Its 3-seed ensemble is better, 75.07 vs. 73.93 (p = 0.0004), because its seeds disagree more.
- No frozen vector from an encoder trained only to predict masked words did well: RoBERTa's `<s>` (67.23) or its
  average over all tokens (66.13), and ModernBERT's, even though it's from 2024 (`[CLS]` 57.26, average 64.93; all
  with LR). That fits Conjecture 1's explanation: masked-word prediction never asks for a summary of the post, so the
  vectors aren't one until the classification loss asks for it (pre-train, then fine-tune: Lecture 11).
- Training an encoder to embed fixes the same weakness without this task's labels. In
  [NewsEmbed](https://arxiv.org/abs/2106.00590) (Liu et al. 2021), BERT-base encoders scored 64.5 (SBERT, trained on
  NLI sentence pairs) and 73.9 (NewsEmbed, trained contrastively on news-article triplets and topic labels) on 20NG,
  with frozen vectors + LR. The paper doesn't say how it preprocessed 20NG, but it's the same for both.

**2. Our fine-tune is as good as published ones, and barely ahead of a tuned Naive Bayes.**
- The closest published fine-tune is RoBERTa-large at 74.1
  ([Edwards & Camacho-Collados 2024](https://arxiv.org/abs/2403.17661); preprocessing unstated). Ours is a third its
  size and within a point. A larger encoder doesn't look like a big lever on this dataset: on full posts, BERT-large
  scored below BERT-base (85.83 vs. 87.21, [Scherp et al.](https://arxiv.org/abs/2204.03954)).
- Sada et al.'s fine-tuned DistilBERT (67.2) is 6 points lower, but
  [their code](https://github.com/groundsada/llms-or-naive-bayes) (`src/eval/ncurve.py`) reads only the first 128
  tokens, trains 2 epochs on 10,000 posts, and scores the first 2,000 test posts. On full posts DistilBERT and
  RoBERTa-base tie (86.9 vs. 86.8, Scherp et al.), so most of that gap is the recipe. Scherp et al. found that
  published 20NG numbers for the same model differ by 2–3 points (more than 3 for RoBERTa-base), and call BERT
  baselines "often undertuned".
- Their Complement Naive Bayes on TF-IDF bigrams scores 71.2 on all 7,532 test posts. Without the 162 empty posts,
  which it can only guess, that's about 72.7 (estimated). So a single fine-tuned RoBERTa (72.69–73.13) is level with
  it to half a point ahead, inside the seed noise, and the ensemble (73.93) is 1.2 ahead. The assignment's Naive
  Bayes (64.08: word counts, default settings) undersells classical models by 8–9 points (Lecture 5).
- Bag-of-words is this strong because each newsgroup has its own vocabulary, so the topics are close to linearly
  separable on word counts alone ([Arras et al. 2017](https://arxiv.org/abs/1612.07843): BoW + SVM 80.1, a CNN
  80.2). Sada et al.'s advice: fine-tune DistilBERT/RoBERTa only "if a 1–3 pp gain justifies GPU use".

**3. A frozen embedding model beat the fine-tuned encoder by 5 points.**
- Frozen Qwen3-Embedding-4B + LR scored 78.00 against fine-tuned RoBERTa's 73.13, without training a single encoder
  weight. Even the frozen 0.6B (73.42) ties fine-tuned RoBERTa, and on baseball vs. hockey frozen 4B + LR (95.24)
  beats it too (93.18). A better starting representation beat fine-tuning a weaker one.
- Qwen3-Embedding isn't an encoder-decoder. It's the decoder-only Qwen3 LLM (causal attention, Lecture 10) without
  its next-word layer, and the embedding is the last token's final hidden state. "Encoder" describes its job (text
  in, one vector out), not its architecture. An encoder-decoder (Lecture 9) has a separate encoder that reads the
  input and a decoder that attends to it; T5 is one.
- It was built for this use. Its contrastive training (about 170M text pairs, with classification among the tasks;
  Conjecture 1) makes the last token a summary of the text, the job fine-tuning had to teach RoBERTa's `<s>`. And
  [MTEB](https://arxiv.org/abs/2210.07316), the benchmark it reports on, scores classification exactly our way:
  logistic regression on frozen embeddings.
- Fine-tuning it has added little so far. LoRA's 3 seeds average 78.67, against frozen 78.00 (LR) and 78.80 (MLP,
  mean of 10 seeds). Its 3-seed ensemble leads the frozen MLP ensemble by 0.76 (79.73 vs. 78.97, p = 0.018), for
  3 × 49 minutes of training against one encoding pass. Tuning LoRA's learning rate didn't change that, and rank
  didn't matter on RoBERTa. And fine-tuning a general LLM only got to where the frozen embedding LLM starts: Sada et al.'s LoRA Qwen3-8B chat
  model scored 77.75 on the same posts.
- The instruction is task adaptation for free: the newsgroup instruction gained 0.85 test points (p = 0.004), more
  than reading 4× the tokens did (`../FrozenResults.md`).

**4. Fine-tuned LLMs beat fine-tuned encoders by 4–5 points on 20NG, with or without the metadata, but nobody held
size fixed.**
- Without metadata: LoRA Qwen3-Embedding-4B vs. fine-tuned RoBERTa-base, 78.67 vs. 73.13 (+5.5, 3-seed means). Sada et al.'s LoRA
  Qwen3-8B vs. the same RoBERTa: 77.75 vs. 73.13 (+4.6).
- Full posts ([Zhao et al. 2025](https://arxiv.org/abs/2412.08587), preprint): fully fine-tuned Llama-3-8B 91.6 vs.
  RoBERTa-base 86.7 (+4.9) and RoBERTa-large 88.1 (+3.5). The best 70B run added only 0.3 more (91.9), on 8 A100s for
  up to 3 days.
- Scherp et al. are more cautious across many benchmarks: an ensemble of fine-tuned Llama-2 models was needed to match
  or slightly beat an encoder on two datasets, and a fine-tuned 7B Llama with its causal mask removed still lost to
  BERT-base on hierarchical classification. They suspect the causal mask, which lets each token see only earlier
  ones, is a handicap for classification.
- The LLMs have tens to hundreds of times the encoders' weights and far more pre-training text, which is Zhao et al.'s
  own explanation for their gap. Our winner is causal, but it reads the one token that has seen the whole post, after
  training that made that token a summary. So on this task, scale and embedding training outweighed attention
  direction; these results can't say which attention is better at equal size.

**5. Prompting an LLM is the weak way to use one when there are labels.**
- Flan-T5 zero-shot 51.0 and LLaMA-2-7B one-shot 37.6, vs. fine-tuned RoBERTa-large 74.1 (Edwards &
  Camacho-Collados; the gap grew with the number of classes). Zero-shot Qwen3.6-27B tied Complement NB (71.8 each, on
  the same 500 posts) and the 397B came last (Sada et al.). On full posts, zero-shot GPT-4 scored 80.4 vs. fine-tuned
  RoBERTa-base's 86.7 (Zhao et al.).
- Only a 1-trillion-parameter model (Kimi-K2, zero-shot) matched frozen Qwen3-Embedding-4B: 77.33 vs. 77.53 on the
  same 494 posts.
- The papers' advice: with enough labels, fine-tune a small model; with none, prompt an LLM (Scherp et al., Edwards &
  Camacho-Collados; Sada et al. say the same with Naive Bayes in place of fine-tuning). These results add a third
  option, which beat both here: an LLM trained into an embedding model, frozen, with LR on top.

**6. Frozen + LR is also cheaper to get right.**
- A fine-tuned score moves with the seed (71.56–72.84 for 3 runs of one notebook), the recipe (under a point), and
  the epoch kept (71.37 when a noisy dev pick kept epoch 3). Each comparison needs several runs at 8 minutes each.
- An LR refit on cached vectors varies by 0.1–0.2 (solver noise) and takes about 2 s, so a CV search over hundreds of
  settings is cheap.

### When I'd still fine-tune an encoder
- **When each prediction must be cheap.** RoBERTa-base has 1/32 of Qwen3-Embedding-4B's weights. Sada et al. measured,
  at batch 1 on an RTX 3090: DistilBERT 5.4 ms per post, their LoRA Qwen3-8B classifier 97 ms, Complement NB 0.14 ms
  on CPU. Our frozen 4B took 7 min 20 s for the 7,370 test posts, about 60 ms each (batch 16, unsorted).
- **When the label isn't about what the text means or is about.** Embedding training decides what the vector keeps:
  NewsEmbed's cost it 1–4 points on XTREME's token-level and reasoning tasks. A model trained on topic and meaning may
  drop what a task about style, tone or authorship needs, while fine-tuning trains the vector on the task's own
  labels. 20NG is a topic task, so these results can't test this.
- **Otherwise, try the cheap options first:** a tuned TF-IDF + Naive Bayes or LR, and a frozen embedding model + LR.
  Fine-tune only if they leave a gap worth the GPU time.

### Caveats
- **Contamination inflates LLM results on public benchmarks.** Flan-T5 nearly matched fine-tuned models only on AG
  News and IMDB, which were in its instruction data (Edwards & Camacho-Collados). Sada et al.'s zero-shot LLM led
  Naive Bayes on Amazon reviews (97.8 vs. 88.2) but lost on a less public sentiment set (73.0 vs. 81.7). Whether
  Qwen3-Embedding was trained on 20NG's labels is unknown ("Was Qwen trained on 20 Newsgroups?" above); if it was,
  point 3 isn't a like-for-like comparison.
- **One kind of task.** Full 20NG and baseball vs. hockey are both topic classification, where keywords carry most of
  the signal.
- **Pending run.** The full fine-tune of Qwen3-Embedding-4B hasn't run.
- **Test was looked at during the Qwen sweep.** The lr 2e-4 follow-up was chosen from a test score. It's reported as
  such, and the LoRA rows above use the protocol's 1e-4.
- **Unequal sizes.** The fine-tuned encoder is base size (125M). A fine-tuned large encoder (355–395M) would narrow
  the size gap, but published numbers (74.1 for RoBERTa-large) suggest it wouldn't close the 5 points.
