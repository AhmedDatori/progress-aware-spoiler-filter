# Experiments

Every version and experiment, in the order we ran them, with its test result and what we decided from it.
Test = 3,105 comments from 4 shows never seen in training. The paper's best (Longformer) is **0.611 MRR**.
All rows come from `results/matching_results.csv` and `results/filter_results.csv`; the notebook
(`Spoiler_Matching.ipynb`) shows the code and output of each one.

## Versions of the episode matcher

| Version | What changed | Test MRR | Decision |
|---|---|---|---|
| v0 | BM25 word matching on summary chunks | 0.363 | Reproduces the paper's BM25 (0.368): our data and metric are right. |
| v1 | Ready-made embedding models, no training (best: nomic ModernBERT embed on chunks) | 0.476 | Already above 3 of the paper's 5 trained models. Use this model as the base. |
| v2 | v1 fine-tuned on 24k comments with hard negatives (24 min on a laptop GPU) | 0.545 | Training on the task matters most. |
| v3 | v2 + BM25, z-score mix, weight 0.3 on validation | 0.559 | BM25 adds names and rare words. Keep the mix. |
| v4 | New recipe: 384 tokens, 3 hard negatives, batch 128, all 72k comments, noisy comments removed; + BM25 | 0.560 | +0.011 on validation, nothing on test → found the validation trap (below). Our **fast model**. |
| v5 | v4 + Qwen3-Embedding-8B (no training) + BM25, fixed weights | 0.598 | Close to the paper, but needs an 8B model. Rejected: the goal was small and fast. |
| **v6** | Two stages: bi-encoder + BM25 keep 15 episodes, a fine-tuned reranker reads 4 chunks each; validation shows left out of all training | **0.637 ± 0.002** | **Beats the paper with 298M.** Main model. |

The first laptop run of v2/v3 gave 0.544/0.557; the Colab re-run gave 0.545/0.559. Training twice moves MRR by about 0.002.

## v6 in detail

| Setting | Test MRR | Validation MRR |
|---|---|---|
| Stage 1 alone: bi-encoder (validation shows left out) | 0.544 | 0.554 |
| Stage 1: bi-encoder + BM25 | 0.556 | 0.573 |
| + reranker, **not** fine-tuned | 0.600 | 0.610 |
| + fine-tuned reranker, 24k comments, seed 0 | 0.636 | 0.639 |
| + fine-tuned reranker, 48k comments, seeds 0 / 1 / 2 | **0.637 / 0.639 / 0.635** | 0.641 / 0.629 / 0.634 |
| 3 rerankers averaged (596M, over the size goal) | 0.641 | 0.639 |

- Every single run is above 0.611. The 95% bootstrap range of seed 0 on test is 0.624–0.650.
- Fine-tuning the reranker adds +0.037. A ready-made reranker is not enough.
- 24k or 48k training comments for the reranker: no real difference.
- Validation now predicts test (0.634 vs 0.635 for seed 2).
- Grid on validation: more episodes (*K*) and more chunks (*M*) help, and stage 1 still counts (weight 1.0 is best).
  Lighter settings trade accuracy for speed: *K*=10, *M*=2 gives 0.620; *K*=5, *M*=1 gives 0.591 in 2 minutes.

## Other models we tried

| Experiment | Test MRR | Note |
|---|---|---|
| all-MiniLM-L6-v2, no training | 0.423 | small, old |
| bge-base-en-v1.5, no training | 0.369 | |
| gte-modernbert-base, no training | 0.448 | |
| bge-m3, no training | 0.458 | |
| Qwen3-Embedding 0.6B / 4B / 8B, no training | 0.493 / 0.534 / 0.557 | size helps, slowly; 8B = our 149M fine-tuned model |
| Qwen3-Embedding-8B + BM25 | 0.569 | |
| Qwen3-Embedding-0.6B fine-tuned (v2 recipe) + BM25 | 0.562 | +0.003 over nomic, 4× the size |
| Whole summary instead of chunks (long-context models) | always lower | e.g. nomic 0.425 vs 0.476 with chunks |
| bge-reranker-v2-m3 on top 10, no training | 0.512 | **hurts**: it judges "is this passage relevant", not "which episode" |
| gte-reranker on top 10, no training | 0.560 | +0.001 for 41 min; dropped until fine-tuned (→ v6) |
| Qwen3-Reranker-0.6B | – | over 30 min on an A100, stopped |

## The plot classifier ("is it about the story?")

| Classifier | Test-show AUC |
|---|---|
| Frozen ModernBERT embed + logistic regression | 0.855 |
| ModernBERT-large, fine-tuned | 0.870 |
| **ModernBERT embed, fine-tuned (used)** | **0.884** |
| + matcher similarity as extra features (`relevance_match.py`) | 0.884 (no gain) |

## The full filter

Spoilers hidden when 10% of safe comments are hidden (206,968 simulated reader–comment pairs):

| Matcher used by the filter | Plot classifier | Progress-aware | Normal detector | Perfect plot labels |
|---|---|---|---|---|
| v3 | frozen + LR | 68.0% | 37.2% | 97.8% |
| v3 | fine-tuned | 70.7% | 44.0% | 97.8% |
| v5 | fine-tuned | 71.1% | 44.0% | 98.2% |
| **v6** | **fine-tuned** | **74.7%** | **44.0%** | **97.5%** |

At 5% safe hidden: v6 64.7%, normal detector 27.8%. AUC: v6 0.914, normal detector 0.826. The normal detector hides 74.3%
of story comments about episodes the reader has already seen; v6 hides 4.8%.

**Takeaway:** better matching helps a little (68% → 75%). A perfect plot classifier would help much more (→ 98%).

## Speed (laptop RTX 4050 6 GB, all 3,105 test comments)

| System | Time | Comments per second |
|---|---|---|
| Fast model (bi-encoder) | 12 s | 262 |
| v6, *K*=5, *M*=1 | 127 s | 24 |
| v6, *K*=15, *M*=4 | 1,256 s (21 min) | 2.5 |
| Longformer, estimate | ≈54,500 s (15 h) | 0.06 |

Summary chunks are encoded once (7–12 s) and cached. The Longformer estimate times single (comment, full summary)
pairs and multiplies by the 120,963 pairs it must read; a real system would batch, so the speed-ups (≈43× for v6) are rough.
v6 is about 100× slower than the fast model, because the reranker reads 60 pairs per comment.

## Error analysis

`error_analysis.py` on v6 (seed 2, test MRR 0.635).

| | |
|---|---|
| Right episode first / top 3 / top 5 | 51% / 72% / 80% |
| MRR by show | Arcane 0.71 (9 episodes) · Korra 0.67 (27) · Kaguya-sama 0.61 (19) · Attack on Titan 0.59 (62) |
| MRR by comment length | 1–10 words 0.51 · 11–20 0.62 · 21–40 0.66 · 41+ 0.69 |
| Wrong top guess | 1 episode away 20% · 2 away 12% · 3–5 away 18% · 6+ away 51%; earlier than the right episode 63% |

60 random wrong predictions, sorted by hand (`results/error_labels_by_hand.csv`):

| Kind | Count | Example (show, right → guessed episode) |
|---|---|---|
| Next/previous or very similar episode | 16 | "Zeke came in ready to shatter Erens mind and left with his world shattered instead" (AoT, 79 → 78) |
| Too short or vague | 14 | "SHE CALLED HIM TWINKLETOES." (Korra, 9 → 41) |
| Small detail not in the summary | 9 | "Interesting Pentakill tattoo around the 17 minute mark." (Arcane, 4 → 7) |
| General reaction, little story | 9 | "Finally an exciting episode! Finally action! …" (AoT, 85 → 54) |
| About older episodes or the whole season | 7 | "I'd like to point out that this is the first time we've seen the asphyxiation technique…" (Korra, 36 → 41) |
| A guess or question about the future | 5 | "Someone please make a poll how many believe Eren … or Zeke …" (AoT, 80 → 76) |

Most of these are hard for a person too: neighbouring episodes continue one story, and short or off-summary comments carry
little information.

## Lessons

1. **Validation must be honest.** The validation shows were also training shows, so validation rewarded memorising them.
   v4's extra data helped validation (+0.011) but not test. From v6 on, validation shows are left out of training.
2. **Rerankers must be trained on the task.** Off the shelf they judge general relevance and can make things worse.
3. **Train small rather than run big.** An untrained 8B model matched our trained 149M model.
4. **Cleaning the automatic labels did not help** (removing comments the classifier calls "not plot", about 12%).
5. **The filter's bottleneck is the plot classifier**, not the matcher.

## Lost along the way

The weights of three v6 rerankers (24k seed 0, 48k seeds 0 and 1) were lost when a Colab machine was reclaimed. Their
score files were saved, so their results above are real, but only seed 2 (`v6_rr48_s2`) exists as a model. It is the
released model and the one used by the demo.
