# Method

This page explains every part of the system: the task, the data, the two models of the matcher, the plot classifier,
the filter, and how we measure. Code references point to `src/`.

## 1. The task

For a reader who has watched episodes 1 … *k* of a show, decide for every comment: **hide or show**.

A comment is a spoiler for this reader if

1. it is about the story (not "this episode was fun"), **and**
2. it is about an episode after *k*.

So the system needs two answers per comment:

| Question | Name | Output |
|---|---|---|
| Is it about the story? | plot classification | P(plot) |
| Which episode is it about? | spoiler matching (Tran et al., 2023) | P(episode) for every episode of the show |

## 2. Data

[bobotran/spoiler-matching](https://huggingface.co/datasets/bobotran/spoiler-matching), from Tran, Xu & McAuley (EMNLP 2023).
Reddit comments from episode discussion threads, and a written summary of every episode (median 1,230 words).

| Split | Comments | Shows | Labels |
|---|---|---|---|
| train | 204,475 | 9 | automatic: the episode thread the comment was posted in |
| validation | 2,787 | 4 | by hand |
| **test** | **3,105** | **4, never in training** | by hand |
| plot / not plot | 11,032 | 8 | by hand: is the comment about the story? |

Test shows: Attack on Titan (62 episodes), The Legend of Korra (27), Kaguya-sama: Love is War (19), Arcane (9).

**A trap in the data.** The 4 validation shows are also among the 9 training shows. A model trained on them looks better
on validation than on new shows. From v6 on we **leave the validation shows out of all training**, so validation predicts test
(0.634 vs 0.635).

## 3. Summaries are long: chunks and MaxP

Most encoder models read 512 tokens or less, and a summary is about 1,600 tokens. We cut every summary into overlapping
windows of **200 words with a step of 150** (`common.chunk_text`). An episode's score is the score of its **best**
chunk ("MaxP", Dai & Callan, 2019): a comment usually talks about one scene, so one matching piece is enough.

## 4. Stage 1: bi-encoder + BM25 (all episodes)

**Bi-encoder.** One model turns a text into a vector. Comment and chunk are encoded separately and compared with
cosine similarity. Chunk vectors are computed once and cached, so scoring a new comment is very fast.

- Base: `nomic-ai/modernbert-embed-base` (149M, ModernBERT architecture). It expects the prefixes `search_query: ` and `search_document: `.
- Fine-tuning (`train_biencoder.py`, model `v6_bi`): 24,000 training comments (validation shows excluded), up to 384 tokens.
  Each example is (comment, its right chunk, 3 hard negatives). The right chunk is the chunk of the correct episode that the
  untrained model finds most similar. Hard negatives are the best chunk of each of the 3 most similar *wrong* episodes of the same show. At most 15,000 comments per show, so one show does not dominate.
- Loss: `CachedMultipleNegativesRankingLoss` (contrastive). Each comment must pick its right chunk among all chunks in
  the batch (128 comments → 512 candidates). The cached version lets a 6 GB GPU use a big batch.
- 1 epoch, learning rate 2e-5, 10% warm-up.

**BM25** (`bm25.py`) scores word overlap. It finds rare exact words, names like "Powder" or "Tarrlok", that the embedding model can blur.

**Mixing** (`fuse.py`). Cosine scores are around 0–1, BM25 scores around 0–40. For each comment we z-normalise each
model's scores over the show's episodes (subtract the mean, divide by the standard deviation) and add:

    stage1 = z(bi-encoder) + 0.3 · z(BM25)          (0.3 chosen on validation)

Stage 1 keeps the **top 15 episodes**.

## 5. Stage 2: fine-tuned cross-encoder reranker (top 15 episodes)

A bi-encoder squeezes each text into one vector before comparing, so details get lost. A **cross-encoder** reads
comment and chunk as one input, so every comment word can look at every summary word ("the guy with the scar" ↔ "Zuko").
It is slower, so it only reads the **4 best chunks** (by bi-encoder similarity) of each of the 15 kept episodes: 60 pairs per comment.

- Base: `Alibaba-NLP/gte-reranker-modernbert-base` (149M).
- Training data (`train_reranker.py`, model `v6_rr48_s2`): 48,000 training comments, validation shows excluded, and not
  the comments the bi-encoder was trained on (on those, its "hard" negatives are too easy). Each example is a group of
  8 chunks: the right one, plus the best chunk of each of the 7 highest-ranked *wrong* episodes from stage 1.
  So the reranker learns to fix exactly the mistakes stage 1 makes.
- Loss: softmax over the 8 scores, cross-entropy on the right one (listwise).
- 1 epoch, learning rate 2e-5, 16 groups per step, 384 tokens. 2.6 h on a laptop RTX 4050 with gradient accumulation.

**Final score** (`combine_rerank.py`), over the 15 kept episodes:

    final = z(best reranker score of the episode's chunks) + 1.0 · z(stage1)

Episodes outside the top 15 stay below, in stage-1 order. *K* = 15 episodes, *M* = 4 chunks and the weight 1.0 were picked on
validation from a grid (*K* ∈ {5, 10, 15}, *M* ∈ {1, 2, 4}, weight ∈ {0, 0.25, 0.5, 1, 2}). Total size: 149M + 149M = **298M**.

## 6. Plot classifier

`relevance.py`, model `relevance_ft_base`: the same ModernBERT embedding model with a 2-class head, fine-tuned on the
hand-labelled plot / not-plot comments of the 4 non-test shows (85% train, 15% dev to pick the epoch). 3 epochs,
learning rate 3e-5, 128 tokens. Test-show AUC 0.884. A frozen model with logistic regression gets 0.855,
ModernBERT-large 0.870, and adding matcher similarity as a feature does not help (0.884, `relevance_match.py`).

## 7. The filter

`filter_eval.py`. Matcher scores become probabilities with a softmax with temperature *T*:

    P(episode i) = softmax(final / T)            T = 1.2512, fitted on validation by maximum likelihood

For a reader who watched up to episode *k*:

    P(spoiler) = P(plot) × Σ P(episode i)  over episodes i after k          hide if > 0.5

The **normal detector** we compare with hides every comment with P(plot) > 0.5, whatever the reader has watched. It
uses the same classifier, so the comparison isolates the value of knowing the reader's position.

## 8. How we measure

**MRR** (mean reciprocal rank, the paper's metric). For each comment, rank the show's episodes. If the right episode is at
rank *r*, the comment scores 1/*r* (1st → 1, 2nd → 0.5, 4th → 0.25). MRR is the average. We also report top-1 and top-3 accuracy.

**Filter simulation.** For each test show, a reader at every episode position, asked about every hand-labelled test comment
(plot and not plot): 206,968 (reader, comment) pairs, 44% spoilers. Main metric: the share of spoilers hidden when each system
is tuned to wrongly hide 10% (and 5%) of safe comments, so every system gets the same false-alarm budget. We also report AUC.

**Rules we followed**

- Every setting (mixing weights, *K*, *M*, temperature, model choice) is chosen on validation only. Test is scored once at the end.
- The main model is trained with 3 seeds (0.637 / 0.639 / 0.635). Training the same model twice moves MRR by about 0.002, so
  smaller differences are treated as noise.
- 95% bootstrap ranges (2,000 resamples of test comments) for single runs.

## References

- Tran, B., Xu, Z., & McAuley, J. (2023). Spoiler Detection as Semantic Text Matching. EMNLP 2023.
- Dai, Z., & Callan, J. (2019). Deeper Text Understanding for IR with Contextual Neural Language Modeling. SIGIR 2019.
- Nussbaum, Z., et al. (2024). Nomic Embed: Training a Reproducible Long Context Text Embedder. arXiv:2402.01613.
- Warner, B., et al. (2024). ModernBERT: Smarter, Better, Faster, Longer. arXiv:2412.13663.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.
- Wan, M., et al. (2019). Fine-Grained Spoiler Detection from Large-Scale Review Corpora. ACL 2019.
- Zhang, Y., et al. (2025). Qwen3 Embedding. arXiv:2506.05176.
