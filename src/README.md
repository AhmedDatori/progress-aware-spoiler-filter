# Code

One script per step. Each file starts with a docstring that explains what it does, why, and how to run it.
Run from this folder: `uv run python <script>.py ...`.

| File | Step | What it does |
|---|---|---|
| `common.py` | shared | Paths, data loading, chunking summaries into 200-word windows, MRR / top-k evaluation, saving scores. |
| `download.sh` | data | Downloads the dataset from Hugging Face into `data/raw/`. |
| `download_models.sh` | models | Downloads our 3 fine-tuned models from the GitHub release into `models/`. |
| `download_results.sh` | results | Downloads all saved model scores and filter simulations from the GitHub release. |
| `bm25.py` | v0 | BM25 word matching over whole summaries and chunks. |
| `zeroshot.py` | v1 | Ready-made embedding models (MiniLM, BGE, GTE, nomic, bge-m3, Qwen3): cosine to chunks, best chunk per episode. |
| `train_biencoder.py` | v2, v4, v6 stage 1 | Fine-tunes the bi-encoder with mined hard negatives and a contrastive loss. |
| `fuse.py` | v3, v6 stage 1 | Mixes two score sets with z-scores: z(dense) + w·z(BM25), w chosen on validation. |
| `ensemble.py` | v5 | Fixed-weight mix of several matchers. |
| `rerank.py` | test | Ready-made cross-encoder rerankers on the top 10 episodes. |
| `train_reranker.py` | v6 stage 2 | Fine-tunes the cross-encoder reranker (listwise, 1 right + 7 hard wrong chunks), then scores val and test. |
| `combine_rerank.py` | v6 | Final score z(rerank) + a·z(stage 1); picks K, M and a on validation. |
| `relevance.py` | filter | Fine-tunes the "is it about the story?" classifier. |
| `relevance_match.py` | filter (F2) | Adds matcher similarity as features to the classifier (no gain). |
| `filter_eval.py` | filter | Turns scores into probabilities and simulates readers at every episode; compares with a normal detector. |
| `speed.py`, `speed_v6.py` | speed | Times the fast model, v6 and an estimate for the paper's Longformer on the same GPU. |
| `error_analysis.py` | errors | Where v6 fails: per show, by length, by distance; joins the 60 hand labels. |
| `demo.py` | demo | The full filter on one comment, from the command line. |
| `make_demo_data.py` | demo | Runs the filter on 200 test comments for the web demo. |
| `demo_page.html`, `build_demo_page.py` | demo | The web demo's template, and the script that puts the data in and writes `docs/index.html`. |
| `figures.py` | report | The 3 figures in `results/figures/`. |

Model and score names used across scripts: `v6_bi` (stage 1 bi-encoder), `v6_first` (stage 1 scores = bi-encoder + BM25),
`v6_rr48_s2` (the released reranker: 48k comments, seed 2), `relevance_ft_base` (plot classifier).
