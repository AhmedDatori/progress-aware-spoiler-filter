# Reproduce the results

There are three levels. Pick the one you need.

| Level | What you get | Needs | Time |
|---|---|---|---|
| A. Run the demo | the full filter on any comment | 6 GB GPU (or CPU, slower) | 5 min |
| B. Check every number | all tables and figures from saved model scores | CPU | 10 min |
| C. Train everything | every model from scratch | a 24 GB GPU (Colab L4/A100), or 6 GB with patience | about 8 GPU hours |

## Setup

```bash
git clone https://github.com/AhmedDatori/progress-aware-spoiler-filter.git
cd progress-aware-spoiler-filter
uv sync                 # Python 3.12 and the exact package versions from uv.lock
bash src/download.sh    # dataset into data/raw/ (from Hugging Face, CC BY-SA 3.0)
```

All scripts run from `src/` with `uv run python <script>.py`. They read and write relative to the project folder
(set `SM_ROOT` to use another folder). For training on a small GPU, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## A. Run the demo

```bash
bash src/download_models.sh        # v6_bi, v6_rr48_s2, relevance_ft_base into models/ (fp16, ≈0.9 GB)
cd src
uv run python demo.py --examples
uv run python demo.py legend_of_korra 20 "Korra finally beat Amon"
```

`demo.html` (the web demo) needs nothing: it reads saved outputs. To rebuild its data: `uv run python make_demo_data.py`.

## B. Check every number from saved scores

```bash
bash src/download_results.sh       # results/scores/ (every model's scores on val and test) and the filter simulations
cd src
uv run python -c "from common import *; print(evaluate(load_split('test'), load_scores('v6_rr48_s2','test'), 'v6'))"
uv run python combine_rerank.py v6_first v6_rr48_s2 v6_rr48_s2      # the K/M/a grid on validation, then test
uv run python error_analysis.py v6_rr48_s2
uv run python figures.py
```

`filter_eval.py` also runs from saved scores, but it loads the plot classifier and a small embedding model, so a GPU helps.

## C. Train everything

Times are on a Colab L4 (24 GB) unless noted. Each step writes `results/scores/<name>__{val,test}.npy` and a row in
`results/matching_results.csv`.

```bash
cd src
# v0, v1: no training
uv run python bm25.py val && uv run python bm25.py test
uv run python zeroshot.py test                       # 5 ready-made models, ~5 min
uv run python zeroshot.py val nomic_modernbert

# v2, v3: fine-tuned bi-encoder + BM25 mix
uv run python train_biencoder.py 24000               # ~15 min (24 min on an RTX 4050)
uv run python fuse.py ft_nomic_24k bm25_chunks       # weight chosen on val

# plot classifier (~3 min)
uv run python relevance.py nomic-ai/modernbert-embed-base relevance_ft_base 3e-5

# v6 stage 1 (~40 min)
uv run python train_biencoder.py 24000 v6_bi --max-len 384 --negs 3 --batch 128 --mini-batch 64 --exclude-shows val
uv run python fuse.py v6_bi bm25_chunks v6_first

# v6 stage 2: reranker (~2.5 h on L4; on a 6 GB GPU add --micro 1, ~2.6 h on an RTX 4050)
for s in 0 1 2; do uv run python train_reranker.py v6_bi v6_first v6_rr48_s$s --n 48000 --seed $s; done
uv run python combine_rerank.py v6_first v6_rr48_s2 v6_rr48_s2
uv run python train_reranker.py v6_bi v6_first gte_rr_zeroshot --no-train      # baseline: reranker not trained
uv run python combine_rerank.py v6_first v6_gte_zeroshot gte_rr_zeroshot

# full filter, speed, errors
uv run python filter_eval.py v6_rr48_s2 relevance_ft_base
uv run python speed_v6.py v6_bi v6_rr48_s2 15 4 0.3
uv run python error_analysis.py v6_rr48_s2
uv run python relevance_match.py                     # F2 (no gain)
uv run python figures.py
```

The other experiments (v4, v5, Qwen3 models, other rerankers) are in the notebook, sections 13–14, with their exact commands.

## The notebook

`Spoiler_Matching.ipynb` tells the whole story in order and runs each script. It was made for Google Colab with the project
folder on Google Drive. To use it elsewhere, change `PROJECT` in cell 1.1. The run switches in cell 1.3 are off by default,
so a first run only reads saved results (download them first, level B).

## Expected variation

Training is seeded, but GPU kernels are not fully deterministic: training the same model twice changed test MRR by about
0.002. The three v6 seeds gave 0.637, 0.639 and 0.635.
