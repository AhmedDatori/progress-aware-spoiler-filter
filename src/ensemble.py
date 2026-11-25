"""Mix several matchers with fixed weights: score = sum_i w_i * z(score_i), per comment.

z() turns each model's scores for one comment into "how many standard deviations above this comment's average",
so models with different score scales can be added. No weights are tuned here: equal weight for the two neural
models, 0.3 for BM25 (the weight every single model picked on val in fuse.py).

Usage: python ensemble.py <name> <tag:weight> [<tag:weight> ...]
  python ensemble.py v4_72k+qwen3_emb_8b+bm25 v4_72k:1 qwen3_emb_8b_chunks:1 bm25_chunks:0.3
"""
import sys

import numpy as np

from common import append_result, evaluate, load_scores, load_split, save_scores
from fuse import z

name = sys.argv[1]
parts = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in sys.argv[2:]]
for split in ["val", "test"]:
    df = load_split(split)
    all_scores = [(load_scores(t, split), w) for t, w in parts]
    mixed = [sum(w * z(np.asarray(s[i], float)) for s, w in all_scores) for i in range(len(df))]
    save_scores(name, split, mixed)
    row = evaluate(df, mixed, name) | {"split": split}
    append_result(row)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
