"""Lexical baseline: BM25 over each show's episode summaries (whole summary and chunked)."""
import sys

import numpy as np
from rank_bm25 import BM25Okapi

from common import append_result, chunk_text, evaluate, load_split, load_summaries, save_scores, tokenize

split = sys.argv[1] if len(sys.argv) > 1 else "test"
summ = load_summaries()
df = load_split(split)

for variant in ["bm25", "bm25_chunks"]:
    index = {}
    for show, (eps, texts) in summ.items():
        if variant == "bm25":
            docs, owner = texts, list(range(len(texts)))
        else:
            docs, owner = [], []
            for i, t in enumerate(texts):
                for c in chunk_text(t):
                    docs.append(c)
                    owner.append(i)
        index[show] = (BM25Okapi([tokenize(d) for d in docs]), np.array(owner), len(texts))
    scores = []
    for show, comment in df[["show", "comment"]].itertuples(index=False):
        bm, owner, n = index[show]
        s = bm.get_scores(tokenize(comment))
        ep_scores = np.full(n, -1e9)
        np.maximum.at(ep_scores, owner, s)  # MaxP: best chunk per episode
        scores.append(ep_scores)
    save_scores(variant, split, scores)
    row = evaluate(df, scores, variant) | {"split": split}
    append_result(row)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
