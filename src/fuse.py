"""Mix two rankers' scores (e.g. fine-tuned bi-encoder + BM25) into one score per episode.

Why: they make different mistakes. BM25 wins on rare exact words ("pink eyes", "Powder");
the embedding model wins on meaning and spelling variants ("Connie"/"Conny", "dead"/"injured").

How: the two score types live on different scales (cosine ~0-1, BM25 ~0-40), so for EACH comment
we z-score each model's episode scores (subtract the mean, divide by the std), then add:
    final = z(dense) + w * z(bm25)
The weight w is chosen on VAL only; test is touched once with that w.
(This is exactly how the 0.557 result from 19 Nov was made: w = 0.3, bm25 = bm25_chunks.)

Usage: python fuse.py <dense tag> [bm25 tag] [out tag]      e.g. python fuse.py ft_nomic_24k bm25_chunks
"""
import sys

import numpy as np

from common import append_result, evaluate, load_scores, load_split, save_scores

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def mix(a, b, w):
    return [z(x) + w * z(y) for x, y in zip(a, b)]


def sweep(dense, sparse, split="val"):
    """MRR on `split` for every weight in WEIGHTS."""
    df = load_split(split)
    d, s = load_scores(dense, split), load_scores(sparse, split)
    return {w: evaluate(df, mix(d, s, w))["MRR"] for w in WEIGHTS}


if __name__ == "__main__":
    dense = sys.argv[1] if len(sys.argv) > 1 else "ft_nomic_24k"
    sparse = sys.argv[2] if len(sys.argv) > 2 else "bm25_chunks"
    val_mrr = sweep(dense, sparse, "val")
    for w, mrr in val_mrr.items():
        print(f"w={w:<4} val MRR {mrr:.4f}")
    best = max(val_mrr, key=val_mrr.get)
    print(f"best w on val: {best}")

    tag = sys.argv[3] if len(sys.argv) > 3 else f"{dense}+bm25_w{best}"
    for split in ["val", "test"]:
        df = load_split(split)
        s = mix(load_scores(dense, split), load_scores(sparse, split), best)
        save_scores(tag, split, s)
        row = evaluate(df, s, tag) | {"split": split}
        append_result(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
