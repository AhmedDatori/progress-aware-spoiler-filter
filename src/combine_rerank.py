"""Turn reranker chunk scores (train_reranker.py) into final episode rankings.

For each comment, the top K episodes of stage 1 get
    final = z(rerank) + a * z(stage 1)          (z = standardised over those K episodes)
    rerank = best score among the episode's top M chunks (MaxP); with several reranker tags (seeds) we average
             their z-scores first
The other episodes stay below them, in stage-1 order.
K (episodes re-scored), M (chunks per episode) and a (how much stage 1 still counts) are picked on VAL only;
test is scored once with the winner.

Usage: python combine_rerank.py <stage-1 tag> <out name> <rr tag> [<rr tag> ...]
  python combine_rerank.py v6_first v6 v6_rr_s0
"""
import pickle
import sys

import numpy as np

from common import SCORES, append_result, evaluate, load_scores, load_split, save_scores
from fuse import z

KS, MS, AS = [5, 10, 15], [1, 2, 4], [0.0, 0.25, 0.5, 1.0, 2.0]


def build(first, rrs, K, M, a):
    out = []
    for i, f in enumerate(first):
        f = np.asarray(f, float)
        top = np.argsort(-f)[:K]
        r = np.mean([z([max(rr[i][int(j)][:M]) for j in top]) for rr in rrs], axis=0)
        s = f - 1e6
        s[top] = r + a * z(f[top])
        out.append(s)
    return out


if __name__ == "__main__":
    first_tag, name, rr_tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    data = {sp: (load_split(sp), load_scores(first_tag, sp), [pickle.load(open(SCORES / f"{t}__{sp}.pkl", "rb")) for t in rr_tags])
            for sp in ["val", "test"]}
    df, first, rrs = data["val"]
    kmax = max(len(d) for d in rrs[0])
    KS = [k for k in KS if k <= kmax]
    grid = {(K, M, a): evaluate(df, build(first, rrs, K, M, a))["MRR"] for K in KS for M in MS for a in AS}
    for K in KS:
        print(f"K={K:<3}" + "  ".join(f"M={M} " + " ".join(f"{grid[K, M, a]:.3f}" for a in AS) for M in MS))
    K, M, a = max(grid, key=grid.get)
    print(f"best on val: K={K} M={M} a={a} (val MRR {grid[K, M, a]:.4f}); a values in each row: {AS}", flush=True)
    for split in ["val", "test"]:
        df, first, rrs = data[split]
        s = build(first, rrs, K, M, a)
        save_scores(name, split, s)
        row = evaluate(df, s, name) | {"split": split, "K": K, "M": M, "a": a}
        append_result(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
