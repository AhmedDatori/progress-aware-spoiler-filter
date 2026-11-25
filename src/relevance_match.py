"""F2: better "is this comment about the plot?" by also asking the episode matcher.

Idea: a plot comment ("Levi killed the Beast Titan") is very close to ONE part of an episode summary.
A general comment ("this episode was fun") is not really close to any part.
So the matcher's best similarity is a second clue next to the classifier's P(plot).

Features per comment (all from models we already have, nothing new is trained except a tiny logistic regression):
  1. logit of P(plot) from the fine-tuned classifier (relevance_ft_base)
  2. best cosine similarity between the comment and any summary chunk of its show (v6_bi)
  3. best BM25 score over the summary chunks (word overlap)
  4. number of words (log)
The logistic regression is fitted ONLY on the dev part of the 4 non-test shows (the same 15% held-out split as
relevance.py), so the classifier's training comments are not reused. Test shows are scored once.
v6_bi never saw these 4 shows in training (val shows were left out), so feature 2 is fair.

Writes results/relevance_pred__relevance_match.parquet (same rows/order as the other relevance files).
"""
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RESULTS, ROOT, chunk_text, load_summaries, tokenize
from relevance import TEST_SHOWS, load_relevance, predict_relevance

f = load_relevance()
summ = load_summaries()

# 1. classifier
f["p_rel"] = predict_relevance(str(ROOT / "models/relevance_ft_base"), f.comment.tolist())

# 2. + 3. matcher similarity
bi = SentenceTransformer(str(ROOT / "models/v6_bi"), device="cuda", model_kwargs={"dtype": torch.float16})
bi.max_seq_length = 384
q = bi.encode(["search_query: " + c for c in f.comment], batch_size=64, normalize_embeddings=True)
f["max_cos"], f["max_bm25"] = 0.0, 0.0
for show, idx in f.groupby("show").groups.items():
    chunks = [c for t in summ[show][1] for c in chunk_text(t)]
    e = bi.encode(["search_document: " + c for c in chunks], batch_size=32, normalize_embeddings=True)
    f.loc[idx, "max_cos"] = (q[idx] @ e.T).max(1)
    bm = BM25Okapi([tokenize(c) for c in chunks])
    f.loc[idx, "max_bm25"] = [bm.get_scores(tokenize(c)).max() for c in f.loc[idx, "comment"]]
f["log_words"] = np.log1p(f.comment.str.split().str.len())

# same split as relevance.py: shuffle non-test shows with seed 0, first 15% = dev
pool = f[~f.show.isin(TEST_SHOWS)].sample(frac=1, random_state=0)
dev = pool.iloc[:int(0.15 * len(pool))]
te = f[f.show.isin(TEST_SHOWS)].reset_index(drop=True)

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
feats = lambda d: np.c_[logit(d.p_rel), d.max_cos, np.log1p(d.max_bm25), d.log_words]
lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(dev), dev.label)
te["p_new"] = lr.predict_proba(feats(te))[:, 1]

print(f"dev comments used to fit: {len(dev)}, test comments: {len(te)}")
print(f"test AUC  classifier alone {roc_auc_score(te.label, te.p_rel):.3f}")
print(f"test AUC  best cosine alone {roc_auc_score(te.label, te.max_cos):.3f}")
print(f"test AUC  classifier + matcher {roc_auc_score(te.label, te.p_new):.3f}")
print("weights (logit p, max cos, log bm25, log words):", lr.coef_.round(2), lr.intercept_.round(2))
for show, g in te.groupby("show"):
    print(f"  {show:28s} {roc_auc_score(g.label, g.p_rel):.3f} -> {roc_auc_score(g.label, g.p_new):.3f}")

out = te[["comment", "show", "episode", "label"]].copy()
out["p_rel"] = te.p_new
out.to_parquet(RESULTS / "relevance_pred__relevance_match.parquet")
f.drop(columns=["comment"]).to_csv(RESULTS / "relevance_features.csv", index=False)
