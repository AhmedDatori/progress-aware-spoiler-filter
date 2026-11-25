"""End-to-end progress-aware spoiler filter (the part Tran et al. left as future work).

A viewer has watched episodes 1..k of a show. For every comment we decide: hide or show.
Ground truth: a comment is a spoiler for this viewer if it is RELEVANT (talks about plot events)
and comes from an episode thread e > k. Irrelevant comments ("this episode was fun") are never spoilers.

Systems compared (all use the same test comments, 4 unseen shows):
  hide_nothing        - no filter
  classic_detector    - hide every comment the relevance classifier flags as plot-revealing
                        (what a normal yes/no spoiler detector does; ignores viewer progress)
  progress_aware      - P(spoiler | k) = P(relevant) * P(episode > k); hide if above threshold
  progress_aware_oracle_rel - same but with the true relevance label (isolates matching quality)

Relevance classifier: frozen embedding model + logistic regression, trained on the hand-labelled
relevant/irrelevant comments of the NON-test shows only.

Usage: python filter_eval.py <matching model tag> [relevance model tag]
  (needs results/scores/<tag>__val.npy and __test.npy; relevance tag = "lr" (default, frozen + logistic
   regression) or a classifier from relevance.py, read from results/relevance_pred__<tag>.parquet)
"""
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RESULTS, ROOT, load_scores, load_split, load_summaries

TAG = sys.argv[1]
REL = sys.argv[2] if len(sys.argv) > 2 else "lr"
summ = load_summaries()
TEST_SHOWS = {"attack_on_titan", "kaguya_sama_love_is_war", "legend_of_korra", "arcane"}

# ---- relevance classifier
f = pd.concat([pd.read_csv(ROOT / f"data/raw/filtering/handlabeled/{s}.csv", header=None) for s in ["train", "dev", "test"]])
f.columns = ["irrelevant", "comment", "show", "episode"]
f["comment"] = f.comment.astype(str)
tr = f[~f.show.isin(TEST_SHOWS)]
te = f[f.show.isin(TEST_SHOWS)].reset_index(drop=True)
emb = SentenceTransformer("nomic-ai/modernbert-embed-base", device="cuda", model_kwargs={"dtype": torch.float16})
emb.max_seq_length = 256
enc = lambda xs: emb.encode(["classification: " + x for x in xs], batch_size=128, normalize_embeddings=True)
clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(enc(tr.comment.tolist()), 1 - tr.irrelevant)
te["p_rel"] = clf.predict_proba(enc(te.comment.tolist()))[:, 1]
if REL != "lr":   # use the fine-tuned classifier's predictions instead (same rows, same order)
    pr = pd.read_parquet(RESULTS / f"relevance_pred__{REL}.parquet")
    assert (pr.comment.values == te.comment.values).all()
    te["p_rel"] = pr.p_rel.values
rel_auc = roc_auc_score(1 - te.irrelevant, te.p_rel)
rel_acc = ((te.p_rel > 0.5) == (te.irrelevant == 0)).mean()
print(f"relevance classifier ({REL}) on test shows: AUC {rel_auc:.3f}, acc {rel_acc:.3f}")

# ---- episode probabilities from matching scores; temperature fitted on val
val, test = load_split("val"), load_split("test")
sv, st = load_scores(TAG, "val"), load_scores(TAG, "test")


def nll(logt, df, scores):
    t = np.exp(logt)
    return -np.mean([np.log(softmax(np.asarray(s, float) / t)[summ[sh][0].index(ep)] + 1e-12)
                     for (sh, ep), s in zip(df[["show", "episode"]].itertuples(index=False), scores)])


logt = minimize_scalar(nll, bounds=(-8, 3), args=(val, sv), method="bounded").x
T = np.exp(logt)
print(f"temperature {T:.4f}")
pep = {c: softmax(np.asarray(s, float) / T) for c, s in zip(test.comment, st)}

# irrelevant comments have no matching scores in the dataset: score them with the same matching model
# outputs are only needed through P(relevant) for them; P(ep > k) for irrelevant comments uses a uniform guess
# (a conservative choice: it only matters when the relevance classifier is wrong).

# ---- simulate viewers at every progress point k
rows = []
for r in te.itertuples(index=False):
    eps = summ[r.show][0]
    n = len(eps)
    if r.irrelevant == 0:
        p = pep[r.comment]
    else:
        p = np.full(n, 1 / n)
    tail = np.cumsum(p[::-1])[::-1]  # tail[j] = P(episode index >= j)
    pos = eps.index(r.episode) if r.episode in eps else int(np.searchsorted(eps, r.episode))
    for j in range(n - 1):  # viewer has watched eps[0..j]
        spoiler = (r.irrelevant == 0) and pos > j
        rows.append((r.show, r.irrelevant == 0, spoiler, r.p_rel, tail[j + 1]))
sim = pd.DataFrame(rows, columns=["show", "relevant", "spoiler", "p_rel", "p_later"])
sim["progress_aware"] = sim.p_rel * sim.p_later
sim["progress_aware_oracle_rel"] = sim.relevant * sim.p_later
sim["classic_detector"] = sim.p_rel
sim["hide_nothing"] = 0.0

out = []
for sys_name in ["hide_nothing", "classic_detector", "progress_aware", "progress_aware_oracle_rel"]:
    s = sim[sys_name]
    hide = s > 0.5
    out.append({
        "system": sys_name,
        "matching_model": TAG,
        "relevance_model": REL,
        "spoilers_hidden": hide[sim.spoiler].mean(),
        "safe_hidden": hide[~sim.spoiler].mean(),
        "safe_plot_comments_hidden": hide[~sim.spoiler & sim.relevant].mean(),
        "AUC": roc_auc_score(sim.spoiler, s) if s.nunique() > 1 else 0.5,
    })
res = pd.DataFrame(out)
# fair comparison: allow each system to hide as many safe comments as the classic detector does,
# and see how many spoilers it catches then
fpr_classic = res.loc[res.system == "classic_detector", "safe_hidden"].item()
for i, r in res.iterrows():
    s = sim[r.system]
    thr = np.quantile(s[~sim.spoiler], 1 - fpr_classic)
    res.loc[i, "spoilers_hidden_at_classic_fpr"] = (s[sim.spoiler] > thr).mean() if s.nunique() > 1 else 0.0
for fpr in (0.05, 0.10):
    for i, r in res.iterrows():
        s = sim[r.system]
        thr = np.quantile(s[~sim.spoiler], 1 - fpr)
        res.loc[i, f"spoilers_hidden_at_{int(fpr * 100)}pct_safe_hidden"] = (s[sim.spoiler] > thr).mean() if s.nunique() > 1 else 0.0
print(f"viewer-comment pairs: {len(sim):,}  spoiler share: {sim.spoiler.mean():.3f}")
print(res.round(3).to_string(index=False))
sim.to_parquet(RESULTS / f"filter_sim__{TAG.replace('+', '_')}{'' if REL == 'lr' else '__' + REL}.parquet")
path = RESULTS / "filter_results.csv"
old = pd.read_csv(path) if path.exists() else pd.DataFrame()
if len(old):
    old["relevance_model"] = old.get("relevance_model", pd.Series(index=old.index, dtype=object)).fillna("lr")
    old = old[~((old.matching_model == TAG) & (old.relevance_model == REL))]
pd.concat([old, res]).to_csv(path, index=False)
