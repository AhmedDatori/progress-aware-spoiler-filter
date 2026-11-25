"""Shared data loading and evaluation for spoiler matching.

Task (Tran et al., EMNLP 2023): given a comment about a known TV show,
rank that show's episode summaries; the correct episode should come first.
Main metric: MRR (mean reciprocal rank) on the test set (4 unseen shows).
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Project folder. Laptop: the folder above src/. Colab: set SM_ROOT (the notebook does this).
ROOT = Path(os.environ.get("SM_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw" / "matching" / "with_autolabels"
RESULTS = ROOT / "results"
SCORES = RESULTS / "scores"  # one .npy score matrix per model and split
for p in (RESULTS, SCORES):
    p.mkdir(parents=True, exist_ok=True)


def warmup(ratio):
    """Linear-warmup setting that works on Transformers 4.x (laptop) and 5.x (Colab).
    v5 removed `warmup_ratio`; there `warmup_steps` takes a float in [0, 1) as a ratio."""
    from transformers import TrainingArguments
    return {"warmup_ratio": ratio} if "warmup_ratio" in TrainingArguments.__dataclass_fields__ else {"warmup_steps": ratio}


def load_summaries():
    """dict show -> (list of episode numbers, list of summary texts), sorted by episode."""
    out = {}
    for show, ep, text in json.load(open(RAW / "summaries.json")):
        out.setdefault(show, []).append((ep, text))
    return {s: ([e for e, _ in sorted(v)], [t for _, t in sorted(v)]) for s, v in out.items()}


def load_split(split):
    rows = json.load(open(RAW / f"{split}.json"))
    return pd.DataFrame(rows, columns=["show", "episode", "comment"])


def chunk_text(text, size=200, stride=150):
    """Split a long summary into overlapping word windows (for short-context models)."""
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - size + stride), stride)]


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def evaluate(df, scores, name=None):
    """scores: list (one per comment) of arrays aligned to that show's episode list.
    Returns dict with MRR, top-1, top-3 accuracy, overall and per show."""
    summ = load_summaries()
    rr, top1, top3, shows = [], [], [], []
    for (show, ep), s in zip(df[["show", "episode"]].itertuples(index=False), scores):
        eps = summ[show][0]
        gold = eps.index(ep)
        rank = 1 + int((np.asarray(s) > s[gold]).sum())
        rr.append(1 / rank)
        top1.append(rank == 1)
        top3.append(rank <= 3)
        shows.append(show)
    res = pd.DataFrame({"show": shows, "rr": rr, "top1": top1, "top3": top3})
    out = {"model": name, "MRR": res.rr.mean(), "top1": res.top1.mean(), "top3": res.top3.mean()}
    for show, g in res.groupby("show"):
        out[f"MRR_{show}"] = g.rr.mean()
    return out


def save_scores(name, split, scores):
    np.save(SCORES / f"{name}__{split}.npy", np.array(scores, dtype=object), allow_pickle=True)


def load_scores(name, split):
    return list(np.load(SCORES / f"{name}__{split}.npy", allow_pickle=True))


def append_result(row, path=RESULTS / "matching_results.csv"):
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old.model == row["model"]) & (old.get("split", "test") == row.get("split", "test")))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
