"""Error analysis for the main matcher (v6_rr48_s2) on the test set.

Questions:
  1. Per show: where does the model do well or badly?
  2. When it is wrong, how far is its top guess from the right episode?
     (a guess 1 episode away is a "near miss"; for the filter it often does not matter much)
  3. Do short comments fail more often?
  4. A random sample of 60 wrong comments to read by hand (results/error_sample.csv). Our hand-made category for
     each one is kept in results/error_labels_by_hand.csv and joined in here, so re-running never loses it.
Usage: python error_analysis.py [model tag]
"""
import sys

import numpy as np
import pandas as pd

from common import RESULTS, load_scores, load_split, load_summaries

TAG = sys.argv[1] if len(sys.argv) > 1 else "v6_rr48_s2"
summ = load_summaries()
test = load_split("test")
scores = load_scores(TAG, "test")

rows = []
for (show, ep, comment), s in zip(test[["show", "episode", "comment"]].itertuples(index=False), scores):
    s = np.asarray(s, float)
    eps = summ[show][0]
    gold = eps.index(ep)
    pred = int(s.argmax())
    rank = 1 + int((s > s[gold]).sum())
    rows.append(dict(show=show, n_eps=len(eps), gold=ep, pred=eps[pred], rank=rank, rr=1 / rank,
                     offset=pred - gold, words=len(comment.split()), comment=comment))
df = pd.DataFrame(rows)
df["correct"] = df["rank"] == 1

print(f"model {TAG}: MRR {df.rr.mean():.3f}  top-1 {df.correct.mean():.3f}  top-3 {(df['rank'] <= 3).mean():.3f}  "
      f"top-5 {(df['rank'] <= 5).mean():.3f}")

# 1. per show
per_show = df.groupby("show").agg(comments=("rr", "size"), episodes=("n_eps", "first"), MRR=("rr", "mean"),
                                  top1=("correct", "mean")).round(3)
per_show["random_MRR"] = [round(np.mean([1 / r for r in range(1, n + 1)]), 3) for n in per_show.episodes]
print("\nper show\n", per_show.to_string())

# 2. how far off is the top guess when it is wrong
wrong = df[~df.correct]
dist = wrong.offset.abs()
bins = pd.cut(dist, [0, 1, 2, 5, 1000], labels=["1 episode away", "2 away", "3-5 away", "6+ away"])
off = bins.value_counts(normalize=True).reindex(["1 episode away", "2 away", "3-5 away", "6+ away"]).round(3)
print(f"\nwrong top guesses: {len(wrong)} ({len(wrong) / len(df):.1%} of comments)")
print("distance of the wrong guess:\n", off.to_string())
print(f"guess is a LATER episode than the right one: {(wrong.offset > 0).mean():.1%}")

# 3. comment length
df["length"] = pd.cut(df.words, [0, 10, 20, 40, 10000], labels=["1-10 words", "11-20", "21-40", "41+"])
by_len = df.groupby("length", observed=True).agg(comments=("rr", "size"), MRR=("rr", "mean"), top1=("correct", "mean")).round(3)
print("\nby comment length\n", by_len.to_string())

# save tables + a fixed random sample of wrong comments for reading by hand
per_show.to_csv(RESULTS / "error_per_show.csv")
by_len.to_csv(RESULTS / "error_by_length.csv")
off.to_csv(RESULTS / "error_distance.csv")
sample = wrong.sample(60, random_state=0)[["show", "gold", "pred", "rank", "words", "comment"]]
labels = RESULTS / "error_labels_by_hand.csv"
if labels.exists() and TAG == "v6_rr48_s2":
    lab = pd.read_csv(labels)
    assert (lab.comment.values == sample.comment.values).all(), "the sample changed; the hand labels no longer match"
    sample["category"] = lab.category.values
    cat = sample.category.value_counts().rename("comments").to_frame()
    cat["share"] = (cat.comments / len(sample)).round(2)
    cat.to_csv(RESULTS / "error_categories.csv")
    print("\nkinds of errors (60 read by hand)\n", cat.to_string())
sample.to_csv(RESULTS / "error_sample.csv", index=False)
df.drop(columns="comment").to_csv(RESULTS / f"predictions__{TAG}.csv", index=False)
