"""Figures for the final report and slides (all numbers are read from results/ or typed from the tables there).
  fig1_mrr.png     matching: the paper's models vs ours (test, unseen shows)
  fig2_filter.png  full filter: spoilers hidden vs safe comments hidden (v6 + fine-tuned plot classifier)
  fig3_errors.png  error analysis: kinds of mistakes (60 read by hand) and MRR by comment length
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

from common import RESULTS

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, GREY = "#3355d1", "#e06a2c", "#1f9a5a", "#9aa3b2"
INK, MUTED, GRID = "#1d2330", "#5d6676", "#dfe3ea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})


def clean(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- 1. MRR comparison
mrr = pd.read_csv(RESULTS / "matching_results.csv")
mrr = mrr[mrr.split == "test"].drop_duplicates("model", keep="last").set_index("model").MRR
rows = [("RoBERTa", 0.352, "paper"), ("BM25", 0.368, "paper"), ("Nyströmformer", 0.402, "paper"),
        ("BigBird", 0.462, "paper"), ("MaxP-RoBERTa", 0.541, "paper"), ("Longformer (paper's best)", 0.611, "paper"),
        ("Ours, no training (149M)", mrr["nomic_modernbert_chunks"], "ours"),
        ("Ours, fast model (149M + BM25)", mrr["v4_72k+bm25_w0.3"], "ours"),
        ("Ours, reranker not trained (298M)", mrr["v6_gte_zeroshot"], "ours"),
        ("Ours, best model v6 (298M)", np.mean([mrr[f"v6_rr48_s{i}"] for i in range(3)]), "best")]
rows = sorted(rows, key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=220)
y = np.arange(len(rows))
col = {"paper": ORANGE, "ours": "#8ea2e6", "best": BLUE}
ax.barh(y, [r[1] for r in rows], height=0.66, color=[col[r[2]] for r in rows])
for i, r in enumerate(rows):
    ax.text(r[1] + 0.006, i, f"{r[1]:.3f}", va="center", fontsize=8.5, color=INK,
            fontweight="bold" if r[2] == "best" else "normal")
ax.axvline(0.611, color=ORANGE, linewidth=1, linestyle=":")
ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
ax.set_xlim(0, 0.72)
ax.set_xlabel("MRR on 4 unseen test shows (higher is better)")
clean(ax)
ax.legend(handles=[Patch(color=ORANGE, label="Tran et al. 2023"), Patch(color=BLUE, label="Ours")],
          loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig1_mrr.png")

# ---- 2. filter trade-off
sim = pd.read_parquet(RESULTS / "filter_sim__v6_rr48_s2__relevance_ft_base.parquet")
fig, ax = plt.subplots(figsize=(4.6, 3.3), dpi=220)
for c, label, color in [("classic_detector", "Normal yes/no detector", ORANGE),
                        ("progress_aware", "Ours: progress-aware", BLUE),
                        ("progress_aware_oracle_rel", "Ours + perfect plot labels", GREEN)]:
    fpr, tpr, _ = roc_curve(sim.spoiler, sim[c])
    ax.plot(fpr * 100, tpr * 100, color=color, linewidth=2, label=label)
    k = np.searchsorted(fpr, 0.10)
    ax.plot(fpr[k] * 100, tpr[k] * 100, "o", ms=5, color=color, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(fpr[k] * 100 + 1.5, tpr[k] * 100 - 5, f"{tpr[k] * 100:.0f}%", fontsize=8.5, color=INK)
ax.axvline(10, color=MUTED, linewidth=0.8, linestyle=":")
ax.set_xlim(0, 50)
ax.set_ylim(0, 102)
ax.set_xlabel("Safe comments hidden by mistake (%)")
ax.set_ylabel("Spoilers hidden (%)")
clean(ax, "both")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig2_filter.png")

# ---- 3. errors
cat = pd.read_csv(RESULTS / "error_categories.csv", index_col=0).comments.sort_values()
by_len = pd.read_csv(RESULTS / "error_by_length.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 2.9), dpi=220, gridspec_kw={"width_ratios": [1.5, 1]})
a1.barh(range(len(cat)), cat.values, color=BLUE, height=0.62)
for i, v in enumerate(cat.values):
    a1.text(v + 0.3, i, str(v), va="center", fontsize=8.5)
a1.set_yticks(range(len(cat)), cat.index, fontsize=8.5)
a1.set_xlabel("wrong predictions (of 60 read by hand)")
a1.set_title("Why the top guess was wrong", fontsize=9.5, loc="left")
clean(a1)
a2.bar(range(len(by_len)), by_len.MRR, color=BLUE, width=0.62)
for i, v in enumerate(by_len.MRR):
    a2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
a2.set_xticks(range(len(by_len)), [l.replace(" words", "") for l in by_len.length], fontsize=8.5)
a2.set_xlabel("comment length (words)")
a2.set_ylim(0, 0.8)
a2.set_ylabel("MRR")
a2.set_title("Longer comments are easier", fontsize=9.5)
clean(a2, "y")
fig.tight_layout()
fig.savefig(FIG / "fig3_errors.png")
print("saved")
