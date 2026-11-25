"""F: fine-tuned plot-relevance classifier ("is this comment about plot events at all?").

Before (filter_eval.py, 19 Nov): frozen nomic vectors + logistic regression, AUC 0.855 on the test shows.
That classifier is the filter's bottleneck (68% -> 98% spoilers caught with perfect labels).
Now: fine-tune the WHOLE encoder with a classification head (cross-entropy), so the model can learn
what "about plot" looks like instead of reusing vectors made for search.

Data: the hand-labelled relevance CSVs. Train ONLY on the 4 non-test shows; 15% of those comments are
held out as "dev" to pick the model (by dev AUC). The test shows are scored once at the end.

Usage
  python relevance.py nomic-ai/modernbert-embed-base relevance_ft_base
  python relevance.py answerdotai/ModernBERT-large   relevance_ft_large
Writes models/<tag>/, results/relevance_results.csv, results/relevance_pred__<tag>.parquet (test-show comments).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

from common import RESULTS, ROOT, warmup

TEST_SHOWS = {"attack_on_titan", "kaguya_sama_love_is_war", "legend_of_korra", "arcane"}
MAX_LEN = 128   # comments: median ~30 words; 128 tokens covers ~95% of them


def load_relevance():
    f = pd.concat([pd.read_csv(ROOT / f"data/raw/filtering/handlabeled/{s}.csv", header=None) for s in ["train", "dev", "test"]])
    f.columns = ["irrelevant", "comment", "show", "episode"]
    f["comment"] = f.comment.astype(str)
    f["label"] = 1 - f.irrelevant          # 1 = about plot
    return f.reset_index(drop=True)


@torch.no_grad()
def predict_relevance(model_dir, texts, batch=256):
    """P(about plot) for a list of comments, with a saved classifier."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    order = np.argsort([len(t) for t in texts])   # similar lengths per batch = faster
    out = np.empty(len(texts))
    for k in range(0, len(texts), batch):
        idx = order[k:k + batch]
        b = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        out[idx] = torch.softmax(m(**b).logits.float(), -1)[:, 1].cpu().numpy()
    del m
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "nomic-ai/modernbert-embed-base"
    tag = sys.argv[2] if len(sys.argv) > 2 else "relevance_ft"
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
    out_dir = ROOT / "models" / tag

    f = load_relevance()
    pool = f[~f.show.isin(TEST_SHOWS)].sample(frac=1, random_state=0)
    n_dev = int(0.15 * len(pool))
    dev, tr = pool.iloc[:n_dev], pool.iloc[n_dev:]
    te = f[f.show.isin(TEST_SHOWS)].reset_index(drop=True)
    print(f"train {len(tr)} | dev {len(dev)} (same shows) | test {len(te)} (unseen shows)", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    enc = lambda d: Dataset.from_dict({"text": d.comment.tolist(), "label": d.label.tolist()}).map(
        lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)

    def metrics(p):
        prob = torch.softmax(torch.tensor(p.predictions).float(), -1)[:, 1].numpy()
        return {"auc": roc_auc_score(p.label_ids, prob), "acc": ((prob > 0.5) == p.label_ids).mean()}

    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpt"), num_train_epochs=3, per_device_train_batch_size=32, per_device_eval_batch_size=128,
        learning_rate=lr, weight_decay=0.01, **warmup(0.1), bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="auc", logging_steps=20, report_to="none", seed=0,
    )
    t0 = time.time()
    trainer = Trainer(model=model, args=targs, train_dataset=enc(tr), eval_dataset=enc(dev),
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    secs = time.time() - t0
    dev_m = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    import shutil
    shutil.rmtree(out_dir / "ckpt", ignore_errors=True)

    te["p_rel"] = predict_relevance(str(out_dir), te.comment.tolist())
    test_auc = roc_auc_score(te.label, te.p_rel)
    test_acc = ((te.p_rel > 0.5) == (te.label == 1)).mean()
    te[["comment", "show", "episode", "label", "p_rel"]].to_parquet(RESULTS / f"relevance_pred__{tag}.parquet")
    row = {"model": tag, "base": base, "lr": lr, "dev_auc": dev_m["eval_auc"], "test_auc": test_auc,
           "test_acc": test_acc, "train_secs": round(secs)}
    path = RESULTS / "relevance_results.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old.model != tag]
    pd.concat([old, pd.DataFrame([row])]).to_csv(path, index=False)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
