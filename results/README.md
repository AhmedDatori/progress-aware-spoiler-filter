# Results

| File | What it holds |
|---|---|
| `matching_results.csv` | One row per (model, split): MRR, top-1, top-3 and MRR per show. The source of every matching number. |
| `filter_results.csv` | Full-filter results per (matcher, plot classifier, system): spoilers hidden at 5% / 10% safe hidden, AUC. |
| `relevance_results.csv` | Plot classifiers: dev and test-show AUC, accuracy, training time. |
| `speed.json`, `speed_v6_K15_M4.json`, `speed_v6_K5_M1.json` | Timings on a laptop RTX 4050, and the Longformer estimate. |
| `predictions__v6_rr48_s2.csv` | v6's rank of the right episode and top guess for every test comment (no comment text). |
| `error_*.csv` | Error analysis tables: per show, by length, by distance, kinds of errors. |
| `error_sample.csv`, `error_labels_by_hand.csv` | The 60 wrong predictions read by hand, with our category for each. |
| `relevance_features.csv` | Features used in the F2 experiment (no comment text). |
| `demo_comments.json` | Data for the web demo: 200 test comments with P(plot) and P(episode). |
| `figures/` | The figures of the README and the slides (made by `src/figures.py`). |
| `logs_v6/` | Training and validation-grid logs of the v6 runs. |
| `log_*.txt`, `logs_relevance_*.txt` | Training logs of earlier runs. |

Not in git (download with `bash src/download_results.sh`): `scores/` (every model's scores per comment and episode, 78 MB) and
`filter_sim__*.parquet` (the simulated reader–comment pairs).
