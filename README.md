# Polymer property prediction — Round 2

Predict 7 properties (`tg`, `egc`, `egb`, `ei`, `eea`, `nc`, `eps`) from homopolymer
repeat-unit SMILES. Data is long-format, one row per molecule–property pair. The
metric is the mean of per-target R², all 7 weighted equally, so a gain on a
220-row target counts exactly as much as one on `tg`.

## Layout

| path | what it is |
|---|---|
| `kaggle_submission/` | **Start here.** Ready-to-run Kaggle notebook + its README |
| `polymer_r2_v8.ipynb` | Same pipeline, local paths, both phase-1 flags off |
| `diagnostics/` | The measurements behind every choice in the notebook |
| `train.csv`, `test.csv` | Phase-2 data (7,409 / 4,940 rows) |
| `archive/` | Phase-1 data — read `diagnostics/README.md` before using |
| `PI1M.zip` | ~1M unlabelled polymer SMILES for self-supervised pretraining |

## Class imbalance

`tg` has 4,143 training rows and `egc` 2,028, but `egb`/`eps`/`nc`/`ei`/`eea` have
only 221–337 each. Most of what the diagnostics found follows from that: feature
width hurts below ~350 rows, learned stack weights are noise at n≈220, and
anything tuned on a fold gets tuned on ~27 rows.

## Current state

Honest out-of-fold ≈ 0.90–0.91. An earlier version reported 0.912 and scored 0.889
on the leaderboard; that 0.023 gap was selection optimism, not bad luck — it tuned
early stopping and GNN checkpoints on the folds it scored. The notebook here fixes
all three sites, so OOF should now track the leaderboard within ~0.01.

`ei` (~0.82) and `eps` (~0.78) did not move for any feature block, model class or
blend tested, and they cap the equal-weight mean. 0.93 honest OOF is not reachable
on this data.

Full evidence, including what was tried and rejected, is in `diagnostics/README.md`.
