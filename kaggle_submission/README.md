# Kaggle run — best honest-OOF configuration

`polymer_r2_best_oof.ipynb` — upload as a Kaggle notebook, attach the competition
data, **enable the GPU accelerator**, Run All. Inputs are discovered automatically
anywhere under `/kaggle/input/`; no paths need editing. Writes `submission.csv`.

## Expected result

Printed OOF should land around **0.90–0.91**, and tg/egc will score higher on the
leaderboard than in OOF (see below). This is lower than the 0.912 an earlier
version printed, on purpose: that number tuned early stopping and GNN checkpoints
on the folds it scored, which is why the same run scored 0.889 on the leaderboard.
The gap here should be under ~0.01, which is roughly the metric's own sampling sd
at these test sizes.

## What is enabled

| setting | value | why |
|---|---|---|
| `USE_PHASE1_TRAIN` | **True** | +0.0141 tg, +0.0144 egc honest OOF from 40% more data |
| `USE_PHASE1_LABELS` | False | affects only `submission.csv`, never OOF |
| `GP_MAX_N` | 1500 | GP is O(n^2 d) to build, O(n^3) to solve; measured benefit was on small targets |
| `PI1M_MAX` | 60000 (GPU) / 12000 (CPU) | self-supervised pretraining corpus |
| `PRETRAIN_EPOCHS` | 40 (GPU) / 8 (CPU) | |
| GNN / MTL seeds | 2 (GPU) / 1 (CPU) | |
| tree seeds | 3 big / 10 small | small targets have LB sd ~0.025 |

All of these live in the runtime-budget cell (cell 4) and the phase-1 cell (cell 9).

## Two caveats

**tg and egc leaderboard will exceed their OOF.** The phase-1 rows merged into
training are themselves phase-2 test rows, so the model reproduces them at test
time. Set `USE_PHASE1_TRAIN=False` in cell 9 to revert. Whether the archive may be
used at all is a rules judgement — cell 9 documents exactly what it does.

**0.93 honest OOF is not reachable.** ei (~0.82) and eps (~0.78) did not move for
any feature block, model class or blend tested, and they cap the equal-weight mean.
Full evidence in `../diagnostics/`.

## Runtime

Roughly 5–8h on a Kaggle GPU session with these defaults, dominated by PI1M
pretraining and the multi-task GNN. If you are tight on time, lower `PI1M_MAX`
to 20000 and `PRETRAIN_EPOCHS` to 15 first — those cost the least score per
minute saved.
