# OOF → leaderboard gap diagnostics

Scripts backing the analysis of the 0.909 nested-CV OOF vs 0.887 LB gap.
Run order: `feats.py` (builds `feats.pkl`) → everything else.

## Ruled out as causes of the gap

| Hypothesis | Script | Result |
|---|---|---|
| Cross-target sibling features available in train but not test | inline | **No.** Sibling-known rate is 96–97% on both sides for eea/ei/eps/nc, 80% for egb, ~5% for egc, ~0% for tg. Matched. |
| Analog-series / scaffold optimism (CV neighbours closer than test neighbours) | `sim.py` | **No.** Median nearest-neighbour Tanimoto is 0.55 train-internal vs 0.56 test→train on small targets; 0.80 vs 0.80 on tg. |
| Duplicate collapsing / label denoising inflating OOF | inline | **No.** 3 conflicting label groups out of 7406; the `0.5*std` conflict rule drops 0 rows. |
| Same-molecule/same-target leakage between train and test | inline | **No.** 0 test rows share an exact (smiles, target) with train, except 2 tg rows. |

## Confirmed causes

`exp.py` — 8-fold OOF R² for one LightGBM under four protocols
(feature selection and early stopping each either leaky or honest):

| target | n | leakFS+leakES | honFS+leakES | leakFS+honES | honFS+honES | inflation |
|---|---|---|---|---|---|---|
| eea | 221 | 0.8652 | 0.8575 | 0.8576 | 0.8534 | **+0.0118** |
| ei  | 222 | 0.8195 | 0.8038 | 0.8074 | 0.8054 | **+0.0141** |
| eps | 229 | 0.7893 | 0.7941 | 0.7475 | 0.7436 | **+0.0458** |
| nc  | 229 | 0.8672 | 0.8704 | 0.8485 | 0.8525 | **+0.0148** |
| egb | 337 | 0.8973 | 0.8948 | 0.8826 | 0.8826 | **+0.0148** |

Early stopping on the fold being scored is the dominant term and is
consistent across targets. Feature selection on 100% of labels adds to it on
eea/ei but is neutral-to-harmful on eps/nc — on ~220 rows the selection is not
buying accuracy, only optimism.

`gnnsim2.py` — Monte-Carlo for the GNN's "keep the best-validation-R² epoch"
rule, where the scoring fold is also the checkpoint-selection set. Residual is
split into a shared irreducible part and an epoch-to-epoch jitter of variance
share `phi`:

| target | fold n | phi=0.10 | phi=0.20 | phi=0.35 |
|---|---|---|---|---|
| eea/ei/eps/nc (~222) | 27 | +0.0416 | +0.0553 | +0.0709 |
| egb (337) | 42 | +0.0346 | +0.0471 | +0.0602 |
| egc (2028) | 253 | +0.0147 | +0.0236 | +0.0392 |
| tg (4143) | 517 | +0.0123 | +0.0224 | +0.0386 |

`noise.py` — irreducible sampling noise of the metric. sd of the 7-target mean
R² is **0.0058** on the full test set, **0.0084** if the public LB is a 50%
split. Per-target sd is 0.025 for ei and eps. So noise explains at most ~0.008
of a 0.022 gap; the rest is genuine optimism.

## Feature findings

`ll.py` — Lorentz–Lorenz / Clausius–Mossotti ratio (Crippen molar refractivity
divided by an additive van der Waals volume) as a single scalar:

| target | pearson | spearman | univariate R² |
|---|---|---|---|
| nc  | +0.769 | +0.845 | **0.592** |
| eps | +0.679 | +0.777 | **0.461** |

`olig.py` — head-to-tail oligomer and macrocyclic ("periodic") reconstruction
of the repeat unit from the two `*` attachment points. Both succeed on
**6565/6565** unique training molecules. Note the cyclization degenerates to a
simple cap for 1,2-substituted vinyl backbones (the two `*` neighbours are
already bonded); the trimer handles those correctly.
