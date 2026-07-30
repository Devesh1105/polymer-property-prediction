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

## Can honest OOF reach 0.93? (exp2.py, exp3.py)

Honest protocol throughout: early stopping on an inner split of the training
fold followed by a refit on the whole fold, feature selection in-fold, 8-fold CV.
`results_exp3.txt` is the primary table.

| target | n | base | +compact | +cmp+smarts | +all | GP | **stack** | gain |
|---|---|---|---|---|---|---|---|---|
| eea | 221 | 0.8660 | **0.8779** | 0.8750 | 0.8604 | 0.8868 | **0.8914** | +0.0255 |
| ei  | 222 | 0.8109 | 0.8101 | **0.8118** | 0.8060 | 0.8022 | **0.8212** | +0.0103 |
| eps | 229 | **0.7800** | 0.7713 | 0.7743 | 0.7599 | 0.7624 | **0.7843** | +0.0043 |
| nc  | 229 | **0.8641** | 0.8524 | 0.8514 | 0.8297 | 0.8387 | **0.8650** | +0.0008 |
| egb | 337 | 0.8911 | 0.9055 | **0.9082** | 0.9019 | 0.8985 | **0.9143** | +0.0232 |

Blocks: compact = PHYS(8) + STRUCT(17) + CONJ(6) + TRI(8); smarts = donor/
acceptor + group-contribution counts (38); all = compact + smarts + AUTOCORR2D(192).

### Conclusions

1. **Feature width is harmful below ~350 rows.** `+all` is worse than `base` on
   four of five targets. `exp2.py` dumped ~1500 columns at once and cost nc and
   eps 0.033 each. The same features in a 39-column block gain +0.012 on eea.
2. **The conjugation/oligomer features help the electronic targets only.**
   eea +0.012 and egb +0.017, versus ~0 on ei and negative on eps and nc.
3. **Lorentz-Lorenz is redundant, not new signal.** Univariate R2 of 0.59 (nc)
   and 0.46 (eps) did not survive as incremental value: nc -0.0117, eps -0.0087.
   The trees already reconstruct it from MolMR and heavy-atom count.
4. **The GP is the one durable gain.** Mean blend gain +0.010 across the four
   ~220-row targets and +0.023 on egb. It wins even where the GP alone is worse
   than the tree (ei: GP 0.8022 vs LGB 0.8109, blend 0.8212) - the value is
   decorrelation, not accuracy.
5. **Learned stack weights are noise at this size.** Shrinkage toward uniform
   selected lambda=1.0 (pure equal weighting) on eea, ei and egb.
6. **PLS is not a viable fixed member** - 0.5565 on eps, 0.6328 on ei; it turned
   equal-weight blends negative in `exp2.py`.

### Reachable score

ei (0.821) and eps (0.784) are immovable across every feature block, model class
and blend tested here, and they cap the equal-weight mean. Projection for the
full pipeline: 0.887 + ~0.008 (GP member) + ~0.004 (seed-bagging small targets)
+ ~0.005 (stack-weight fix) = **~0.90 honest OOF**, tracking the leaderboard
within noise. 0.93 is not reachable on this data under these rules.
