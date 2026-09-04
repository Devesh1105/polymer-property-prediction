# Invariance-First Polymer Property Prediction

**Seven polymer properties from a repeat-unit SMILES string — `Tg`, `Egc`, `Egb`, `Ei`, `Eea`, `Nc`, `EPS` — in one self-contained notebook.**

**Mean out-of-fold R² = 0.9178** across the seven targets, on grouped 8-fold CV where a structure and every alternative writing of it are confined to a single fold. 401 minutes end to end on one GPU, no external weights, no network access at runtime.

Built for Round 3 of the ANRF AISEHack 2.0 Polymer Property Prediction Challenge, whose two themes were **explainability** and **polymer invariance**.

---

## Results

| Property | | OOF R² | RMSE | Members |
|---|---|---|---|---|
| `Egb` | bulk bandgap | **0.9506** | 0.440 eV | 17 |
| `Eea` | electron affinity | **0.9458** | 0.258 eV | 17 |
| `Nc` | refractive index | **0.9318** | 0.062 | 17 |
| `Egc` | chain bandgap | **0.9239** | 0.433 eV | 9 |
| `Tg` | glass transition | **0.9217** | 30.5 K | 8 |
| `EPS` | dielectric constant | **0.8735** | 0.389 | 17 |
| `Ei` | ionisation energy | **0.8771** | 0.367 eV | 17 |
| | **mean** | **0.9178** | | |

Every number is the notebook's own out-of-fold estimate under `GroupKFold` on the canonical repeat unit, with nested early stopping and a meta-learner cross-validated on the *same* grouped folds. Feature selection, scaling and target transforms are all fitted inside the training partition of each fold.

### Against published baselines

The strongest published multitask model on this property set is polyGNN (Kuenneth & Ramprasad, *Chem. Mater.* 2023). Converting their reported RMSEs to R² under this dataset's per-property variance:

| Property | polyGNN RMSE | → R² | This work | Δ |
|---|---|---|---|---|
| `Ei` | 0.540 eV | 0.734 | **0.8771** | **+0.143** |
| `Egb` | 0.468 eV | 0.944 | **0.9506** | **+0.007** |
| `Tg` | 31.7 K | 0.916 | **0.9217** | **+0.006** |
| `Egc` | 0.445 eV | 0.920 | **0.9239** | **+0.004** |
| `Nc` | 0.0507 | 0.954 | 0.9318 | −0.022 |
| `Eea`, `EPS` | not reported | — | 0.9458 / 0.8735 | — |

Ahead on four of the five comparable properties, and by a wide margin on `Ei`, which is the hardest of the seven (222 labels). Behind on `Nc`.

**Read this comparison with its caveats.** The two studies use different train/test splits and different dataset sizes, so this is a like-for-like comparison of *reported numbers*, not a controlled head-to-head. Part of the `Ei`/`Eea` margin comes from the cross-property channel below, which uses sibling property values available in the training table — legitimate information here, but not something the published single-property baselines were given. Treated as a benchmark result, the honest claim is: **this pipeline matches or exceeds published multitask-GNN accuracy on most of these targets while additionally being exactly invariant and fully explainable**, which the baselines are not.

---

## What is actually new here

### 1 · The Canonical Repeat Unit — invariance as a guarantee, not a score

A repeat-unit SMILES carries five arbitrary choices with no physical meaning: atom ordering, which end is written first, aromatic vs Kekulé form, **where the repeat unit was cut**, and **how many units were written**. `*CCO*`, `*COC*`, `*OCC*` and `*CCOCCO*` are all poly(ethylene oxide).

Graph featurization handles the first three for free. The last two break models. The CRU map collapses them by closing the unit into its periodic form, reducing to the primitive period, and taking a canonical form over the equivalence class of valid cuts.

The audit measures this at the **feature level**, which is where it becomes a guarantee rather than a statistic:

```
through the CRU pipeline : max |Δfeature| = 0.00e+00   prediction drift = 0.0000 % of a target sd
bypassing the CRU map    : max |Δfeature| = 1.00e+06   prediction drift = 13.91 % of a target sd

CRU keys agreeing with the original, over 400 held-out structures × 6 transforms:
  atom order 100%  ·  Kekulé form 100%  ·  endpoint swap 100%
  repeat-unit recut 100%  ·  dimer view 100%  ·  trimer view 100%
```

Identical feature vectors mean *every* downstream model — tree, kernel or graph network, fitted or not — returns an identical number. Re-cut the unit, flip the ends, shuffle atom order, hand it a trimer: same polymer in, same number out, exactly.

The model underneath is a separate question, and the notebook answers it separately: bypassing the CRU map costs 13.9% of a target standard deviation at worst, and oligomer augmentation during training halved the dimer-view drift (12.6% → 5.8%). The two layers are independent — the canonical form makes it exact, the training makes it degrade gracefully if anything upstream hands the model a structure the canonicaliser never saw.

### 2 · Physics operators — the Bloch Hamiltonian of the chain

A repeat unit with two attachment points is not a molecule. It is the **unit cell of a one-dimensional crystal**, so the polymer has a Hamiltonian:

$$H(k) = H_0 + H_1 e^{ik} + H_1^{\dagger} e^{-ik}, \qquad k \in [0, \pi]$$

Diagonalising over the Brillouin zone gives the **band structure of the polymer** — gap, band edges, bandwidths, curvatures — as the n→∞ limit in closed form. No oligomer extrapolation, no finite-size error, one eigendecomposition per molecule at ~4 ms. These become 31 features computed from SMILES alone, with no fitting and no training data.

Because it is that cheap, the same operator doubles as a **free physics labeller** for self-supervised pretraining (below), which is what makes the unlabelled corpus worth something after the masked-atom objective saturates.

### 3 · Pretraining on the unlabelled corpora, weighted by what they're worth

| Corpus | Size | `*` notation | Mean Tanimoto to the competition set |
|---|---|---|---|
| `PI1M` | ~1 M | 100% | **0.61** — the same chemical manifold |
| `smile_r3` | ~5.97 M | 0% | 0.27 — drug-like, a different manifold |

`PI1M` carries the value and is used in full. `smile_r3` is chemically distant and is used only as a minority slice for breadth in the masked-atom objective, and as the reference pool for the applicability-domain check. The pretext task is masked atoms **plus the 31 physics targets from the Bloch operator** — masked atoms alone were solved (loss 0.0006 by epoch 4 on 500k molecules), at which point more unlabelled data buys nothing.

### 4 · Cross-property structure

Each row carries one of seven labels, but the same molecule often appears elsewhere under a different property, and those properties are physically coupled: `Egc`–`Egb` correlate at 0.96, `Nc`–`EPS` at 0.92, and Koopmans' theorem ties `Ei − Eea ≈ Egc` (verified at R² ≈ 0.97 in this data). Roughly 130 molecules carry all five minority properties.

Every molecule gets its six sibling values — measured where the training table has one, model-estimated otherwise — each paired with a **known/estimated flag** so the model can decide how far to trust it. When a property is being predicted, its own column is removed; `allowed_cols()` enforces that no target-derived feature ever reaches its own model.

Imposing Koopmans as a hard loss penalty measured **−0.009**. Supplying it as a gated feature measured **+0.018** on `Ei`. The notebook shows the model tracking the law closely when both inputs are measured and loosening when they are estimates — using the physics as evidence, not obeying it blindly.

### 5 · A deliberately heterogeneous ensemble

Five of seven properties have ~220 labels. At that size the biggest lever is not a better model, it is **more models that fail differently**. Seventeen members per target where the data supports it: XGBoost, LightGBM (incl. extra-trees mode), CatBoost, ExtraTrees, GPR, KRR, SVR-Tanimoto, ridge, a periodic D-MPNN operator network, and a cross-property residual channel.

ExtraTrees was the single strongest model on `Egb` and `Nc`; GPR on `Ei`. The blend beat the best single member by **+0.014** mean R².

The meta-learner is NNLS shrunk toward the equal-weight mean (`λ = 0.25`) — an unshrunk NNLS on 220 rows puts spiky weights on whichever member got lucky in that fold split, and those weights do not transfer. The meta-learner is validated on the same grouped folds as the base models; a shuffled KFold at the meta level is exactly the mistake that produced a +0.02 CV gain in an earlier round that never appeared on the leaderboard.

---

## Explainability

Three levels of attribution, each answering a different question:

- **Which kind of chemistry?** SHAP aggregated by feature family.
- **Which specific features?** SHAP per target, with every feature decoded back into chemistry — a triplet motif rendered as element/degree tokens, a MACCS bit as its SMARTS pattern, a descriptor by name. No bare hash indices.
- **Which atoms?** Gradient × embedding from the graph network, drawn on the molecule. Read off the same network that contributes to the submitted ensemble, not a surrogate fitted afterwards to explain it.

The attributions reproduce textbook polymer physics without being told it. `Tg`'s top drivers come out as rotatable-bond fraction along the backbone (lowers), backbone bonds inside rings (raises), and backbone conjugation and aromaticity (raise). `Nc` leans on the dielectric sibling, the inverse bandgap and the Koopmans term. The notebook plots the measured effects alongside the SHAP rankings so the two can be checked against each other.

## Trust

Every prediction ships with a conformal interval and an applicability-domain flag. Coverage is calibrated to 90% and lands at 89.6–90.0% on all seven targets. Ensemble disagreement correlates with actual error (r = 0.13–0.38), which is what lets the adaptive interval narrow on easy molecules instead of applying one width to everything. Molecules below the novelty threshold are flagged as extrapolations — they are still submitted, but the flag travels with them.

Written to `predictions_with_uncertainty.csv` alongside `submission.csv`.

---

## Reproducing

```bash
# Kaggle: attach the competition dataset, set accelerator to GPU, Run All.
# Locally:
pip install rdkit xgboost lightgbm catboost shap torch scikit-learn pandas matplotlib
jupyter nbconvert --to notebook --execute best.ipynb
```

Expected inputs, auto-discovered under `/kaggle/input/**` (archived copies of earlier rounds are excluded deliberately):

| File | Rows | Role |
|---|---|---|
| `train.csv` | 7 409 | `smiles, target, target_type` |
| `test.csv` | 4 940 | `id, smiles, target_type` |
| `PI1M.csv` | ~1 M | unlabelled polymers — pretraining |
| `smile_r3.csv` | ~5.97 M | unlabelled small molecules — breadth + AD reference |

Outputs: `submission.csv`, `predictions_with_uncertainty.csv`.

**Runtime** ≈ 401 min on a single GPU. Set `SMOKE = True` for a ~15 min dry run that exercises every code path at reduced scale.

**Determinism** `SEED = 42` across NumPy, PyTorch and every learner.

### Self-containment

The notebook runs a 14-point compliance audit as its last cell, all passing:

- reads only competition-provided files; no external dataset attached
- pretrained encoder computed inside the notebook; no weights loaded from disk
- no network access at runtime
- feature selection and scaling fitted inside the CV training partition
- CV grouped so a structure never spans folds
- no target-derived feature reaches its own model
- submission schema, row count and ids validated against `test.csv`

---

## Repository layout

```
best.ipynb                        the entire pipeline, top to bottom
submission.csv                    predictions
predictions_with_uncertainty.csv  predictions + conformal intervals + AD flags
README.md
```

The notebook is deliberately one file. Every claim in it is measured in the cell that makes it — the ablations (oligomer augmentation, Koopmans as penalty vs feature, shrinkage λ, corpus weighting) are run, not asserted.

---

## Limitations

- `EPS` (0.874) and `Ei` (0.877) are the weak targets. Both have ~225 labels and `EPS`'s best physical proxy, `Nc`, explains only 84% of its variance — this is a data-quantity ceiling, not a modelling one.
- `Nc` trails polyGNN's reported figure.
- The cross-property channel needs sibling labels to be present. Coverage is 84% of test rows for the minority properties but only 14% for `Egc` and 0% for `Tg`, so those two are earned from structure alone.
- The Bloch operator is tight-binding, not DFT. It supplies a physically-shaped prior, not a ground-truth band structure.
- OOF R² under grouped CV is an honest estimate but is not a leaderboard score; the two differ by split composition.

## References

- Kuenneth & Ramprasad, *polyGNN: Polymer Informatics at Scale with Multitask Graph Neural Networks*, Chem. Mater. 2023 — [paper](https://pubs.acs.org/doi/10.1021/acs.chemmater.2c02991)
- Aldeghi & Coley, *A graph representation of molecular ensembles for polymer property prediction*, Chem. Sci. 2022 — [paper](https://pubs.rsc.org/sc/article/13/35/10486/786353/A-graph-representation-of-molecular-ensembles-for)
- Kuenneth et al., *Polymer informatics with multi-task learning*, Patterns 2021 — [paper](https://www.cell.com/patterns/fulltext/S2666-3899(21)00058-1)
- Ma & Luo, *PI1M: A Benchmark Database for Polymer Informatics*, J. Chem. Inf. Model. 2020
- Ramprasad Group, `canonicalize_psmiles` / `psmiles` — [repo](https://github.com/Ramprasad-Group/psmiles)
