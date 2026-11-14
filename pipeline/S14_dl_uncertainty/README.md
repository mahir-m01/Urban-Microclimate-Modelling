# S14 — Uncertainty and Calibration

**Owner:** Ayush Aryan · **Issue:** #39 · **Depends on:** S11, S13

---

## Goal

Give every predicted wind map an uncertainty map whose 90 % intervals stay calibrated on unseen cities and layout families.

---

## Status

| Task | Status |
|---|---|
| `src/urbanwind/dl/uq.py` — `S13UNet`, conformal, ensemble, MC dropout, Spearman, selective | ✅ Done |
| Tests — 45/45 passing (`uv run pytest src/urbanwind/dl/tests/test_uq.py`) | ✅ Done |
| S07 conformal validation (CPU, per-case MAE proxy) | ✅ Done |
| S13 ensemble + MC-dropout predictions (requires GPU, `M/archive (17)`) | 🔲 Pending (see below) |
| Coverage analysis, reliability diagram, figures | 🔲 Pending (after ensemble preds) |
| `lo` / `hi` saved in prediction `.npz` files | 🔲 Pending |

---

## Inputs

| File | Description |
|---|---|
| `M/archive (17)/s13_unet_tuned_e200_seed{0,1,2}/fold{0-4}.pt` | 15 S13 checkpoints (3 seeds × 5 folds) |
| `pipeline/S07_pixel_model/outputs/per_case_metrics.csv` | Per-case MAE / R² for the S07 pixel GBM (used as conformal proxy) |
| `data/splits.json` | 5-fold group split (482 development cases, 93 groups) |

---

## Method

### 1 — Deep ensemble (3 seeds)

For fold `k`, load the fold-`k` checkpoint from each of the 3 seeds and run inference on the held-out cases.
Mean and cell-wise standard deviation across the 3 predictions give the ensemble uncertainty.

```
mean_k, std_k = deep_ensemble_predict(
    ckpt_paths=[seed0/fold{k}.pt, seed1/fold{k}.pt, seed2/fold{k}.pt],
    input_data=x,          # [B, 5, H, W]
)
```

### 2 — MC dropout (seed 0, 30 passes)

Load seed-0 fold-`k` checkpoint, switch only `Dropout2d` layers to training mode, and run 30 stochastic forward passes.

```python
model = load_s13_checkpoint("seed0/fold{k}.pt")
mean_mc, std_mc = mc_dropout_predict(model, x, n_passes=30)
```

> **GroupNorm note:** GroupNorm has no running statistics, so switching dropout on does not change normalisation. Verified against `nets.py` and the S13 architecture.

### 3 — Split-conformal calibration

For each fold `k`, the threshold is calibrated on the **absolute errors from the other 4 folds**:

```python
cal_errors = |y_true_all_other_folds − y_pred_all_other_folds|   # pixel-level
threshold_k = group_conformal_calibrate(cal_errors, alpha=0.10)  # 90 % interval
```

The conformal interval for a predicted map is then `[ŷ − q, ŷ + q]`. This is a **group-aware** calibration: each fold holds out whole cities and idealized layout families, so the threshold generalises to unseen groups.

### 4 — Coverage analysis

Computed by `compute_coverage()` / `per_case_coverage()`:

- Overall (pooled cells)
- By subset: **idealized** and **realistic**
- By distance-to-wall bin: `[0–1], [1–2], [2–4], [4–8], [8–16], [16–32], [>32]` m
- Spread of per-case coverage (cells within a case are correlated — pooled pixel count overstates certainty)

**Definition of done:** 90 % intervals cover 85 %–95 % of held-out cells.

### 5 — Spearman correlation

```python
rho = compute_spearman(uncertainty_map, abs_error_map)
```

Computed per case and pooled. **Definition of done:** ρ > 0.3.

### 6 — Selective prediction

Error after dropping the 10 % most uncertain cells:

```python
mae_selective = selective_prediction_error(y_true, y_pred, uncertainty, drop_pct=0.10)
```

### 7 — Uncertainty maps

Saved beside error maps for the same held-out realistic cases used in S07 / S11:
`AU-Mel-U15_d90`, `SG-Boo-U1_d30`, and the S07 worst / median / best realistic cases.

### 8 — Save `lo` and `hi` in `.npz` files

```python
np.savez_compressed(
    path / f"{case}.npz",
    y_true=true_map, y_pred=pred_map, mask=~valid,
    lo=pred_map - threshold, hi=pred_map + threshold,
)
```

The evaluator ignores `lo` / `hi`; coverage is computed in `uq.py`.

---

## S14 UNet architecture (`S13UNet`)

The S13 Kaggle checkpoints use a different layer layout from the `nets.UNet` class (S11 baseline).
`S13UNet` in `uq.py` matches the checkpoint keys exactly:

| Component | Layer names | Notes |
|---|---|---|
| Encoder | `down.{0-3}` | ConvBlock(CircConv, GN, ReLU) × 2, Dropout2d(0.08) |
| Bottleneck | `mid` | Same ConvBlock structure |
| Channel projection | `up_conv.{0-3}` | 3×3 circular conv; halves channels |
| Decoder | `up_block.{0-3}` | ConvBlock on cat(proj, skip) |
| Output | `head` | 1×1 conv → Softplus |

Input channels (5): `H/16`, `building_mask`, `dist_to_wall/16`, `cos(angle)`, `sin(angle)` (from S10).
Widths: `[24, 48, 96, 192]`, bottleneck `384`, groups `8`, dropout `0.08`.

---

## S07 conformal validation (CPU, done)

The S07 per-case MAE (`pipeline/S07_pixel_model/outputs/per_case_metrics.csv`) is used as a proxy for pixel-level errors to validate the conformal code before GPU predictions are available.

| Fold | Calibration cases | Test coverage (90 % interval on case MAE) |
|---|---|---|
| 0 | folds 1–4 | ≥ 85 % ✅ |
| 1 | folds 0,2–4 | ≥ 85 % ✅ |
| 2 | folds 0–1,3–4 | ≥ 85 % ✅ |
| 3 | folds 0–2,4 | ≥ 85 % ✅ |
| 4 | folds 0–3 | ≥ 85 % ✅ |

Spearman between per-case MAE and (1 − R²): **> 0.3** ✅ (verified in `TestConformalOnS07Metrics`).

---

## Reproduce (CPU steps — no GPU needed)

```bash
uv sync
uv run pytest src/urbanwind/dl/tests/test_uq.py -v
```

## Reproduce (GPU steps — requires S13 weights)

```bash
# Step 1: ensemble predictions (3 seeds × 5 folds = 15 runs)
# Run on Kaggle with M/archive (17) attached as a dataset
uv run python pipeline/S14_dl_uncertainty/run_s14_ensemble.py

# Step 2: MC dropout (seed 0, 30 passes, all 5 folds)
uv run python pipeline/S14_dl_uncertainty/run_s14_mc_dropout.py

# Step 3: coverage analysis and figures (CPU)
uv run python pipeline/S14_dl_uncertainty/run_s14_analysis.py
```

---

## Caveats

1. **Cell correlation:** cells within one case are strongly spatially correlated. Pooled pixel coverage overstates certainty; always report the spread of per-case coverage.
2. **Subset calibration:** idealized and realistic errors differ in scale. Calibrate and report coverage separately for each subset.
3. **Three ensemble members:** gives a rough spread. The conformal step sets the final interval width; the ensemble spread provides the *shape* of the uncertainty map but not a calibrated scale.
4. **GroupNorm:** has no running statistics, so activating only `Dropout2d` for MC dropout does not change normalisation. Verified for S13UNet.
5. **XGBoost tuning instability:** noted in S06c — hyperparameters chosen on training cities do not always carry over to unseen cities. The same risk applies to conformal thresholds; using whole-group folds mitigates but does not eliminate it.

---

## Files

```
pipeline/S14_dl_uncertainty/
├── README.md                        ← this file
├── figures/                         ← reliability diagram, coverage plots, uncertainty maps
│   └── (generated by run_s14_analysis.py)
├── run_s14_ensemble.py              ← ensemble predictions (GPU, Kaggle)
├── run_s14_mc_dropout.py            ← MC dropout predictions (GPU, Kaggle)
└── run_s14_analysis.py              ← coverage / Spearman / figures (CPU)

src/urbanwind/dl/
├── uq.py                            ← S13UNet, conformal, ensemble, MC dropout, coverage, Spearman
└── tests/test_uq.py                 ← 45 tests, all passing

results/preds/
├── s14_ensemble/                    ← {case}.npz with y_true, y_pred, mask, lo, hi (pending)
└── s14_mc_dropout/                  ← {case}.npz with y_true, y_pred, mask, lo, hi (pending)
```

