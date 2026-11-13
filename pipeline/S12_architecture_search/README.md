# S12 Architecture Search

Tests whether alternative deep-learning architectures predict the pedestrian-level wind pattern $U/u^*$ better than the S11 periodic U-Net under strictly matched parameter and training budgets.

| | |
| --- | --- |
| **Inputs** | S10 cache (`data/processed/dl/`), `data/splits.json`, S11 baseline weights / predictions |
| **Outputs** | `configs/` (`res_unet.yaml`, `attn_unet.yaml`, `fno.yaml`); `overfit_check_<model>.json`; `outputs/summary.csv`, `outputs/vs_s11_seed1.csv`, figures; `kaggle_jobs.json` |
| **Code** | `src/urbanwind/dl/nets.py` (added `ResBlock`, `ResUNet`, `AttentionGate`, `AttnUNet`, `SpectralBlock`, `FNO`); tests in `src/urbanwind/dl/tests/test_s12_nets.py`; stage scripts `param_counts.py`, `overfit_check.py`, `s12_results.py` |

---

## 1. Candidate Architectures & Parameter Budget

All candidate models are implemented in `src/urbanwind/dl/nets.py` and registered via `@register("<name>")`. Convolutions of kernel size $\ge 3$ use circular padding and upsampling uses `upsample_periodic` to ensure exact periodic shift-equivariance across neighbourhood boundaries. Outputs pass through `softplus` to enforce positive wind speed predictions.

Baseline: **Tuned S11 periodic U-Net** (`widths: [24, 48, 96, 192], bottleneck: 384, dropout: 0.08`, **4,860,313 parameters**).
Every architecture is constrained to be within $\pm 25\%$ of the baseline budget (3,645,235 to 6,075,391 parameters).

| Architecture | Registry Name | Key Components | Parameters | Ratio vs Baseline | Multiple |
| --- | --- | --- | --- | --- | --- |
| **Tuned U-Net (Baseline)** | `unet` | 4-level U-Net, double circular convs, GroupNorm, spatial dropout | 4,860,313 | 100.0% | 16 |
| **Periodic Res-U-Net** | `res_unet` | Residual blocks (two circular 3x3 convs + GroupNorm + ReLU with 1x1 projection shortcuts) | 5,057,377 | 104.0% (+4.0%) | 16 |
| **Periodic Attention U-Net** | `attn_unet` | Attention gates (Oktay et al. 2018) on skip connections using 1x1 convs and periodic upsampling | 5,009,717 | 103.1% (+3.1%) | 16 |
| **Fourier Neural Operator (FNO)** | `fno` | 1x1 lifting, 4 spectral conv layers (float32 2D rfft2/irfft2 + complex weights + 1x1 bypass + GELU), 1x1 projection | 4,728,337 | 97.3% (−2.7%) | 1 |

All three models fall within $4.1\%$ of the target budget.

---

## 2. Equal Budget Rules

To isolate the impact of architecture design from hyperparameter advantages:
1. **Identical Optimisation & Schedule:**
   - AdamW with base learning rate $3.6 \times 10^{-4}$, weight decay $6.5 \times 10^{-5}$, min lr $1.0 \times 10^{-5}$ (cosine annealing over 100 epochs).
   - Batch size 16, 70 gradient steps per epoch, gradient clipping at 1.0.
   - Masked L1 loss on open ground cells.
2. **Identical Data & Folds:**
   - 256×256 wrap-around training patches with border margin 16.
   - Same grouped 5-fold cross-validation (`splits.json`). The frozen test set is never read.
   - Same 15% inner-validation set held back per fold with early stopping patience 30 on balanced validation RMSE.
   - Exponential Moving Average (EMA, decay 0.999 with warm-up) evaluated after every epoch.
3. **No Per-Architecture Hyperparameter Tuning:**
   - No architecture-specific learning rate or decay tuning was conducted.

---

## 3. FNO Mode Representation & Tiling

- **Grid-cell mode counting:**
  Modes in FNO are indexed in grid cells. In a domain of $N$ cells with grid spacing $\Delta x = 1\text{ m}$, Fourier mode index $k$ corresponds to physical wavelength $\lambda = \frac{N \Delta x}{k}$.
  For example, mode 16 represents a 16 m wave on a 256-cell patch, but a 64 m wave on a 1,024-cell full domain.
- **Wrap-around tiling inference:**
  To guarantee that frequency modes correspond to the exact same physical spatial wavelengths during inference as during training, the FNO config enforces `infer_window: 256` and `infer_border: 16`. Full domains are inferred via overlapping 256×256 wrap-around tiles matching the training distribution.
- **Float32 Spectral Execution under Mixed Precision:**
  PyTorch AMP autocast is explicitly disabled inside `SpectralBlock`. FFT and complex multiplication run in `float32` because half-precision (fp16) cuFFT restricts sizes to powers of two and risks precision loss, whereas float32 FFT supports arbitrary non-power-of-two domain sizes.
- **AMP Disabled for FNO (`amp: false`):**
  FNO uses AMP disabled because its complex-valued spectral gradients are incompatible with the current PyTorch GradScaler unscale operation. Res-U-Net and Attention U-Net retain AMP enabled. This is an implementation compatibility adjustment, not an architectural change.

---

## 4. Design Sanity Checks

| Check | Tool / Test | Result |
| --- | --- | --- |
| **Parameter Budget** | `param_counts.py` | All candidates within 4.1% of baseline (4.73M to 5.06M). |
| **Periodicity & Shift Equivariance** | `test_s12_nets.py` | Machine-precision shift equivariance ($< 10^{-10}$ in float64) under periodic roll for all candidates (including non-multiple shifts of 7 and 13 cells for FNO). |
| **Autocast & Precision** | `test_s12_nets.py` | FNO executes under `torch.autocast` on CPU/bfloat16 with an internal float32 spectral block. Under CUDA fp16 the GradScaler cannot unscale complex gradients, so the FNO is trained with `amp: false` (section 3). |
| **Overfit 4 patches (sanity check)** | `overfit_check.py` | 30-step CPU smoke check confirms rapid loss decrease without collapse (`res_unet` $0.44 \to 0.25$, `attn_unet` $1.53 \to 0.25$, `fno` $0.57 \to 0.24$). Full 2,000-step overfit validation runs on Kaggle GPU. |

---

## 5. Running the Pipeline

### Local Sanity & Verification
```bash
# Verify model parameters and budget compliance
uv run python pipeline/S12_architecture_search/param_counts.py

# Run unit and integration tests
URBANWIND_DEVICE=cpu uv run pytest src/urbanwind/dl/tests/test_s12_nets.py src/urbanwind/dl/tests/test_s12_results.py -v

# Run fast CPU overfit sanity check (30 steps)
URBANWIND_DEVICE=cpu uv run python pipeline/S12_architecture_search/overfit_check.py --model all --steps 30
```

### Full Training on Kaggle GPU
The job queue in `kaggle_jobs.json` defines the execution order for `pipeline/S11_unet_baseline/kaggle_runner.py`:
1. **Screening Phase:** Fold 0, Seed 0 for all 3 models (`res_unet`, `attn_unet`, `fno`).
2. **Remaining Folds:** Folds 1–4, Seed 0 for all models (completed).
3. **Multi-Seed Confirmation:** Seeds 1 and 2 for the top 2 performing architectures across all 5 folds (Seed 1 completed; Seed 2 pending review).

To run a single fold locally or on GPU:
```bash
uv run python -m urbanwind.dl.train --config pipeline/S12_architecture_search/configs/res_unet.yaml --seed 0 --folds 0
```

### Scoring and Comparison Tooling
Once training completes:
```bash
# Score finished runs and generate outputs/summary.csv and outputs/vs_s11_seed1.csv
uv run python pipeline/S12_architecture_search/s12_results.py

# Plot visual comparison between S12 candidates and S11 baseline range
uv run python pipeline/S12_architecture_search/plot_comparison.py
```

---

## 6. Experimental Results

All candidate architectures were trained under the equal-budget protocol (section 2) across 5 outer cross-validation folds. Seed 0 was evaluated for all 3 architectures (`attn_unet`, `res_unet`, `fno`). Multi-seed confirmation (seed 1) was subsequently executed for the top two performing candidates (`attn_unet` and `res_unet`). All evaluations were scored on the development set through `urbanwind.evaluate`. Raw numbers: `outputs/summary.csv`.

Reading the table: Every row covers all 482 development cases (190 idealized, 292 realistic). The S11 rows are copied from `pipeline/S11_unet_baseline/outputs/summary.csv` (grouped split) and GPU minutes represent the sum of the fold training durations.

| Model | Seed | Parameters | GPU minutes | Pooled $R^2$ | Median per-case $R^2$ (all) | Median per-case $R^2$ (idealized) | Median per-case $R^2$ (realistic) | Mean $\Delta P_{95}$ | Inference s / case |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| S11 tuned U-Net (ref) | 0 | 4,860,313 | 327.8 | 0.7092 | 0.5662 | 0.6673 | 0.5234 | −0.4571 | n/a |
| S11 tuned U-Net (ref) | 1 | 4,860,313 | 329.9 | 0.7122 | 0.5597 | 0.6210 | 0.5227 | −0.4569 | n/a |
| S11 tuned U-Net (ref) | 2 | 4,860,313 | 337.2 | 0.7111 | 0.5953 | 0.6828 | 0.5437 | −0.4385 | n/a |
| Attention U-Net | 0 | 5,009,717 | 412.2 | 0.6937 | 0.5291 | 0.6287 | 0.4988 | −0.4813 | 0.0442 |
| Attention U-Net | 1 | 5,009,717 | 411.1 | 0.6786 | 0.5259 | 0.6048 | 0.4995 | −0.4880 | 0.0661 |
| Res-U-Net | 0 | 5,057,377 | 359.5 | 0.6830 | 0.5288 | 0.6274 | 0.4998 | −0.6733 | 0.0382 |
| Res-U-Net | 1 | 5,057,377 | 354.7 | 0.6710 | 0.5302 | 0.5966 | 0.5031 | −0.6989 | 0.0539 |
| FNO | 0 | 4,728,337 | 251.1 | 0.5578 | 0.3552 | 0.3808 | 0.3434 | −0.6099 | 0.0245 |

### Multi-Seed Aggregate Summary

| Model | Seeds | Pooled $R^2$ mean (range) | Median case $R^2$ all (mean) | Mean $\Delta P_{95}$ (mean) |
| --- | :---: | :---: | :---: | :---: |
| Attention U-Net | 2 | 0.6862 (0.6786–0.6937) | 0.5275 | −0.4847 |
| Res-U-Net | 2 | 0.6770 (0.6710–0.6830) | 0.5295 | −0.6861 |
| FNO | 1 (seed 0) | 0.5578 | 0.3552 | −0.6099 |
| S11 tuned U-Net (ref) | 3 | 0.709–0.712 (0.7108) | about 0.566 (seed 0: 0.5662, seed 1: 0.5597, seed 2: 0.5953; mean 0.5737) | −0.4508 (−0.4571 to −0.4385) |

*Inference seconds per case differ between seeds (0.0442 vs 0.0661 for Attention U-Net) because the sessions used different GPUs, so timing is only a rough guide.*

### S12 vs S11 Group-Bootstrap Comparison (Seed 1)

Paired group-bootstrap comparisons were conducted against `s11_unet_tuned_seed0` using 2,000 cluster bootstrap resamples across whole `split_group` units. Results are saved in `outputs/vs_s11_seed1.csv` (seed 1 only):

| Model (Seed 1) | Subset | Metric | Difference vs S11 | 95% Bootstrap CI | Share Cases Better | Folds Better (of 5) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention U-Net** | All | Pooled $R^2$ | −0.0305 | [−0.0589, −0.0037] | 29.05% (140/482) | 1 / 5 |
| | All | Median case $R^2$ | −0.0403 | [−0.0547, −0.0160] | 29.05% (140/482) | 1 / 5 |
| | All | Mean $\Delta P_{95}$ | −0.0309 | [−0.0528, −0.0096] | 29.05% (140/482) | 0 / 5 |
| | Idealized | Pooled $R^2$ | −0.0053 | [−0.0113, −0.0019] | 26.32% (50/190) | 1 / 5 |
| | Idealized | Median case $R^2$ | −0.0626 | [−0.0844, −0.0173] | 26.32% (50/190) | 0 / 5 |
| | Idealized | Mean $\Delta P_{95}$ | −0.0364 | [−0.0622, −0.0122] | 26.32% (50/190) | 0 / 5 |
| | Realistic | Pooled $R^2$ | −0.0342 | [−0.0655, −0.0039] | 30.82% (90/292) | 1 / 5 |
| | Realistic | Median case $R^2$ | −0.0239 | [−0.0489, −0.0012] | 30.82% (90/292) | 0 / 5 |
| | Realistic | Mean $\Delta P_{95}$ | −0.0273 | [−0.0584, +0.0049] | 30.82% (90/292) | 1 / 5 |
| **Res-U-Net** | All | Pooled $R^2$ | −0.0382 | [−0.0504, −0.0277] | 25.52% (123/482) | 0 / 5 |
| | All | Median case $R^2$ | −0.0360 | [−0.0485, −0.0103] | 25.52% (123/482) | 0 / 5 |
| | All | Mean $\Delta P_{95}$ | −0.2418 | [−0.2714, −0.2086] | 25.52% (123/482) | 0 / 5 |
| | Idealized | Pooled $R^2$ | −0.0313 | [−0.0423, −0.0226] | 17.89% (34/190) | 0 / 5 |
| | Idealized | Median case $R^2$ | −0.0708 | [−0.0934, −0.0230] | 17.89% (34/190) | 0 / 5 |
| | Idealized | Mean $\Delta P_{95}$ | −0.1792 | [−0.2135, −0.1426] | 17.89% (34/190) | 0 / 5 |
| | Realistic | Pooled $R^2$ | −0.0392 | [−0.0546, −0.0276] | 30.48% (89/292) | 0 / 5 |
| | Realistic | Median case $R^2$ | −0.0203 | [−0.0427, −0.0032] | 30.48% (89/292) | 0 / 5 |
| | Realistic | Mean $\Delta P_{95}$ | −0.2824 | [−0.3237, −0.2436] | 30.48% (89/292) | 0 / 5 |

*Note on $\Delta P_{95}$ directionality: $\Delta P_{95} = P_{95,\text{pred}} - P_{95,\text{true}}$ is a signed peak-speed error where $0$ is ideal. Because all baseline and candidate models operate in the negative regime ($\Delta P_{95} < 0$, under-predicting extreme wind speeds), a negative difference indicates moving algebraically further below zero, establishing a larger under-prediction deficit rather than an improvement.*

### Analysis & Key Findings

- **Paired Group-Bootstrap Comparison (Seed 1):** In the paired group-bootstrap comparison on Seed 1, the paired group-bootstrap 95% intervals exclude zero for both models (seed 1, resampling cases by split group, against a single S11 seed). Note that the Attention U-Net pooled $R^2$ interval vs S11 reaches −0.0037, close to zero.
- **Low Case-Level Win Rates:** Attention U-Net improves upon S11 on only 29.05% of development cases (140/482), while Res-U-Net improves on only 25.52% (123/482). S11 outperforms both candidates on $> 70\%$ of cases.
- **Seed Spread:** S12 seed spread in pooled $R^2$ is 0.015 (Attention U-Net: 0.6937 vs 0.6786) and 0.012 (Res-U-Net: 0.6830 vs 0.6710), against about 0.003 for S11, so these models are less stable across seeds than S11.
- **Consistent Seed 1 Replication:** Across both seeds 0 and 1, Attention U-Net and Res-U-Net maintain median per-case $R^2 \approx 0.526\text{--}0.530$, well below S11's three-seed range ($0.560\text{--}0.595$). Pooled $R^2$ for seed 1 (0.679 for Attention U-Net, 0.671 for Res-U-Net) dropped slightly compared to seed 0 (0.694 and 0.683), confirming that neither candidate matches S11's pooled $R^2$ ($0.709\text{--}0.712$).
- **High-Speed Tail Deficit:** Every model exhibits negative $\Delta P_{95}$, under-predicting extreme wind speeds. However, the deficit is markedly more severe in Res-U-Net ($\Delta P_{95} \approx -0.673$ in seed 0, $-0.699$ in seed 1) than in Attention U-Net ($-0.481$ to $-0.488$) or S11 ($-0.439$ to $-0.457$).
- **FNO Limitations:** The FNO remains the lowest performing architecture (pooled $R^2 = 0.558$, median per-case $R^2 = 0.355$). While fast (251 GPU minutes, 0.0245 s/case), its global frequency mode representation struggles with sharp building discontinuities.

### Status of Multi-Seed Confirmation & Seed 2

The experimental protocol in `kaggle_jobs.json` outlines a 3-seed confirmation schedule (seeds 0, 1, 2) for the two top architectures. Seed 2 has **not yet been started**:
1. **Consistent Negative Result Across Multiple Seeds:** Both seed 0 and seed 1 independently demonstrate that neither Attention U-Net nor Res-U-Net outperforms the baseline S11 periodic U-Net on any headline metric.
2. **Bootstrap Comparison (Seed 1):** The paired group-bootstrap 95% intervals exclude zero for both models (seed 1, resampling cases by split group, against a single S11 seed), though the Attention U-Net pooled $R^2$ interval reaches −0.0037, close to zero.

Seed 2 would cost about 12.7 GPU hours (about 760 GPU minutes).

---

## 7. Definition of Done

At least three architectures compared under equal budgets. Either the winner beats the S11 tuned U-Net on median per-case $R^2$ or mean $\Delta P_{95}$ by more than the seed spread (S11 seeds 0–2: median per-case $R^2 \in [0.560, 0.595]$, mean absolute error $\in [0.472, 0.486]\ u^*$), or the negative result is documented honestly.

**Outcome:** Negative result documented and confirmed across multiple seeds and paired group-bootstrap testing; S11 periodic U-Net remains the best architecture.

---

## 8. Limitations

1. **Equal-budget constraint:** Hyperparameters (learning rate $3.6 \times 10^{-4}$, weight decay $6.5 \times 10^{-5}$) were tuned specifically for the baseline U-Net in S11. FNO or ResNet architectures may prefer different learning rates or schedules; holding them constant is a deliberate protocol choice to ensure equal training budgets, but represents a known limitation for architectures with different optimization dynamics.
2. **Wavelength tiling trade-off:** While 256×256 tiled inference preserves physical wavelength interpretation for FNO, it prevents the model from exploiting global cross-neighbourhood low-frequency modes across multi-kilometer domains in a single pass.
3. **Multi-seed confirmation status:** FNO was evaluated on seed 0 only. Attention U-Net and Res-U-Net have completed seeds 0 and 1; seed 2 has not yet been executed.
4. **FNO numeric precision (`amp: false`):** The FNO was trained with `amp: false`, because fp16 `GradScaler` cannot unscale complex gradients. The other models used AMP, so numeric precision is not identical.
5. **Seed 0 was trained across Kaggle and Colab.** Folds 0-3 ran on Kaggle and fold 4 on Colab. The final seed-0 scoring covers all 482 development cases. A seed-0 group-bootstrap vs S11 was not produced because the seed-0 predictions live on a different machine than the scoring environment.
6. **Approximate GPU minutes and resumed fold 4:** GPU minutes are approximate: folds ran on Kaggle and Colab T4 GPUs, two jobs shared a node on Kaggle, and resumed folds count only the minutes of the final segment. Fold 4 of seed 0 was resumed from checkpoints created on a different machine (same code and configuration).
7. **Inference time benchmarking:** Inference time was measured on a single GPU model in one session and is only comparable among the S12 rows.
