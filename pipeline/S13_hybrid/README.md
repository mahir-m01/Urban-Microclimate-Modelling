# S13 Hybrid ML + DL, and beating the S11 U-Net

Asks where the S11 U-Net's remaining error lies, whether the Phase 1 tree models can fix it (the hybrid), and whether a different network or a longer schedule does better.

| | |
| --- | --- |
| Inputs | the S11 tuned predictions (`results/preds/s11_unet_tuned_seed{0,1,2}/`, not tracked), `pipeline/S11_unet_baseline/outputs/per_case_metrics.csv`, the S06c out-of-fold predictions, the S10 cache |
| Outputs | `outputs/level_test.csv`, `outputs/error_anatomy.csv`, `outputs/fold0_screen.csv`, `outputs/convnext_diagnosis.csv`, `outputs/s13_unet_tuned_e200_seed0_summary.csv` and `_vs_s11.csv`, `outputs/scale_test.csv`, `outputs/final_summary.csv`, `outputs/final_vs_s11.csv`, `outputs/s13_h3_stack*` and `outputs/h3*_feature_importance.csv`; `outputs/runs/<exp_id>/fold<k>.json` and `.csv` |
| Code | `s13_level_pattern.py`, `s13_fold0.py`, `convnext_diagnosis.py`, `s13_vs_s11.py`, `s13_scale.py`, `s13_final.py`, `s13_h3_stack.py`, `configs/`; `ConvNeXtUNet` in `src/urbanwind/dl/nets.py`; tests in `src/urbanwind/dl/tests/test_s13_*.py` and `tests/` |

Status: complete. The level test (H1), the error anatomy, the fold-0 screen, the ConvNeXt diagnosis, the scale test, three seeds of the 200-epoch U-Net over all five folds, and residual stacking (H3) are done. The FiLM network (H2) was not run.

## 1. Where the S11 error is (`s13_level_pattern.py`)

No training: both measurements use the saved predictions of the three S11 tuned seeds.

### Error anatomy (`outputs/error_anatomy.csv`)

- **Realistic cases are the problem.** They hold 84% of the scored cells but 97% to 98% of the squared error. The overall pooled R² is therefore close to the realistic one.
- **Within realistic cases, the error is the pattern, not the level.** 85% to 89% of the realistic squared error is the pattern inside each case; 11% to 15% is the case's overall speed level. The truth itself varies mostly within cases: 70% of the realistic variance is within cases.
- **The error is spread out.** The 20 worst realistic cases hold 28% to 31% of the realistic squared error and 15% of its cells, so no handful of cases explains the score.

### Level test (`outputs/level_test.csv`)

The hybrid idea H1 lets the Phase 1 LightGBM model set each case's level. Done after the fact, each case's prediction is multiplied by one factor so its open-ground mean equals a chosen level. The LightGBM level for a case is the S06c out-of-fold prediction, from a model that never saw its group.

| Level taken from | Pooled R², seeds 0 to 2 | Pooled R², mean of the 3 seeds | Realistic pooled R², mean of the 3 seeds |
| --- | --- | --- | --- |
| U-Net (unchanged) | 0.7092, 0.7122, 0.7111 | 0.7194 | 0.6874 |
| LightGBM | 0.6980, 0.6997, 0.7109 | 0.7111 | 0.6779 |
| Average of the two | 0.7089, 0.7114, 0.7173 | 0.7207 | 0.6887 |
| True mean (upper bound) | 0.7435, 0.7452, 0.7553 | 0.7568 | 0.7297 |

- The U-Net already predicts each case's mean speed as well as LightGBM does: R² of the case means 0.874 to 0.896 for the three seeds, against 0.876 for LightGBM.
- Replacing the level with LightGBM's lowers the score; averaging the two changes it by -0.001 to +0.006.
- Even the true level, which no model has, lifts pooled R² only to about 0.74 to 0.76.

So a level-anchored hybrid cannot add much, and the remaining gains must come from the pattern within cases.

## 2. Candidates for a better network

Each candidate trains fold 0 with seed 0 under the S11 protocol and tuned settings. It counts as better only if it beats the best of the three S11 tuned seeds on fold 0's 97 cases (pooled R² 0.789 to 0.802, median per-case R² 0.567 to 0.633); only then does it get all five folds.

- **`unet_tuned_e200`: the S11 tuned U-Net for 200 epochs instead of 100.** In 14 of the 15 S11 tuned trainings the best inner-validation error came at the last epoch, and it fell by 0.005 RMSE on average over the last 20 epochs (`pipeline/S11_unet_baseline/outputs/runs/s11_unet_tuned_seed*/fold*.csv`).
- **`convnext_unet`: the U-Net layout built from ConvNeXt blocks** (`ConvNeXtUNet`, 5,461,281 parameters against the tuned U-Net's 4,860,313):
  - each block has a 7×7 depthwise convolution with circular padding, a per-cell LayerNorm, a 1×1 expansion to four times the channels with GELU, a 1×1 projection back, and a residual connection with a layer scale;
  - the larger filters see more of the street around each cell at every level;
  - it has no positional or coordinate input, and a float64 test checks that shifts by multiples of 16 cells commute with the network exactly.

### Fold-0 screen (`outputs/fold0_screen.csv`)

| Run | Pooled R² | Median per-case R² | Realistic pooled R² | Mean MAE (u*) | Inner-validation RMSE | Best epoch | T4 minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S11 tuned U-Net, seeds 0 to 2 | 0.7907, 0.7887, 0.8020 | 0.5897, 0.5671, 0.6331 | 0.7856, 0.7838, 0.7970 | 0.4570, 0.4646, 0.4419 | 0.5992, 0.6079, 0.5781 | 99 | 63 to 66 |
| `unet_tuned_e200`, seed 0 | 0.8054 | 0.6321 | 0.7999 | 0.4338 | 0.5695 | 169 | 129 |
| `convnext_unet`, seed 0 | 0.7414 | 0.5120 | 0.7357 | 0.5041 | 0.6627 | 99 | 129 |

- **Training longer helps a little.** Against S11 seed 0 (the same training seed), the 200-epoch U-Net adds 0.015 pooled R² and 0.042 median per-case R² on fold 0. It is the best run on pooled R², realistic pooled R², mean MAE and inner-validation RMSE, and ties the best S11 seed on median per-case R². Its inner-validation error stopped improving at epoch 169, so a longer schedule would add little more. It passes the screen and is being trained on the other four folds.
- **The ConvNeXt U-Net is clearly worse**, on every measure and on both subsets, and took twice as long to train.

### Why the ConvNeXt U-Net falls short (`convnext_diagnosis.py`, `outputs/convnext_diagnosis.csv`)

Its training loss is higher than the U-Net's at every epoch of the fold-0 run (0.592 against 0.532 at the last epoch, `outputs/runs/`), so it underfits rather than overfits. Short controlled runs (400 steps of 16 patches of 128×128 cells, the same patches in the same order, S11 tuned optimiser settings) change one thing at a time:

| Variant | Parameters | Training loss, last 100 steps |
| --- | --- | --- |
| S11 tuned U-Net (reference) | 4,860,313 | 0.8579 |
| ConvNeXt U-Net as trained | 5,461,281 | 0.8956 |
| Layer scale 1 at the start | 5,461,281 | 0.8898 |
| No LayerNorm before the output | 5,461,217 | 0.9149 |
| Three times the learning rate | 5,461,281 | 0.9060 |
| No gradient clipping | 5,461,281 | 0.9011 |
| GroupNorm instead of LayerNorm in the blocks | 5,461,281 | 0.8969 |
| Two ConvNeXt blocks per decoder level | 6,185,441 | 0.8963 |
| Dense 3×3 convolutions to merge the skips | 7,639,681 | 0.8757 |

- The optimiser settings (learning rate, clipping), the layer-scale start, the normalisation type and the decoder depth are not the cause.
- Only dense 3×3 convolutions help: putting the U-Net's convolution block back where each skip is merged closes about half the gap, with 40% more parameters, and still trails the U-Net.
- The depthwise convolution at the heart of the ConvNeXt block filters each channel on its own and mixes channels only within a cell. For rebuilding sharp flow patterns at building edges, the dense 3×3 convolutions of the U-Net, which mix all channels across neighbouring cells, fit better at an equal budget.

## 3. The 200-epoch U-Net on all five folds (`s13_vs_s11.py`)

Seed 0 of `unet_tuned_e200` on the 482 development cases (`outputs/s13_unet_tuned_e200_seed0_summary.csv`), next to the three S11 tuned seeds:

| Run | Pooled R² | Median per-case R² | Mean MAE (u*) | Mean ΔP95 (u*) | Realistic pooled R² |
| --- | --- | --- | --- | --- | --- |
| S11 tuned U-Net, seeds 0, 1, 2 | 0.7092, 0.7122, 0.7111 | 0.5662, 0.5597, 0.5953 | 0.4856, 0.4840, 0.4720 | -0.4571, -0.4569, -0.4385 | 0.6768, 0.6799, 0.6776 |
| `unet_tuned_e200`, seed 0 | 0.7193 | 0.6016 | 0.4641 | -0.4127 | 0.6863 |

Against S11 seed 0, which differs only in the schedule length, case by case with 95% intervals from 2,000 resamples of whole cities and layout families (`outputs/s13_unet_tuned_e200_seed0_vs_s11.csv`):

| Subset | Pooled R² | Median per-case R² | Mean MAE (u*) | Cases with higher R² | Folds better (pooled R²) |
| --- | --- | --- | --- | --- | --- |
| All | +0.0101 [0.0024, 0.0177] | +0.0355 [0.0268, 0.0629] | -0.0216 [-0.0298, -0.0148] | 81% | 3 of 5 |
| Idealized | +0.0140 [0.0110, 0.0203] | +0.0613 [0.0355, 0.1063] | -0.0382 [-0.0494, -0.0304] | 96% | 5 of 5 |
| Realistic | +0.0096 [0.0002, 0.0183] | +0.0222 [0.0064, 0.0416] | -0.0108 [-0.0180, -0.0039] | 71% | 3 of 5 |

- The longer schedule beats every S11 seed on pooled R², median per-case R², mean MAE and mean ΔP95. On its own it matches the pooled R² of the three S11 seeds averaged (0.7194, `outputs/level_test.csv`).
- The gain is largest on the idealized layouts. On the realistic cities it is smaller, and its pooled-R² interval only just excludes zero.
- These intervals come from one training seed on each side; section 4 adds the other two seeds.
- Fold 3 stopped early, at epoch 176, after 30 epochs without improvement; the others ran all 200 epochs, with their best epochs at 169 to 199.

## 4. Final: three seeds of the 200-epoch U-Net (`s13_final.py`)

Seeds 1 and 2 of `unet_tuned_e200` were trained on all five folds as well. Each seed beats the S11 run with the same seed on pooled R², median per-case R² and mean MAE, with intervals that exclude zero (`outputs/s13_unet_tuned_e200_seed{0,1,2}_vs_s11.csv`).

The three seeds of each recipe are averaged cell by cell (`outputs/final_summary.csv`):

| Model | Pooled R² | Median per-case R² | Mean MAE (u*) | Mean ΔP95 (u*) | Realistic pooled R² | Realistic median per-case R² |
| --- | --- | --- | --- | --- | --- | --- |
| S11 tuned U-Net, seeds 0, 1, 2 | 0.7092, 0.7122, 0.7111 | 0.5662, 0.5597, 0.5953 | 0.4856, 0.4840, 0.4720 | -0.4571, -0.4569, -0.4385 | 0.6768, 0.6799, 0.6776 | 0.5234, 0.5227, 0.5437 |
| S11 tuned U-Net, mean of 3 seeds | 0.7194 | 0.5896 | 0.4728 | -0.4634 | 0.6874 | 0.5419 |
| 200-epoch U-Net, seeds 0, 1, 2 | 0.7193, 0.7282, 0.7223 | 0.6016, 0.6072, 0.6239 | 0.4641, 0.4589, 0.4547 | -0.4127, -0.4020, -0.3839 | 0.6863, 0.6961, 0.6891 | 0.5457, 0.5597, 0.5635 |
| **200-epoch U-Net, mean of 3 seeds** | **0.7339** | **0.6288** | **0.4502** | **-0.4149** | **0.7023** | **0.5731** |

The two ensembles case by case, with 95% intervals from 2,000 resamples of whole cities and layout families (`outputs/final_vs_s11.csv`):

| Subset | Pooled R² | Median per-case R² | Mean MAE (u*) | Cases with higher R² | Folds better |
| --- | --- | --- | --- | --- | --- |
| All | +0.0145 [0.0071, 0.0241] | +0.0392 [0.0275, 0.0619] | -0.0226 [-0.0294, -0.0172] | 86% | 5 of 5 |
| Idealized | +0.0118 [0.0092, 0.0175] | +0.0754 [0.0403, 0.0995] | -0.0361 [-0.0449, -0.0298] | 97% | 5 of 5 |
| Realistic | +0.0149 [0.0065, 0.0258] | +0.0311 [0.0146, 0.0440] | -0.0138 [-0.0205, -0.0082] | 78% | 5 of 5 |

- The longer schedule is a consistent gain: every seed of the 200-epoch U-Net scores above every S11 seed, and the ensembles differ on every measure and every fold.
- The three-seed mean of the 200-epoch U-Net is the strongest model on the development set. The frozen test set has not been read.

## 5. At what scale is the remaining error? (`s13_scale.py`, `outputs/scale_test.csv`)

The networks predict every open-ground cell. Averaging both truth and prediction over square blocks before scoring shows where the error lives: fine detail and simulation noise would vanish quickly with the block size, while a street-scale pattern error would not.

| Block size | Realistic pooled R², S11 seed 0 | Realistic pooled R², 200-epoch U-Net | Idealized pooled R², 200-epoch U-Net |
| --- | --- | --- | --- |
| 1 m (each cell) | 0.6768 | 0.6863 | 0.9489 |
| 4 m | 0.6873 | 0.6960 | 0.9569 |
| 16 m | 0.7149 | 0.7197 | 0.9715 |

Even over 16 m blocks, about the size of a city block, the realistic R² rises by only about 0.03 to 0.04. The remaining error is in the street-scale pattern of cities the network has not seen, not in fine detail. (Idealized cases are stored at 0.5 m, so their 1 m row averages 2 × 2 cells.)

## 6. H3: residual stacking with the Phase 1 pixel features (`s13_h3_stack.py`)

A LightGBM model learns the final network's error cell by cell (residual = truth - network prediction) from the network's out-of-fold prediction and the S07 pixel features. The hybrid prediction is the network prediction plus the predicted residual. As in S07, the stacker for outer fold k trains on 3,000 random open-ground cells of each case in the other four folds and predicts every open cell of the fold-k cases. Three variants were run:

| Model | Pooled R² | Median per-case R² | Mean MAE (u*) | Mean ΔP95 (u*) | Realistic pooled R² |
| --- | --- | --- | --- | --- | --- |
| Final U-Net, mean of 3 seeds | 0.7339 | 0.6288 | 0.4502 | -0.4149 | 0.7023 |
| H3, all S07 features, S07 tree settings | 0.7178 | 0.6291 | 0.4596 | -0.3338 | 0.6838 |
| H3, per-cell features only | 0.7214 | 0.6226 | 0.4603 | -0.3281 | 0.6880 |
| H3, per-cell features, small model (200 trees, 15 leaves) | 0.7234 | 0.6252 | 0.4578 | -0.3293 | 0.6905 |

Against the final U-Net, case by case, with 95% intervals from 2,000 resamples of whole cities and layout families (`outputs/s13_h3_stack*_vs_s13_unet_tuned_e200_mean3.csv`):

| Variant | Pooled R² | Mean MAE (u*) | Realistic cases with higher R² |
| --- | --- | --- | --- |
| All S07 features | -0.0161 [-0.0277, -0.0028] | +0.0094 [0.0043, 0.0138] | 34% |
| Per-cell features only | -0.0125 [-0.0341, 0.0069] | +0.0102 [0.0032, 0.0170] | 29% |
| Per-cell features, small model | -0.0104 [-0.0334, 0.0082] | +0.0077 [0.0012, 0.0147] | 52% |

- **No variant improves pooled R² or MAE.** Each costs about 0.01 to 0.016 pooled R², and each raises the mean MAE, with intervals that exclude zero.
- **Every variant reduces the under-prediction of peak speeds**, from a mean ΔP95 of -0.41 to -0.33 u*, about a fifth. The stacker corrects the network's bias at fast cells but adds error elsewhere.
- **With the case descriptors, the stacker leans on them most** (frontal area index 15%, plan area density 9% of the gain; `outputs/h3_feature_importance.csv`). It learns per-case corrections that do not carry over to unseen cities, the same lesson as the level test in section 1.

The hybrid therefore trades mean accuracy for better peaks. The final model stays the 200-epoch U-Net; the stacker is an option only where peak speeds matter more than the mean field.

## Running it

```bash
uv run python pipeline/S13_hybrid/s13_level_pattern.py      # needs the S11 tuned predictions
uv run python -m urbanwind.dl.train --config pipeline/S13_hybrid/configs/convnext_unet.yaml --seed 0 --folds 0
uv run python -m pipeline.S13_hybrid.s13_fold0               # scores every finished candidate on fold 0
uv run python -m pipeline.S13_hybrid.convnext_diagnosis      # about 30 minutes on an Apple M-series GPU
uv run python -m pipeline.S13_hybrid.s13_vs_s11              # needs all five folds of the candidate's predictions
uv run python -m pipeline.S13_hybrid.s13_scale               # block-averaged R² of chosen runs
uv run python -m pipeline.S13_hybrid.s13_final               # the three-seed ensembles; needs every seed's predictions
uv run python -m pipeline.S13_hybrid.s13_h3_stack [--local] [--small]   # H3 stacker (LightGBM, no PyTorch), after s13_final
uv run python -m pipeline.S13_hybrid.s13_vs_s11 s13_h3_stack s13_unet_tuned_e200_mean3   # H3 against the final U-Net
```

The candidates were trained on Kaggle T4 GPUs with `pipeline/S11_unet_baseline/kaggle_runner.py`, and their fold-0 predictions rebuilt locally with `--repredict`.

## Limitations

- H3 trains on out-of-fold network predictions, as issue #38 asks; those predictions for the other folds came from networks that had seen fold k's cases, a second-order dependence that is standard in stacking.
- The FiLM network (H2) was not run.
- The level test rescales finished predictions; a network trained to predict U/U_mean could behave differently, but the true-mean upper bound limits what any level correction can add.
- The diagnosis runs are short (400 steps on 128×128 patches); they rank the variants but do not show where each would end after full training.
- The ConvNeXt U-Net was not tuned; it uses the settings Optuna chose for the U-Net. The diagnosis shows the learning rate and clipping are not what holds it back.
- The fold-0 screen uses one seed and 97 cases. The S11 seed spread on those cases is wide for the median per-case R² (0.567 to 0.633), so only a clear win moves on to all five folds.
