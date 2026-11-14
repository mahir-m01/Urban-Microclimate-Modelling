# S11 Periodic U-Net and the shared trainer

Trains a convolutional network that predicts the pedestrian-level wind speed U/u* over a whole neighbourhood from the S10 inputs, under the same grouped five-fold protocol as Phase 1, and provides the training code every later deep-learning stage reuses.

| | |
| --- | --- |
| Inputs | the S10 cache (`data/processed/dl/`), `data/splits.json` |
| Outputs | `results/preds/<exp_id>/<case>.npz` and `results/checkpoints/<exp_id>/` (not tracked); `outputs/runs/<exp_id>/fold<k>.json` and `.csv`; `outputs/optuna/` (the tuning study); `outputs/summary.csv`, `outputs/vs_s07.csv`, `outputs/fold0_versions.csv`, figures; `S11a_unet_baseline.ipynb` |
| Code | `src/urbanwind/dl/nets.py`, `losses.py`, `train.py`, `datasets.py`; `configs/`, `overfit_check.py`, `tune_optuna.py`, `s11_results.py`, `kaggle_runner.py`; tests in `src/urbanwind/dl/tests/` |

Status: complete. The code is tested, the settings are tuned (see Tuning below), and the results cover three seeds over all five grouped folds, plus a random-split comparison.

## Network

`UNet` in `nets.py`, 8,636,961 parameters with the default widths:
- four levels of 32, 64, 128 and 256 channels and a 512-channel bottleneck; two 3×3 convolutions with GroupNorm and ReLU per block;
- **periodic everywhere**: circular padding in every convolution, and bilinear upsampling that wraps across the edges. A test checks in float64 that shifting the input by a multiple of 16 cells shifts the output by exactly the same amount;
- spatial dropout after the bottleneck and the decoder blocks, but not before the output layer; He initialisation; a softplus output, so speeds stay positive.

New architectures register themselves with `@register("name")` and are built with `build_model(name, in_ch, **kwargs)`.

## Training protocol (`train.py`)

1. **Folds.** Each outer fold trains on the other four grouped folds of `splits.json` and is scored on its own held-out cities and layout families. The frozen test set is never read.
2. **Inner validation.** Whole groups from the training folds are held back for model selection: 15% of the idealized and 15% of the realistic training cases, with no single city or layout family above a quarter of the set (61 to 67 cases per fold, at most 14 from one group). The choice depends on the fold only, so seeds differ in training alone.
3. **Optimisation.**
   - AdamW with cosine decay from the learning rate to 1e-5, weight decay, batch 16;
   - mixed precision with the loss in float32, and gradient clipping at 1.0;
   - masked L1 or MSE loss (`losses.py`), computed on valid open-ground cells only.
4. **Equal budgets.** Every epoch is 70 steps of 256×256 wrap-around patches, so all folds get the same training. Patches are mirrored at random, and patches of 45° cases are also transposed, which keeps the wind at 45°.
5. **Weights and selection.** A moving average of the weights (EMA, decay 0.999 with a warm-up) is validated after every epoch: each held-out case is predicted whole, its RMSE is averaged within each subset, then over the two subsets. The best EMA weights are kept, and training stops after `patience` epochs without improvement. With `inner_val_fraction: 0` the network instead trains on every training case for the full schedule and keeps its final weights.
6. **Records.** Every fold writes the code version (commit and whether the code differed from it), a hash of its settings, its inner-validation cases and its per-epoch history. A stopped run resumes from its last finished epoch and refuses to resume with changed settings.

## Prediction (`datasets.predict_arrays`)

- **One pass over the whole domain.** A side that the network's pooling already divides needs no padding, because circular padding then wraps the domain onto itself. Any other side gets at least 128 cells of its own periodic continuation.
- **Symmetric copies.** The prediction is averaged over the exact copies of the case: the mirror, and at 45° also the transpose.
- **Building cells.** These take the prediction of the nearest open cell, using the building map only.
- **Back to the native grid.** The result is turned back out of the canonical frame. For the cases stored at 0.5 m, the 1 m prediction is spread by periodic bilinear interpolation.

## Design checks (measured)

| Check | Result |
| --- | --- |
| Overfit 4 patches, 2,000 steps (`overfit_check.py`) | masked L1 0.983 → 0.0365, 2% of the mean target |
| Prediction mode, on fold 0's inner-validation cases (balanced RMSE) | whole domain 0.666 (margin 64 or 128 alike); tiled 256-cell windows 0.676 to 0.678. Whole-domain prediction is kept. |
| A perfect 1 m prediction on the 194 cases stored at 0.5 m | copied to sub-cells: median per-case R² 0.987 (worst 0.846), MAE 0.052 u*; bilinear: 0.992 (worst 0.890), MAE 0.030 u*, better in 193 of 194 cases. Bilinear is kept; the rest is the cost of the 1 m grid. |
| Speed per training epoch of the default network | about 95 s on an Apple M3 Pro GPU, about 52 s on a Kaggle T4 |

## Results

`configs/unet_tuned.yaml`, three training seeds, each over all five grouped folds. Every one of the 482 development cases is scored once per seed, out of fold, by the Phase 1 evaluator. The numbers below are in `outputs/summary.csv`, `outputs/vs_s07.csv` and `outputs/fold0_versions.csv`; `S11a_unet_baseline.ipynb` presents them.

| All 482 cases | Seed 0 | Seed 1 | Seed 2 | S07 pixel model |
| --- | --- | --- | --- | --- |
| Pooled R² | 0.709 | 0.712 | 0.711 | 0.591 |
| Pooled R², idealized (190) | 0.935 | 0.937 | 0.944 | 0.876 |
| Pooled R², realistic (292) | 0.677 | 0.680 | 0.678 | 0.550 |
| Median per-case R² | 0.566 | 0.560 | 0.595 | 0.304 |
| Median per-case R², idealized | 0.667 | 0.621 | 0.683 | 0.418 |
| Median per-case R², realistic | 0.523 | 0.523 | 0.544 | 0.247 |
| Mean absolute error | 0.486 u* (0.100 m/s) | 0.484 u* (0.100 m/s) | 0.472 u* (0.098 m/s) | 0.611 u* (0.126 m/s) |

**Against S07, with uncertainty** (`outputs/vs_s07.csv`, 95% intervals from resampling whole cities and layout families 2,000 times):
- **Pooled R²:** higher by 0.118 to 0.122 across the seeds. The lowest interval bound is +0.068.
- **Median per-case R²:** higher by 0.255 to 0.291. The lowest bound is +0.220.
- **Mean absolute error:** lower by 0.125 to 0.139 u*. The bound closest to zero is −0.104.
- **Consistency:** every interval excludes zero, for all cases and for each subset alone. The U-Net has the higher per-case R² in 96.5% to 96.9% of the cases and wins all five folds on every measure.

**What the fixes and the tuning added**, on fold 0's 97 scored cases (`outputs/fold0_versions.csv`):

| Fold 0, 97 cases | Pooled R² | Median per-case R² | Mean absolute error (u*) |
| --- | --- | --- | --- |
| First version of the recipe, before the audit fixes | 0.777 | 0.552 | 0.478 |
| Final code, default settings | 0.787 | 0.572 | 0.466 |
| Final code, tuned settings, seeds 0 / 1 / 2 | 0.791 / 0.789 / 0.802 | 0.590 / 0.567 / 0.633 | 0.457 / 0.465 / 0.442 |
| S07 pixel model | 0.658 | 0.280 | 0.605 |

- **The fixes helped most on the idealized layouts.** Their median per-case R² rose from 0.467 to 0.545.
- **The tuning added a little more, with a network about half the size.** On the shared inner-validation groups it scored 0.578 to 0.608 RMSE, against 0.608 for the defaults.
- **Fold 0 is an easier fold than average.** S07 scores 0.658 pooled here against 0.591 over all folds.

**Leakage: a random split of the same cases** (`configs/unet_tuned_random.yaml`, one seed):
- **Most cases do better.** The random split, which lets a test case's city or layout family into training, beats the mean of the grouped seeds in 312 of 482 cases. Its median per-case R² is 0.593, against 0.560 to 0.595.
- **The pooled R² is lower (0.683), because of one case: AU-Preston.** It is the largest Melbourne case (750,911 open cells), and its mean speed is 5.49 u*, twice that of any other Melbourne case.
  - The grouped models, which never see Melbourne when they predict it, get its mean within 0.06 to 0.25 u*.
  - The random-split model, trained on most of the other, much slower Melbourne cases, is 3.4 u* too low.
- **Without AU-Preston,** the random split scores 0.723 pooled, against 0.692 to 0.695 for the grouped seeds.
- **So leakage flatters this model by about 0.03 pooled R²,** and on one case it shows the opposite risk: a model that has seen a city expects the next case from it to look like the rest.

## Running it

```bash
uv run python pipeline/S11_unet_baseline/overfit_check.py
uv run python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet_tuned.yaml --seed 0            # all five folds
uv run python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet_tuned.yaml --seed 0 --repredict  # new predictions from saved weights
uv run python pipeline/S11_unet_baseline/s11_results.py
```

`URBANWIND_DEVICE=cpu` forces the CPU. Each `--folds k` call trains one fold; the run is scored once all five folds exist.

**On Kaggle or Colab.**
1. Attach the private dataset `mahirm01/urbantales-dl-cache` on Kaggle (ask for access). On Colab, download it with the Kaggle API.
2. Clone the repository, link the cache to `data/processed/dl`, and run with the platform's own PyTorch:

```bash
git clone https://github.com/mahir-m01/Urban-Microclimate-Modelling.git repo && cd repo
mkdir -p data/processed && ln -s "$(dirname "$(find /kaggle/input -name UA25_d45.npz | head -1)")" data/processed/dl
PYTHONPATH=src:. python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet.yaml --seed 0 --folds 0
```

The cache stores the native-grid truth, so the evaluator files can be written without the raw simulation files. `kaggle_runner.py` runs a queue of trainings on all GPUs of a Kaggle machine and keeps only the small results.

**Tuning.** `tune_optuna.py` searches the learning rate, weight decay, dropout, width, depth, patch context and loss on fold 0's inner-validation groups:
- **Trials:** 25 epochs each, with median pruning, and the first trial is the default setting.
- **Workers:** several can share one study, one per GPU.
- **Deadline:** a trial cut short is recorded as pruned and is never ranked.
- **Optimism:** the inner-validation cases it tunes on are scored in folds 1 to 4. The tuned five-fold result is therefore slightly optimistic, and the frozen test set gives the clean number.

**The study.** Run on Kaggle with two T4 workers for 4.9 hours. It ran 33 trials: 17 complete and 16 pruned. All trials are in `outputs/optuna/trials.csv`, and the best trial is in `outputs/optuna/best.json`.

| Trial | Base width, depth | Learning rate | Weight decay | Dropout | Context, loss | Score (balanced RMSE, u*) |
| --- | --- | --- | --- | --- | --- | --- |
| 15, the best | 24, 4 | 3.64e-4 | 6.5e-5 | 0.079 | 256/16, L1 | 0.691 |
| 0, the defaults | 32, 4 | 1.0e-3 | 1.0e-4 | 0.10 | 256/16, L1 | 0.726 |

- **What the study found.**
  - The eight best trials all use base width 24, 256/16 context and the L1 loss, and seven of them use depth 4.
  - They score between 0.691 and 0.697. Their learning rates span 3.1e-4 to 5.8e-4, and their dropout spans 0 to 0.08.
  - The score is flat across this range, so nothing suggests a better learning rate below the searched range (which starts at 3e-4).
- **Final settings.** `configs/unet_tuned.yaml` takes trial 15, rounded to two significant figures, and trains for the full 100 epochs. The network has 4,860,313 parameters and takes about 38 s per epoch on a T4.
- **Leakage comparison.** `configs/unet_tuned_random.yaml` uses the same settings on a random five-fold split.
- **Caveat.** Only three trials paired a wider network with a learning rate in the chosen range:
  - width 32: pruned after 8 epochs, while its score was still within the range of the width-24 trials at the same epoch;
  - width 48: pruned after 7 epochs;
  - width 48 with depth 5 and the MSE loss: completed, with a score of 0.751.

  So the study does not show that width 24 beats width 32 at these learning rates. It shows only that the chosen setting is the best it found, and it is the cheaper one.

## Interfaces for later stages

| Call | Use |
| --- | --- |
| `build_model(name, in_ch, **kwargs)`, `@register(name)` | S12a adds architectures |
| `LOSSES[name](pred, y, mask)` | S12b adds losses with this signature (float32, mean over valid cells, 0 when none) |
| `python -m urbanwind.dl.train --config <yaml> --seed <n>` | every experiment; configs only set what differs from `train.DEFAULTS` |
| `predict_arrays(model, with_inputs(build_arrays(load_case(case))))` | final scoring of test cases without putting them in the cache |
| `model.train()` before `predict_native` | S14 keeps dropout on for MC dropout |

Tests that import PyTorch go in `src/urbanwind/dl/tests/` (on macOS they run in a process of their own; see the root `conftest.py`).

## Limitations

- The network sees the building map and the wind angle only. Anything about a case that the raster does not show, such as the height of the simulated domain, is out of its reach, and it has no case-level descriptors. S13 adds a case-level model.
- Predictions are made on the 1 m grid; the cost on the 0.5 m cases is the ceiling in the table above.
- The validation cases of each fold are not used for training, unless the run sets `inner_val_fraction: 0`.
- **The schedule length was not varied.** Every fold ran the full 100 epochs, and 17 of the 20 final trainings kept their last epoch. The error flattens as the learning rate decays to its floor, but a longer schedule was not tried.
- **The random split was trained with one seed only.** Its gap to the grouped seeds rests on that one run.
- **The trained weights are not in the repository.** The 20 final trainings ran on Kaggle (two T4 GPUs per session, about 65 to 72 minutes per fold). Their predictions were rebuilt locally with `--repredict` and reproduce the scores computed on Kaggle exactly; ask for access to the weights.
