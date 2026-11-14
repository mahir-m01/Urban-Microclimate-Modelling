# S10 Deep-learning data pipeline

Turns every development case into network-ready arrays on one 1 m grid and in one canonical wind frame, with a tested patch sampler and a helper that predicts a whole case back on its native grid. Every Phase 2 model reads its data through this stage.

| | |
| --- | --- |
| Inputs | raw case files through `urbanwind.data.load_case`; `data/splits.json` |
| Outputs | `data/processed/dl/<case>.npz` (not tracked, 873 MB), `outputs/cache_summary.csv`, `outputs/checks.json`, figures |
| Code | `src/urbanwind/dl/cache.py`, `src/urbanwind/dl/datasets.py`, `build_cache.py`; tests in `src/urbanwind/dl/tests/test_dl_data.py` |

## Method

**Cases.** The 482 development cases (190 idealized, 292 realistic). The frozen test groups and the three excluded cases never enter the cache: `build_case` refuses them before opening any file. The six flagged cases are kept, with their `status`, as in Phase 1.

**1 m grid.** 194 cases are stored at 0.5 m: all 190 idealized ones and four realistic ones (Val-AU-Mel-U2, U9, U11 and U12). Each 2×2 block becomes one 1 m cell:
- the height is the tallest of the four sub-cells, so thin walls survive;
- U, u, v and TKE are the mean of the four;
- the cell is valid only if all four sub-cells are valid.

**Canonical frame.** Each case is turned by a multiple of 90° and, if needed, transposed, so the wind blows within 0° to 45° of +columns. The components u and v are turned with the field. No run has Coriolis forcing (`omega = 0` in every p3d file), so the turned case solves the same equations exactly. The stored transform `(k, transposed)` is undone exactly at prediction time.

| Transform | Cases | Raw directions |
| --- | --- | --- |
| none | 396 | 0° to 45° |
| one quarter turn (`k=1`) | 74 | 90° |
| three quarter turns (`k=3`) | 7 | 270°, 285°, 292.5°, 300°, 315° |
| three quarter turns and transpose | 3 | 330°, 337.5°, 345° |
| transpose | 2 | 60°, 75° |

After the turn, 321 cases blow at 0°, 21 at 15°, 3 at 22.5°, 21 at 30° and 116 at 45°.

**Inputs and target.**

| Channel | Content |
| --- | --- |
| 0 | building height / 16 m |
| 1 | building mask (1 on buildings) |
| 2 | periodic distance to the nearest building / 16 m (the S05 transform, which wraps across the domain edges) |
| 3, 4 | cos and sin of the remaining wind angle |

Only the building raster and the wind direction go in. The target is U/u* (each case's own u*), set to 0 outside `valid` so a masked loss never meets NaN. The loss mask is `valid`. Heights, distances and wind fields are stored as float16; the largest rounding error in the targets is 0.0039 u*. Scores always use the native float64 fields from the loader.

**Training patches** (`PatchDataset`).
- 256×256 crops at a random cell, wrapping across the domain edges. Idealized domains narrower than 256 m (the smallest side is 60 m) are tiled periodically.
- Cases are drawn with probability proportional to the square root of their open area, so large cities do not drown out the small idealized layouts.
- Mirror augmentation with probability 0.5: flip the rows and flip the sign of the wind angle. This is exact for the same reason as the turn.
- The network pads each patch circularly, which joins its top edge to its bottom edge although they are not neighbours in the city. The 16 cells along every side are therefore left out of the loss.
- One epoch holds as many patches as it takes to cover the total area of its cases once (1,586 for all 482 cases). A patch depends only on `(seed, epoch, index)`, so runs are reproducible with any number of DataLoader workers.

**Full-domain prediction** (`predict_arrays`, and `predict_native` for cached cases).
1. Wrap the canonical domain periodically by at least 64 cells on every side, rounding each side up to a multiple of 16 (the U-Net pools four times). The seam that circular padding creates then lies outside the city, instead of inside it as with plain padding to a multiple of 16.
2. Predict, give every building cell the prediction of the nearest open cell (see below), crop back, undo the transform, and copy each 1 m value to its four sub-cells for 0.5 m cases.
3. The result lines up cell by cell with `load_case(case)["U"]`, so `urbanwind.evaluate` scores exactly the same cells as S07.

**Why the fill.** In a 0.5 m case, a 1 m block that straddles a wall counts as a building, so the loss never trains the network there, yet the block still holds open 0.5 m sub-cells that the evaluator scores. Such cells exist in 97 of the 194 cases at 0.5 m; in those 97 they are 3.9% of a case's scored cells on average (median 1.3%, at most 13.3% in VA64-CL-28_d00). They lie next to walls, so they take the prediction of the nearest open cell rather than the untrained output. The fill uses the building map only, never the simulation, so it works the same for test cases. In 1 m cases it touches no scored cell.

## Interfaces

| Call | Returns |
| --- | --- |
| `dev_cases()` | the 482 development cases with `subset`, `split_group`, `status` and outer `fold` |
| `load_cached(case)` | `x` (5 × ny × nx float32), `y`, `valid`, `subset`, `split_group`, `fold`, `U_mean`, `angle_deg`, `k`, `transposed`, the stored `H`, `dist`, `U`, `u`, `v`, `TKE` in the canonical frame, and `U_native`, the float32 truth on the native grid and orientation that the evaluator scores against |
| `read_cached(case)` | the stored arrays only, without `x` and `y` |
| `PatchDataset(cases, patch=256, mirror=True, seed=0)` | `(x, y, mask)` tensors of shape `[5, P, P]`, `[1, P, P]` and `[1, P, P]`; set `dataset.epoch` before each epoch, and leave the DataLoader's `persistent_workers` off so workers see it |
| `predict_native(model, case)` | U/u* on the native grid for a cached development case, ready for the evaluator's field format |
| `predict_arrays(model, with_inputs(build_arrays(load_case(case))))` | the same for a case held in memory; final scoring uses this for test cases, which never enter the cache |

`U_mean` is the case mean of U/u* on the native grid; it matches `pipeline/S02_preprocessing/outputs/case_targets.csv` for every case. A cache built by an older version of this stage, or for a different `splits.json`, fails to load with a message to rebuild it.

## Checks (`outputs/checks.json`)

| Check | Result |
| --- | --- |
| Cases in the cache | 482 (190 idealized, 292 realistic); the files on disk match the case list exactly |
| Test groups in the cache | 0 |
| Height field turned back to the raw orientation | identical to the loader's 1 m height for all 482 cases |
| Wind field turned back | the same missing cells as the loader for all 482 cases; largest float16 difference 0.0039 u* |
| Native-grid truth (`U_native`) | identical to the loader's field for all 482 cases, so runs without the raw files (Kaggle, Colab) can still write evaluator files |
| Angle between the mean wind and the forcing, before and after the turn | the same for every case (largest gap 0.002°), mirrored for transposed cases |
| Case means against S02 | agree to the 6 significant digits S02 stores |
| 1,000 batches of 16 patches | all load with finite inputs and targets; 56.8% of patch cells carry loss on average (at least 48.6% in a batch); no patch without loss cells |
| Speed | the cache builds in 20 s on 8 cores and the whole script runs in 52 s; loading all cases into a `PatchDataset` takes 5.0 s and holds about 0.7 GB; drawing one batch takes 0.025 s (recorded run) |

The unit tests check each transform and its inverse for all 8 symmetries of the square, the remaining angle for every development direction, the 2×2 resampling on a hand-made example, the mirror against the inputs of the mirrored case, wrap-around and tiling crops, that patches do not depend on the order of the cases, the fill, and `predict_native` and `predict_arrays` on synthetic and real cases.

## Caveats for later stages

- **The mean pedestrian wind often runs against the forcing.** In 41 development cases (21 idealized, 20 realistic) the mean of (u, v) at 1.75 m points more than 90° away from the forcing direction. This is physical, not a sign error: for 32 of the 33 such cases at 0°, the plane-averaged profile u at 1.75 m is negative as well, and it turns positive higher up. Near the ground the air in the streets circulates back against the wind above. Anything that uses u and v should expect this.
- **The `v` column of the profile CSV files is not a signed mean.** At 1.75 m it is positive in all 482 cases, even at 0°, where the mean v at pedestrian level is close to zero. The profile `u` column does agree with the pedestrian-level mean u (correlation at least 0.998 within each of the six directions that have more than two cases).
- **PyTorch and LightGBM do not mix in one process on macOS.** Each loads its own copy of the OpenMP runtime, and a process that uses both hangs or crashes (seen with PyTorch 2.15.0 and LightGBM 4.7.0). Tests that import PyTorch therefore go in `src/urbanwind/dl/tests/`; on macOS the root `conftest.py` runs that folder in a second pytest process, so `uv run pytest -q` still runs everything. A script that needs both, such as a hybrid of a tree model and a network, should run them in separate processes on macOS. On Linux, including CI, all tests run in one process.
- **Most cases share two angles.** After the turn, 437 of 482 cases blow at exactly 0° or 45°. Only 5 cases need the transpose, so the angle channels carry little variety; the mirror adds the negative angles.

## Figures

- `outputs/figures/canonical_frame.png`: three cases in the raw files, in the canonical frame and mirrored (a 0.5 m idealized case at 45°, a city at 90° and a city at 60°).
- `outputs/figures/wind_angles.png`: wind directions before and after the turn.
- `outputs/figures/sample_patches.png`: six training patches as the network sees them.

## Reproduce

```bash
uv sync                                                        # includes PyTorch (the dl group)
uv run python pipeline/S10_dl_data_pipeline/build_cache.py     # under a minute on 8 cores
uv run pytest -q src/urbanwind/dl/tests
```

On Linux, `uv sync` installs the CPU build of PyTorch. On a machine with an NVIDIA GPU (Kaggle, Colab), keep the platform's CUDA build and install this package with `pip install -e . --no-deps`, adding any missing package with `pip`. PyTorch publishes no wheels for Intel Macs, so there `uv sync` leaves it out and the deep-learning tests skip themselves.
