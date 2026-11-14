"""
cache.py: network-ready arrays for every development case.

Each case goes through the same steps once, and the result is saved to
``data/processed/dl/<case>.npz``:

1. **1 m grid.** The 0.5 m cases are merged into 2×2 blocks, so every case shares one grid.
2. **Canonical wind frame.** The case is rotated by a multiple of 90° and, if needed,
   transposed, so the wind blows within 0° to 45° of +columns. The wind components u, v are
   turned with it. No run has Coriolis forcing (``omega = 0`` in every p3d file), so the
   turned case is an exact solution of the same equations.
3. **Wall distance.** The periodic distance to the nearest building, from the S05 code.

Test-group cases never enter the cache: ``build_case`` refuses them before reading any file.

Public API
----------
dev_cases()                   : the development cases with their outer fold
resample_to_1m(H, fields, dx) : 2×2 block resampling of a 0.5 m case
canonical_transform(angle)    : (k, transposed, remaining angle) for one wind direction
to_canonical, from_canonical  : apply or undo that transform on a field
vector_to_canonical(u, v, ..) : the same for the wind components
make_inputs(H, dist, angle)   : the five input channels
build_arrays(c)               : cached arrays from a ``load_case`` dict
build_case(case), write_case  : build and save one development case
read_cached(case)             : the stored arrays and metadata of one case
with_inputs(arrays)           : add the inputs ``x`` and the target ``y`` to built or stored arrays
load_cached(case)             : ``with_inputs(read_cached(case))``
"""

from __future__ import annotations

import functools
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from urbanwind.data import DATA_DIR, RAW_DIR, case_table, load_case
from urbanwind.features import compute_periodic_distance_transform
from urbanwind.splits import load_splits

CACHE_DIR = DATA_DIR / "processed" / "dl"
CACHE_VERSION = 2  # bump when the cached arrays change, so an old copy fails loudly (2: U_native added)
FINE_DX = 0.5
H_SCALE = 16.0  # metres; brings building height and wall distance to order 1
DIST_SCALE = 16.0
CHANNELS = ("height", "building", "wall_distance", "cos_angle", "sin_angle")
SIN_CHANNEL = CHANNELS.index("sin_angle")
BUILDING_CHANNEL = CHANNELS.index("building")
FIELDS = ("U", "u", "v", "TKE")
SCALARS = ("angle_deg", "k", "transposed", "dx_native", "wind_dir_deg", "U_mean", "u_tau",
           "subset", "split_group", "status", "fold", "version", "splits_hash")


def dev_cases() -> pd.DataFrame:
    """Development cases (frozen test groups and excluded cases left out) with their outer fold."""
    splits = load_splits()
    table = case_table()
    dev = table[(table["status"] != "exclude") & ~table["split_group"].isin(splits["test_groups"])].copy()
    # astype(int) fails on a group without a fold, which is what we want: no silent extra cases
    dev["fold"] = dev["split_group"].map(splits["folds"]).astype(int)
    return dev


def resample_to_1m(H: np.ndarray, fields: dict[str, np.ndarray], dx: float) -> tuple[np.ndarray, dict]:
    """Bring one case to the 1 m grid; 1 m cases pass through unchanged.

    A 1 m cell takes the tallest building of its four sub-cells, so thin walls survive. Its wind
    value is the mean of the four, and because a NaN in any sub-cell makes that mean NaN, a 1 m
    cell is valid only when all four of its sub-cells are.
    """
    if np.isclose(dx, 1.0):
        return H, fields
    if not np.isclose(dx, FINE_DX):
        raise ValueError(f"unexpected grid spacing {dx} m")
    ny, nx = H.shape
    if ny % 2 or nx % 2:
        raise ValueError(f"a 0.5 m grid of shape {H.shape} cannot be split into 2x2 blocks")

    def blocks(a: np.ndarray) -> np.ndarray:
        return a.reshape(ny // 2, 2, nx // 2, 2)

    return blocks(H).max(axis=(1, 3)), {name: blocks(a).mean(axis=(1, 3)) for name, a in fields.items()}


def canonical_transform(angle_deg: float) -> tuple[int, bool, float]:
    """The square symmetry that brings the wind within 0 to 45 degrees of +columns.

    Returns ``(k, transposed, remaining_angle)``: turn the array k times with ``np.rot90``, then
    transpose it if ``transposed``. Each ``np.rot90`` turns the wind direction by -90 degrees and
    the transpose reflects it to 90 minus the angle. Angles run from +columns towards +rows.
    """
    theta = float(angle_deg) % 360.0 % 360.0  # the second % catches -1e-14 % 360 == 360.0
    k = int(theta // 90.0) % 4
    rest = theta - 90.0 * k
    transposed = rest > 45.0
    return k, transposed, (90.0 - rest) if transposed else rest


def to_canonical(a: np.ndarray, k: int, transposed: bool) -> np.ndarray:
    """Apply the transform to the last two axes of ``a`` (a field or a stack of channels)."""
    a = np.rot90(a, k, axes=(-2, -1))
    return np.swapaxes(a, -2, -1) if transposed else a


def from_canonical(a: np.ndarray, k: int, transposed: bool) -> np.ndarray:
    """Undo ``to_canonical`` exactly: the steps in reverse order."""
    if transposed:
        a = np.swapaxes(a, -2, -1)
    return np.rot90(a, -k, axes=(-2, -1))


def vector_to_canonical(u: np.ndarray, v: np.ndarray, k: int, transposed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Move the wind components with the field and turn them too.

    u points along +columns and v along +rows. One ``np.rot90`` maps (u, v) to (v, -u), and the
    transpose swaps the two.
    """
    u, v = to_canonical(u, k, transposed), to_canonical(v, k, transposed)
    for _ in range(k):
        u, v = v, -u
    return (v, u) if transposed else (u, v)


def make_inputs(H: np.ndarray, dist: np.ndarray, angle_deg: float) -> np.ndarray:
    """The five input channels, float32 with shape (5, ny, nx).

    Only the building raster and the wind angle go in; nothing from the simulation output.
    """
    H = H.astype(np.float32)
    angle = np.deg2rad(angle_deg)
    ones = np.ones_like(H)
    return np.stack([H / H_SCALE, (H > 0).astype(np.float32), dist.astype(np.float32) / DIST_SCALE,
                     np.float32(np.cos(angle)) * ones, np.float32(np.sin(angle)) * ones])


def build_arrays(c: dict[str, Any]) -> dict[str, Any]:
    """Cached arrays of one case, from the dict that ``load_case`` returns."""
    H, fields = resample_to_1m(c["H"], {name: c[name] for name in FIELDS}, c["dx"])
    k, transposed, angle = canonical_transform(c["wind_dir_deg"])
    H = to_canonical(H, k, transposed)
    u, v = vector_to_canonical(fields["u"], fields["v"], k, transposed)
    U, TKE = to_canonical(fields["U"], k, transposed), to_canonical(fields["TKE"], k, transposed)
    dist, _ = compute_periodic_distance_transform(H, dx=1.0)
    valid = np.isfinite(U)
    if (valid & (H > 0)).any():
        raise ValueError("a valid wind cell sits on a building")
    # float16 halves the cache; the scores always use the native float64 truth, so this only
    # touches the training targets (the build script reports the largest rounding error)
    small = {name: np.ascontiguousarray(a, dtype=np.float16) for name, a in
             (("H", H), ("dist", dist), ("U", U), ("u", u), ("v", v), ("TKE", TKE))}
    return small | {"valid": np.ascontiguousarray(valid), "angle_deg": angle, "k": k, "transposed": transposed,
                    "native_shape": np.array(c["H"].shape), "dx_native": float(c["dx"]),
                    "wind_dir_deg": float(c["wind_dir_deg"]), "U_mean": float(np.nanmean(c["U"])),
                    # the truth on the native grid and orientation, as the evaluator scores it, so runs on
                    # machines without the raw files (Kaggle, Colab) can still write evaluator files
                    "U_native": np.ascontiguousarray(c["U"], dtype=np.float32)}


@functools.lru_cache(maxsize=1)
def _splits_hash() -> str:
    return load_splits()["hash"]


def build_case(case: str, raw_dir: pathlib.Path = RAW_DIR) -> dict[str, Any]:
    """Load one development case through the shared loader and build its cached arrays."""
    dev = dev_cases()
    if case not in dev.index:
        raise ValueError(f"{case} is not a development case (test group, excluded or unknown)")
    info = dev.loc[case]
    c = load_case(case, raw_dir)
    return build_arrays(c) | {"u_tau": c["u_tau"], "subset": info["subset"], "split_group": info["split_group"],
                              "status": info["status"], "fold": int(info["fold"]),
                              "version": CACHE_VERSION, "splits_hash": _splits_hash()}


def write_case(case: str, arrays: dict[str, Any], cache_dir: pathlib.Path = CACHE_DIR) -> pathlib.Path:
    """Save one case; writing to a temporary name first means an interrupted build leaves no half file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, tmp = cache_dir / f"{case}.npz", cache_dir / f"{case}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def read_cached(case: str, cache_dir: pathlib.Path = CACHE_DIR) -> dict[str, Any]:
    """The stored arrays and metadata of one cached case, after checking the cache is current."""
    with np.load(cache_dir / f"{case}.npz") as f:
        out = {name: f[name] for name in f.files}
    for name in SCALARS:
        out[name] = out[name].item()
    if out["version"] != CACHE_VERSION or out["splits_hash"] != _splits_hash():
        raise ValueError(f"{case}: the cache is out of date; rebuild it with pipeline/S10_dl_data_pipeline/build_cache.py")
    return out


def with_inputs(arrays: dict[str, Any]) -> dict[str, Any]:
    """Add the network inputs and target to the arrays of one case (from the cache or ``build_arrays``).

      x  float32 (5, ny, nx) input channels, see ``CHANNELS``
      y  float32 (ny, nx) U/u_tau, set to 0 where ``valid`` is False so a masked loss never sees NaN
    """
    x = make_inputs(arrays["H"], arrays["dist"], arrays["angle_deg"])
    return arrays | {"x": x, "y": np.nan_to_num(arrays["U"].astype(np.float32), nan=0.0)}


def load_cached(case: str, cache_dir: pathlib.Path = CACHE_DIR) -> dict[str, Any]:
    """One cached case, ready for a network: every stored array plus ``x`` and ``y``."""
    return with_inputs(read_cached(case, cache_dir))
