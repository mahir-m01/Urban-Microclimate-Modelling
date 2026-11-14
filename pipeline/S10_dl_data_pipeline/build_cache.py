"""
S10: build the deep-learning cache for the development cases and check it.

Each case is read through urbanwind.data.load_case, brought to the 1 m grid, turned into the
canonical wind frame and saved with its periodic wall distance (urbanwind.dl.cache). The script
then checks every cached case against the loader, draws 1,000 training batches the way S11 will,
and saves QC figures.

Outputs
- data/processed/dl/<case>.npz            the cache (not tracked)
- outputs/cache_summary.csv               one row per case
- outputs/checks.json                     every check below, with its numbers
- outputs/figures/canonical_frame.png     three cases in the raw files, the canonical frame and mirrored
- outputs/figures/wind_angles.png         wind directions before and after the turn
- outputs/figures/sample_patches.png      training patches as the network sees them

Usage: uv run python pipeline/S10_dl_data_pipeline/build_cache.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from urbanwind.data import load_case
from urbanwind.dl.cache import (CACHE_DIR, CHANNELS, H_SCALE, build_case, dev_cases, from_canonical, load_cached,
                                resample_to_1m, write_case)
from urbanwind.dl.datasets import BORDER, PATCH, PatchDataset
from urbanwind.splits import load_splits

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "pipeline" / "S10_dl_data_pipeline" / "outputs"
FIG = OUT / "figures"
CASE_TARGETS = REPO / "pipeline" / "S02_preprocessing" / "outputs" / "case_targets.csv"

N_BATCHES = 1000
BATCH = 16
# a 0.5 m idealized case at 45 degrees (resampled, not turned), a 90 degree city (one quarter turn)
# and a 60 degree city (transposed); CN-BE-V1_d150 is labelled 150 but its forcing points at 60
EXAMPLES = ["UA25_d45", "AU-Ade-V1_d90", "CN-BE-V1_d150"]


def flow_offset(u: np.ndarray, v: np.ndarray, angle_deg: float) -> float:
    """Angle between the domain-mean wind and the forcing direction, in degrees from -180 to 180."""
    mean_dir = np.degrees(np.arctan2(np.nanmean(v), np.nanmean(u)))
    return float((mean_dir - angle_deg + 180.0) % 360.0 - 180.0)


def build_one(case: str) -> dict:
    """Build and save one case, then read it again through the loader to check the round trip."""
    arrays = build_case(case)
    path = write_case(case, arrays)
    c = load_case(case)
    H, fields = resample_to_1m(c["H"], {"U": c["U"], "u": c["u"], "v": c["v"]}, c["dx"])
    k, t = arrays["k"], arrays["transposed"]
    U_back = from_canonical(arrays["U"], k, t).astype(float)
    # scored native cells whose 1 m block holds a building: predict_arrays fills these from open cells
    in_building_block = (H > 0).repeat(2, axis=0).repeat(2, axis=1) if np.isclose(c["dx"], 0.5) else H > 0
    to64 = {name: arrays[name].astype(float) for name in ("u", "v")}
    return {
        "case": case, "subset": arrays["subset"], "split_group": arrays["split_group"], "fold": arrays["fold"],
        "status": arrays["status"], "dx_native": c["dx"], "ny_native": c["H"].shape[0], "nx_native": c["H"].shape[1],
        "ny_1m": H.shape[0], "nx_1m": H.shape[1], "wind_dir_deg": c["wind_dir_deg"], "k": k, "transposed": t,
        "angle_deg": arrays["angle_deg"], "n_valid_native": int(np.isfinite(c["U"]).sum()),
        "n_valid_1m": int(arrays["valid"].sum()),
        "n_scored_filled": int((np.isfinite(c["U"]) & in_building_block).sum()),
        "U_mean": arrays["U_mean"], "u_tau": arrays["u_tau"],
        "H_round_trip_exact": bool(np.array_equal(from_canonical(arrays["H"], k, t).astype(float), H)),
        "U_nan_pattern_matches": bool(np.array_equal(np.isnan(U_back), np.isnan(fields["U"]))),
        "U_float16_max_error": float(np.nanmax(np.abs(U_back - fields["U"]))),
        "U_native_exact": bool(np.array_equal(arrays["U_native"], c["U"].astype(np.float32), equal_nan=True)),
        # the mean wind should keep the same angle to the forcing after the turn (mirrored if transposed)
        "flow_offset_native_deg": flow_offset(fields["u"], fields["v"], c["wind_dir_deg"]),
        "flow_offset_canonical_deg": flow_offset(to64["u"], to64["v"], arrays["angle_deg"]),
        "file_mb": path.stat().st_size / 1e6,
    }


def build_all(cases: list[str], workers: int) -> pd.DataFrame:
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(build_one, cases, chunksize=4))
    print(f"built {len(rows)} cases in {time.time() - t0:.0f} s")
    return pd.DataFrame(rows).sort_values("case").reset_index(drop=True)


def cache_checks(summary: pd.DataFrame) -> dict:
    """Checks on the whole cache; any failure stops the script."""
    splits = load_splits()
    targets = pd.read_csv(CASE_TARGETS).set_index("case")["U_mean"].reindex(summary["case"]).to_numpy()
    files = {p.name.removesuffix(".npz") for p in CACHE_DIR.glob("*.npz")}
    sign = np.where(summary["transposed"], -1.0, 1.0)
    offset_gap = (summary["flow_offset_canonical_deg"] - sign * summary["flow_offset_native_deg"]).abs()
    transforms = summary.groupby(["k", "transposed"]).size()
    checks = {
        "n_cases": len(summary),
        "n_by_subset": summary["subset"].value_counts().to_dict(),
        "n_resampled_from_0p5m": int((summary["dx_native"] == 0.5).sum()),
        "cache_files_match_dev_cases": files == set(summary["case"]),
        "n_test_groups_in_cache": len(set(summary["split_group"]) & set(splits["test_groups"])),
        "U_mean_max_rel_diff_vs_S02": float(np.abs(summary["U_mean"].to_numpy() / targets - 1.0).max()),
        "H_round_trip_exact_all": bool(summary["H_round_trip_exact"].all()),
        "U_nan_pattern_matches_all": bool(summary["U_nan_pattern_matches"].all()),
        "U_native_exact_all": bool(summary["U_native_exact"].all()),
        "U_float16_max_error": float(summary["U_float16_max_error"].max()),
        "flow_offset_preserved_max_gap_deg": float(offset_gap.max()),
        "flow_offset_native_abs_max_deg": float(summary["flow_offset_native_deg"].abs().max()),
        "transform_counts": {f"k={k}, transposed={t}": int(n) for (k, t), n in transforms.items()},
        "cache_size_mb": round(float(summary["file_mb"].sum()), 1),
        "n_cells_1m": int((summary["ny_1m"] * summary["nx_1m"]).sum()),
    }
    assert checks["n_cases"] == 482 and checks["cache_files_match_dev_cases"], "cache and case list differ"
    assert checks["n_test_groups_in_cache"] == 0, "a test group reached the cache"
    # S02 stores 6 significant digits; a NaN (a case missing from S02) also fails, since NaN < x is False
    assert checks["U_mean_max_rel_diff_vs_S02"] < 1e-5, "case means differ from S02"
    assert checks["H_round_trip_exact_all"] and checks["U_nan_pattern_matches_all"], "round trip failed"
    assert checks["U_native_exact_all"], "the stored native truth differs from the loader"
    assert checks["flow_offset_preserved_max_gap_deg"] < 0.5, "the wind components were not turned with the field"
    return checks


def loader_check(ds: PatchDataset) -> dict:
    """Draw N_BATCHES batches exactly as S11 will, and check every one."""
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=True)
    fractions, n_empty, done, t0 = [], 0, 0, time.time()
    while done < N_BATCHES:
        for x, y, mask in loader:
            assert x.shape == (BATCH, 5, PATCH, PATCH) and y.shape == mask.shape == (BATCH, 1, PATCH, PATCH)
            assert torch.isfinite(x).all() and torch.isfinite(y).all() and (y[mask] > 0).all()
            fractions.append(mask.float().mean().item())
            n_empty += int((mask.sum(dim=(1, 2, 3)) == 0).sum())
            done += 1
            if done == N_BATCHES:
                break
        ds.epoch += 1
    ds.epoch = 0
    return {"n_batches": done, "batch_size": BATCH, "patch": PATCH, "border": BORDER, "patches_per_epoch": len(ds),
            "seconds_per_batch": round((time.time() - t0) / done, 4),
            "loss_cells_fraction_mean": round(float(np.mean(fractions)), 4),
            "loss_cells_fraction_min": round(float(np.min(fractions)), 4), "patches_without_loss_cells": n_empty}


def wind_cmap(name: str = "viridis") -> matplotlib.colors.Colormap:
    """The colour map with NaN (buildings, cells left out) drawn grey; a hidden axis would show them white."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("0.6")
    return cmap


def show(ax, field: np.ndarray, angle_deg: float, title: str, vmax: float) -> matplotlib.image.AxesImage:
    im = ax.imshow(field.astype(float), origin="lower", cmap=wind_cmap(), vmin=0, vmax=vmax)
    ny, nx = field.shape
    a = np.deg2rad(angle_deg)
    length = 0.3 * min(ny, nx)
    arrow = ax.annotate("", xy=(nx / 2 + length * np.cos(a), ny / 2 + length * np.sin(a)), xytext=(nx / 2, ny / 2),
                        arrowprops={"arrowstyle": "-|>,head_width=0.5,head_length=0.8", "color": "#d62728", "lw": 3})
    arrow.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="white"), patheffects.Normal()])
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def degrees(a: float) -> str:
    return f"{a + 0.0:g}°"  # + 0.0 turns -0.0 into 0.0


def plot_canonical_frame() -> None:
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(13, 4.0 * len(EXAMPLES)), layout="constrained")
    for row, case in zip(axes, EXAMPLES):
        c, cached = load_case(case), load_cached(case)
        vmax = float(np.nanpercentile(c["U"], 99))
        a = cached["angle_deg"]
        show(row[0], c["U"], c["wind_dir_deg"], f"{case}: raw files, {c['dx']:g} m, wind {degrees(c['wind_dir_deg'])}", vmax)
        show(row[1], cached["U"], a, f"canonical, 1 m (k={cached['k']}, transposed={cached['transposed']}), {degrees(a)}", vmax)
        im = show(row[2], cached["U"][::-1], -a, f"mirrored, {degrees(-a)}", vmax)
        fig.colorbar(im, ax=row.tolist(), shrink=0.85, label="U/u*")
    fig.suptitle("Pedestrian-level wind before and after the canonical turn (red arrow = forcing direction, grey = buildings)",
                 fontsize=12)
    fig.savefig(FIG / "canonical_frame.png", dpi=110)
    plt.close(fig)


def plot_wind_angles(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, col, title in zip(axes, ("wind_dir_deg", "angle_deg"), ("raw files", "canonical frame")):
        counts = summary[col].round(1).value_counts().sort_index()
        pos = np.arange(len(counts))
        ax.bar(pos, counts.values, color="#4c72b0")
        ax.set_xticks(pos, [f"{a:g}" for a in counts.index])
        for i, n in zip(pos, counts.values):
            ax.text(i, n, str(n), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Wind direction, {title}")
        ax.set_xlabel("degrees from +columns")
        ax.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("development cases")
    fig.tight_layout()
    fig.savefig(FIG / "wind_angles.png", dpi=150)
    plt.close(fig)


def plot_sample_patches(ds: PatchDataset) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
    for i in range(6):
        x, y, mask = ds[i]
        target = np.where(mask[0].numpy(), y[0].numpy(), np.nan)
        cos, sin = float(x[CHANNELS.index("cos_angle"), 0, 0]), float(x[CHANNELS.index("sin_angle"), 0, 0])
        axes[0, i].imshow(x[CHANNELS.index("height")].numpy() * H_SCALE, origin="lower", cmap="Greys")
        axes[0, i].set_title(f"building height, wind {degrees(round(np.degrees(np.arctan2(sin, cos)), 1))}", fontsize=9)
        axes[1, i].imshow(target, origin="lower", cmap=wind_cmap(), vmin=0, vmax=np.nanpercentile(target, 99))
        axes[1, i].set_title("U/u* on loss cells", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{PATCH}×{PATCH} training patches (grey = buildings and the {BORDER}-cell border left out of the loss)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "sample_patches.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cases = dev_cases().index.tolist()
    summary = build_all(cases, args.workers)
    build_s = time.time() - start
    summary.to_csv(OUT / "cache_summary.csv", index=False, float_format="%.6g")
    checks = cache_checks(summary)
    t0 = time.time()
    ds = PatchDataset(cases, seed=0)
    checks["loader"] = {"dataset_load_s": round(time.time() - t0, 1)} | loader_check(ds)
    plot_canonical_frame()
    plot_wind_angles(summary)
    plot_sample_patches(ds)
    checks["timing"] = {"workers": args.workers, "build_s": round(build_s, 1), "script_s": round(time.time() - start, 1)}
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
