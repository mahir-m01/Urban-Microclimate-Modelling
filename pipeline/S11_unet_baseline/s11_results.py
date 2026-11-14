"""
S11 results: score every finished U-Net run against S07 and draw the figures.

For each run (one training seed, five folds) it reads the out-of-fold fields in results/preds,
computes the same per-case and pooled metrics as S07, adds the mean absolute error in m/s (the
U/u* error times each case's u*), and compares with the S07 pixel model case by case. A run's split
(grouped, or random for the leakage comparison) comes from the config whose settings hash its folds
recorded. Only grouped runs are compared with S07, which was scored on the same grouped folds.

Outputs
- outputs/per_case_metrics.csv   one row per run and case
- outputs/summary.csv            one row per run and subset (also without the flagged cases), with S07
- outputs/vs_s07.csv             each grouped run minus S07, with 95% group-bootstrap intervals, cases and folds won
- outputs/fold0_versions.csv     fold 0's scored cases only: the first recipe, the fixed code with the default
                                 settings (both fold 0 only), each tuned run and S07
- outputs/run_settings.json      the settings, prediction method and code each run's five folds share
- outputs/figures/learning_curves.png, error_maps_realistic.png, per_case_vs_s07.png

Usage: uv run python pipeline/S11_unet_baseline/s11_results.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbanwind.dl.cache import dev_cases
from urbanwind.dl.train import config_hash, fold_table, load_config
from urbanwind.metrics import case_metrics

REPO = pathlib.Path(__file__).resolve().parents[2]
STAGE = REPO / "pipeline" / "S11_unet_baseline"
OUT, FIG, RUNS = STAGE / "outputs", STAGE / "outputs" / "figures", STAGE / "outputs" / "runs"
PREDS = REPO / "results" / "preds"
S07 = REPO / "pipeline" / "S07_pixel_model" / "outputs"
SUBSETS = ("all", "idealized", "realistic")
BASELINES = ("s07_pixel_gbm", "s07_level_oracle")  # scored from their saved fields, so m/s errors exist too
FOLD0_ONLY = ("s11_unet_v1_seed0", "s11_unet_seed0")  # fold 0 only: the first recipe, and the fixed code with defaults


def finished_runs() -> list[str]:
    """Runs whose five folds are done and whose predictions cover every development case."""
    n_cases = len(dev_cases())
    return sorted(d.name for d in RUNS.iterdir() if d.is_dir() and len(list(d.glob("fold*.json"))) == 5
                  and len(list((PREDS / d.name).glob("*.npz"))) == n_cases)


def run_split(run: str) -> str:
    """'grouped' or 'random': the split of the config in configs/ whose settings hash the run recorded."""
    recorded = json.loads((RUNS / run / "fold0.json").read_text())["config_hash"]
    for path in sorted((STAGE / "configs").glob("*.yaml")):
        cfg = load_config(path)
        if config_hash(cfg) == recorded:
            return cfg["split"]
    raise ValueError(f"{run}: no config in configs/ has its settings hash {recorded}")


def score_run(run: str, dev: pd.DataFrame) -> pd.DataFrame:
    """Per-case metrics; ``dev`` gives each case's outer fold under the run's own split."""
    rows = []
    for path in sorted((PREDS / run).glob("*.npz")):
        with np.load(path) as f:
            y_true, y_pred, mask = f["y_true"], f["y_pred"], f["mask"]
        case = path.stem
        m = case_metrics(y_true=y_true, y_pred=y_pred, mask=mask, case=case)
        t, p = y_true[~mask].astype(float), y_pred[~mask].astype(float)
        info = dev.loc[case]
        rows.append(m | {"run": run, "subset": info["subset"], "split_group": info["split_group"],
                         "outer_fold": int(info["fold"]),
                         "flagged": info["status"] == "flag", "u_tau": info["u_tau"],
                         "mae_ms": m["mae"] * info["u_tau"],
                         # sums for the pooled R² over all cells of all cases
                         "sum_t": t.sum(), "sum_t2": (t * t).sum(), "sse": ((p - t) ** 2).sum()})
    return pd.DataFrame(rows)


def pooled_r2(rows: pd.DataFrame) -> float:
    n, s, ss, sse = rows["n"].sum(), rows["sum_t"].sum(), rows["sum_t2"].sum(), rows["sse"].sum()
    return float(1 - sse / (ss - s * s / n))


def summarise(per_case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, by_run in per_case.groupby("run"):
        for subset in (*SUBSETS, "all, without flagged"):
            sub = {"all": by_run, "all, without flagged": by_run[~by_run["flagged"]]}.get(
                subset, by_run[by_run["subset"] == subset])
            rows.append({"variant": run, "split": by_run["split"].iloc[0], "subset": subset, "n_cases": len(sub),
                         "pooled_r2": pooled_r2(sub),
                         "median_within_case_r2": sub["r2"].median(), "mean_mae": sub["mae"].mean(),
                         "mean_mae_ms": sub["mae_ms"].mean(), "median_nrmse": sub["nrmse"].median(),
                         "median_corr": sub["corr"].median(), "mean_delta_p95": sub["delta_p95"].mean(),
                         "mean_delta_max": sub["delta_max"].mean(), "mean_level_err": sub["level_err"].mean(),
                         "mean_pattern_err": sub["pattern_err"].mean()})
    return pd.DataFrame(rows)


def plot_learning_curves(runs: list[str]) -> None:
    rows = (len(runs) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows), squeeze=False, sharey=True)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    for ax, run in zip(axes.flat, runs):
        for k in range(5):
            hist = pd.read_csv(RUNS / run / f"fold{k}.csv")
            val = hist["val"] if "val" in hist else hist["val_l1"]  # runs before the metric choice logged L1
            line, = ax.plot(hist["epoch"], val, label=f"fold {k} validation")
            ax.plot(hist["epoch"], hist["train_loss"], ls=":", color=line.get_color())
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("error (u*)")
        ax.legend(fontsize=8)
    fig.suptitle("Training loss (dotted) and inner-validation error (solid) per fold")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=130)
    plt.close(fig)


def wind_cmap(name: str) -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("0.6")  # buildings and cells left out
    return cmap


def plot_error_maps(run: str) -> None:
    cases = json.loads((S07 / "run_info.json").read_text())["error_map_cases"]  # the same cases as S07
    fig, axes = plt.subplots(len(cases), 4, figsize=(18, 4.4 * len(cases)), layout="constrained")
    for row, case in zip(axes, cases):
        unet, gbm = np.load(PREDS / run / f"{case}.npz"), np.load(PREDS / "s07_pixel_gbm" / f"{case}.npz")
        true = unet["y_true"]
        vmax = float(np.nanpercentile(true, 99))
        panels = [(true, f"{case}: LES (U/u*)", "viridis", 0, vmax), (unet["y_pred"], "U-Net", "viridis", 0, vmax),
                  (unet["y_pred"] - true, "U-Net error", "RdBu_r", -2, 2),
                  (gbm["y_pred"] - true, "S07 pixel GBM error", "RdBu_r", -2, 2)]
        for ax, (field, title, cmap, lo, hi) in zip(row, panels):
            im = ax.imshow(field, origin="lower", cmap=wind_cmap(cmap), vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Held-out realistic cases chosen by S07 (worst, median, best); {run}; grey = buildings")
    fig.savefig(FIG / "error_maps_realistic.png", dpi=100)
    plt.close(fig)


def plot_per_case_vs_s07(per_case: pd.DataFrame, run: str) -> None:
    s07 = pd.read_csv(S07 / "per_case_metrics.csv").query("variant == 'pixel_gbm'").set_index("case")["r2"]
    unet = per_case[per_case["run"] == run].set_index("case")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, subset in zip(axes, ("idealized", "realistic")):
        sub = unet[unet["subset"] == subset]
        better = (sub["r2"] > s07.reindex(sub.index)).mean()
        outside = int(((sub["r2"] < -1) | (s07.reindex(sub.index) < -1)).sum())
        ax.scatter(s07.reindex(sub.index), sub["r2"], s=10, alpha=0.6, color="#4c72b0")
        ax.plot([-1, 1], [-1, 1], color="0.4", lw=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{subset}: U-Net better in {better:.0%} of {len(sub)} cases"
                     + (f"\n({outside} below -1, not shown)" if outside else ""), fontsize=10)
        ax.set_xlabel("S07 pixel GBM, per-case R²")
    axes[0].set_ylabel(f"{run}, per-case R²")
    fig.tight_layout()
    fig.savefig(FIG / "per_case_vs_s07.png", dpi=130)
    plt.close(fig)


def check_run(run: str) -> dict:
    """All five folds of a run must share settings, prediction method and code; returns what they share."""
    folds = [json.loads((RUNS / run / f"fold{k}.json").read_text()) for k in range(5)]
    keys = ("config_hash", "native_upsample", "infer_window", "infer_border", "infer_margin", "mirror_tta",
            "train_code", "predict_code")
    shared = {}
    for key in keys:
        values = {json.dumps(f.get(key), sort_keys=True) for f in folds}
        if len(values) > 1:
            raise ValueError(f"{run}: the folds differ in {key}: {sorted(values)}")
        shared[key] = folds[0].get(key)
    return shared


def bootstrap_vs_s07(per_case: pd.DataFrame, run: str, n_boot: int = 2000, ref: str = "s07_pixel_gbm") -> pd.DataFrame:
    """The run against a reference (S07 by default) case by case, with 95% intervals from resampling whole split groups."""
    both = per_case[per_case["run"].isin([run, ref])].pivot_table(
        index=["case", "split_group", "subset", "outer_fold"], columns="run",
        values=["r2", "mae", "n", "sum_t", "sum_t2", "sse"]).reset_index()
    rng = np.random.default_rng(0)
    rows = []
    for subset in SUBSETS:
        sub = both if subset == "all" else both[both["subset"] == subset]
        groups = sub["split_group"].unique()
        rows_of = {g: np.flatnonzero(sub["split_group"].to_numpy() == g) for g in groups}
        stats = {name: [] for name in ("pooled_r2", "median_case_r2", "mean_mae")}
        for _ in range(n_boot):
            pick = sub.iloc[np.concatenate([rows_of[g] for g in rng.choice(groups, len(groups))])]
            for name, value in _differences(pick, run, ref).items():
                stats[name].append(value)
        point = _differences(sub, run, ref)
        for name in stats:
            lo, hi = np.percentile(stats[name], [2.5, 97.5])
            rows.append({"run": run, "subset": subset, "metric": name, "difference": point[name], "ci_low": lo,
                         "ci_high": hi, "share_of_cases_better": _share_better(sub, run, name, ref),
                         "folds_better": _folds_better(sub, run, name, ref)})
    return pd.DataFrame(rows)


def _pooled(sub: pd.DataFrame, run: str) -> float:
    n, s, ss, sse = (sub[(k, run)].sum() for k in ("n", "sum_t", "sum_t2", "sse"))
    return float(1 - sse / (ss - s * s / n))


def _differences(sub: pd.DataFrame, run: str, ref: str = "s07_pixel_gbm") -> dict:
    """Run minus the reference: positive is better for R², negative is better for MAE."""
    return {"pooled_r2": _pooled(sub, run) - _pooled(sub, ref),
            "median_case_r2": float(sub[("r2", run)].median() - sub[("r2", ref)].median()),
            "mean_mae": float(sub[("mae", run)].mean() - sub[("mae", ref)].mean())}


def _share_better(sub: pd.DataFrame, run: str, name: str, ref: str = "s07_pixel_gbm") -> float:
    if name == "mean_mae":
        return float((sub[("mae", run)] < sub[("mae", ref)]).mean())
    return float((sub[("r2", run)] > sub[("r2", ref)]).mean())


def _folds_better(sub: pd.DataFrame, run: str, name: str, ref: str = "s07_pixel_gbm") -> int:
    better = 0
    for _, fold in sub.groupby("outer_fold"):
        diff = _differences(fold, run, ref)[name]
        better += int(diff < 0 if name == "mean_mae" else diff > 0)
    return better


def fold0_versions(per_case: pd.DataFrame, grouped: list[str], dev: pd.DataFrame) -> pd.DataFrame:
    """Fold 0's scored cases: what the fixes (first recipe against fixed defaults) and the tuning each added."""
    extra = [run for run in FOLD0_ONLY if (RUNS / run / "fold0.json").exists() and (PREDS / run).is_dir()]
    scored = pd.concat([per_case[per_case["run"].isin([*extra, *grouped, "s07_pixel_gbm"])],
                        *(score_run(run, dev) for run in extra)], ignore_index=True)
    scored = scored[scored["case"].isin(dev.index[dev["fold"] == 0])]
    rows = []
    for run in [*extra, *grouped, "s07_pixel_gbm"]:
        sub = scored[scored["run"] == run]
        record = RUNS / run / "fold0.json"
        info = json.loads(record.read_text()) if record.exists() else {}
        rows.append({"run": run, "n_cases": len(sub), "pooled_r2": pooled_r2(sub), "median_case_r2": sub["r2"].median(),
                     "median_case_r2_idealized": sub.loc[sub["subset"] == "idealized", "r2"].median(),
                     "median_case_r2_realistic": sub.loc[sub["subset"] == "realistic", "r2"].median(),
                     "mean_mae": sub["mae"].mean(),
                     # the inner-validation RMSE the run selected on; the first recipe selected on L1 instead
                     "inner_val_rmse": info["best_val"] if info.get("val_metric") == "rmse" else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    runs = finished_runs()
    if not runs:
        raise SystemExit("no finished S11 run yet")
    split = {run: run_split(run) for run in runs} | {run: "grouped" for run in BASELINES}
    grouped = [run for run in runs if split[run] == "grouped"]
    if not grouped:
        raise SystemExit("no finished grouped S11 run yet")
    shared = {run: check_run(run) | {"split": split[run]} for run in runs}
    (OUT / "run_settings.json").write_text(json.dumps(shared, indent=2, default=str) + "\n")
    folds = {name: fold_table(name) for name in set(split.values())}
    per_case = pd.concat([score_run(run, folds[split[run]]).assign(split=split[run]) for run in [*runs, *BASELINES]],
                         ignore_index=True)
    per_case.drop(columns=["sum_t", "sum_t2", "sse"]).to_csv(OUT / "per_case_metrics.csv", index=False,
                                                              float_format="%.5g")
    summary = summarise(per_case)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False))
    versions = fold0_versions(per_case, grouped, folds["grouped"])
    versions.to_csv(OUT / "fold0_versions.csv", index=False, float_format="%.4f")
    print(versions.to_string(index=False))
    paired = pd.concat([bootstrap_vs_s07(per_case, run) for run in grouped], ignore_index=True)
    paired.to_csv(OUT / "vs_s07.csv", index=False, float_format="%.4f")
    print(paired.to_string(index=False))
    plot_learning_curves(runs)
    plot_error_maps(grouped[0])
    plot_per_case_vs_s07(per_case, grouped[0])


if __name__ == "__main__":
    main()
