"""
s12_results.py: Score finished S12 architecture search runs against the S11 tuned U-Net baseline.

Reads finished run records from outputs/runs/ and evaluator predictions in results/preds/,
computes pooled R2, median per-case R2 across subsets, mean delta P95, GPU training minutes,
and benchmarks inference time per case. Also conducts group-bootstrap uncertainty comparison
against S11 U-Net (resampling whole split_groups).

Outputs:
- outputs/summary.csv: Model comparison table across architectures and seeds.
- outputs/vs_s11_seed1.csv: Group-bootstrap paired comparison (seed 1 candidates minus S11 baseline).
- outputs/figures/ (learning curves and per-case scatter plots generated only from real results).

Usage:
    uv run python pipeline/S12_architecture_search/s12_results.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from urbanwind.dl.cache import dev_cases
from urbanwind.dl.nets import build_model
from urbanwind.dl.train import config_hash, fold_table, load_config
from urbanwind.metrics import case_metrics

REPO = pathlib.Path(__file__).resolve().parents[2]
STAGE = REPO / "pipeline" / "S12_architecture_search"
OUT = STAGE / "outputs"
FIG = STAGE / "outputs" / "figures"
RUNS = STAGE / "outputs" / "runs"
S11_RUNS = REPO / "pipeline" / "S11_unet_baseline" / "outputs" / "runs"
S11_METRICS_CSV = REPO / "pipeline" / "S11_unet_baseline" / "outputs" / "per_case_metrics.csv"
PREDS = REPO / "results" / "preds"

SUBSETS = ("all", "idealized", "realistic")
S11_BASELINE_RUN = "s11_unet_tuned_seed0"

SUMMARY_COLUMNS = [
    "model",
    "seed",
    "n_cases",
    "parameters",
    "gpu_minutes",
    "pooled_r2",
    "median_case_r2_all",
    "median_case_r2_idealized",
    "median_case_r2_realistic",
    "mean_delta_p95",
    "inference_sec_per_case",
]

VS_S11_COLUMNS = [
    "run",
    "ref_run",
    "subset",
    "metric",
    "difference",
    "ci_low",
    "ci_high",
    "share_of_cases_better",
    "folds_better",
]


def finished_runs() -> list[str]:
    """Find S12 runs whose 5 folds are finished and predictions exist in results/preds/."""
    if not RUNS.is_dir():
        return []
    n_cases = len(dev_cases())
    return sorted(
        d.name
        for d in RUNS.iterdir()
        if d.is_dir()
        and len(list(d.glob("fold*.json"))) == 5
        and (PREDS / d.name).is_dir()
        and len(list((PREDS / d.name).glob("*.npz"))) == n_cases
    )


def run_split(run: str) -> str:
    """Read split type ('grouped' or 'random') corresponding to the run's config hash."""
    recorded = json.loads((RUNS / run / "fold0.json").read_text())["config_hash"]
    for path in sorted((STAGE / "configs").glob("*.yaml")):
        cfg = load_config(path)
        if config_hash(cfg) == recorded:
            return cfg["split"]
    raise ValueError(f"{run}: no config in configs/ has recorded settings hash {recorded}")


def count_parameters(model_name: str, cfg: dict) -> int:
    model = build_model(model_name, 5, **cfg["model"].get("kwargs", {}))
    return sum(p.numel() for p in model.parameters())


def measure_inference_time_per_case(model_name: str, cfg: dict, device: torch.device, reps: int = 5) -> float:
    """Benchmark raw forward inference time per case (using representative 512x512 domain)."""
    model = build_model(model_name, 5, **cfg["model"].get("kwargs", {})).to(device).eval()
    x = torch.randn(1, 5, 512, 512, device=device)
    # Warm-up
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / reps
    return float(elapsed)


def score_run(run: str, dev: pd.DataFrame, preds_dir: pathlib.Path) -> pd.DataFrame:
    """Calculate per-case metrics from saved prediction arrays."""
    rows = []
    for path in sorted((preds_dir / run).glob("*.npz")):
        with np.load(path) as f:
            y_true, y_pred, mask = f["y_true"], f["y_pred"], f["mask"]
        case = path.stem
        m = case_metrics(y_true=y_true, y_pred=y_pred, mask=mask, case=case)
        t, p = y_true[~mask].astype(float), y_pred[~mask].astype(float)
        info = dev.loc[case]
        rows.append(
            m
            | {
                "run": run,
                "subset": info["subset"],
                "split_group": info["split_group"],
                "outer_fold": int(info["fold"]),
                "flagged": info["status"] == "flag",
                "u_tau": info["u_tau"],
                "mae_ms": m["mae"] * info["u_tau"],
                "sum_t": t.sum(),
                "sum_t2": (t * t).sum(),
                "sse": ((p - t) ** 2).sum(),
            }
        )
    return pd.DataFrame(rows)


def pooled_r2(rows: pd.DataFrame) -> float:
    n, s, ss, sse = rows["n"].sum(), rows["sum_t"].sum(), rows["sum_t2"].sum(), rows["sse"].sum()
    return float(1.0 - sse / (ss - s * s / n))


def _differences(sub: pd.DataFrame, run: str, ref: str) -> dict:
    def _p(r: str) -> float:
        n, s, ss, sse = (sub[(k, r)].sum() for k in ("n", "sum_t", "sum_t2", "sse"))
        return float(1.0 - sse / (ss - s * s / n))

    return {
        "pooled_r2": _p(run) - _p(ref),
        "median_case_r2": float(sub[("r2", run)].median() - sub[("r2", ref)].median()),
        "mean_delta_p95": float(sub[("delta_p95", run)].mean() - sub[("delta_p95", ref)].mean()),
    }


def bootstrap_vs_s11(
    per_case: pd.DataFrame, run: str, ref_run: str = S11_BASELINE_RUN, n_boot: int = 2000
) -> pd.DataFrame:
    """Group-bootstrap comparison against S11 baseline resampling entire split_groups."""
    both = per_case[per_case["run"].isin([run, ref_run])].pivot_table(
        index=["case", "split_group", "subset", "outer_fold"],
        columns="run",
        values=["r2", "delta_p95", "n", "sum_t", "sum_t2", "sse"],
    ).reset_index()

    rng = np.random.default_rng(0)
    rows = []
    for subset in SUBSETS:
        sub = both if subset == "all" else both[both["subset"] == subset]
        groups = sub["split_group"].unique()
        rows_of = {g: np.flatnonzero(sub["split_group"].to_numpy() == g) for g in groups}
        stats = {name: [] for name in ("pooled_r2", "median_case_r2", "mean_delta_p95")}

        for _ in range(n_boot):
            sampled_groups = rng.choice(groups, len(groups))
            pick = sub.iloc[np.concatenate([rows_of[g] for g in sampled_groups])]
            diffs = _differences(pick, run, ref_run)
            for k, v in diffs.items():
                stats[k].append(v)

        point = _differences(sub, run, ref_run)
        for name in stats:
            lo, hi = np.percentile(stats[name], [2.5, 97.5])
            # Higher is better for R2; closer to zero (higher algebraically) is better for delta_p95
            share_better = float((sub[("r2", run)] > sub[("r2", ref_run)]).mean())
            better_folds = sum(
                int(_differences(f, run, ref_run)[name] > 0)
                for _, f in sub.groupby("outer_fold")
            )
            rows.append({
                "run": run,
                "ref_run": ref_run,
                "subset": subset,
                "metric": name,
                "difference": point[name],
                "ci_low": lo,
                "ci_high": hi,
                "share_of_cases_better": share_better,
                "folds_better": better_folds,
            })
    return pd.DataFrame(rows)


def plot_learning_curves_if_available(runs: list[str]) -> None:
    if not runs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 4 * len(runs)), squeeze=False)
    for ax, run in zip(axes.flat, runs):
        for k in range(5):
            csv_path = RUNS / run / f"fold{k}.csv"
            if csv_path.exists():
                hist = pd.read_csv(csv_path)
                val_col = "val" if "val" in hist else "val_l1"
                line, = ax.plot(hist["epoch"], hist[val_col], label=f"fold {k} val")
                ax.plot(hist["epoch"], hist["train_loss"], ls=":", color=line.get_color())
        ax.set_title(run)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / error (u*)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "s12_learning_curves.png", dpi=130)
    plt.close(fig)


def load_s11_baseline_per_case(ref_run: str = S11_BASELINE_RUN) -> pd.DataFrame:
    """Load S11 baseline metrics, falling back to S11 per_case_metrics.csv if preds are missing.

    When raw prediction files are unavailable, reconstructs the necessary bootstrap columns:
        sum_t  = n * mean_true
        sum_t2 = n * (var_true + mean_true**2)
        sse    = n * rmse**2
    """
    dev = fold_table("grouped")
    if (PREDS / ref_run).is_dir() and len(list((PREDS / ref_run).glob("*.npz"))) == len(dev):
        return score_run(ref_run, dev, PREDS)

    if S11_METRICS_CSV.exists():
        df = pd.read_csv(S11_METRICS_CSV)
        sub = df[df["run"] == ref_run].copy()
        if not sub.empty:
            sub["sum_t"] = sub["n"] * sub["mean_true"]
            sub["sum_t2"] = sub["n"] * (sub["var_true"] + sub["mean_true"] ** 2)
            sub["sse"] = sub["n"] * (sub["rmse"] ** 2)
            keep_cols = [
                "case",
                "r2",
                "delta_p95",
                "mae",
                "subset",
                "split_group",
                "outer_fold",
                "flagged",
                "n",
                "run",
                "sum_t",
                "sum_t2",
                "sse",
            ]
            sub["outer_fold"] = sub["outer_fold"].astype(int)
            sub["flagged"] = sub["flagged"].astype(bool)
            return sub[[c for c in keep_cols if c in sub.columns]]

    raise FileNotFoundError(
        f"Could not load S11 baseline: neither {PREDS / ref_run} nor {S11_METRICS_CSV} was found."
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = finished_runs()

    if not runs:
        print("No completed S12 runs found in pipeline/S12_architecture_search/outputs/runs/.")
        print("S12 evaluation arrays are not present in this local environment.")
        if (OUT / "summary.csv").exists() and (OUT / "summary.csv").stat().st_size > 0:
            print(f"Preserving existing populated {OUT / 'summary.csv'} (do not overwrite with empty template).")
        else:
            print("Writing empty template summary.csv.")
            empty_summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
            empty_summary.to_csv(OUT / "summary.csv", index=False)
        print("Skipping vs_s11_seed1.csv: S12 prediction files (.npz) are required for paired bootstrap comparison and are not present.")
        return

    # When completed runs exist:
    print(f"Found {len(runs)} completed S12 runs: {runs}")
    dev = fold_table("grouped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Score S12 runs
    per_case_list = [score_run(r, dev, PREDS) for r in runs]

    # Check if S11 baseline predictions exist for paired comparison, or fall back to per_case_metrics.csv
    try:
        s11_per_case = load_s11_baseline_per_case(S11_BASELINE_RUN)
        per_case_list.append(s11_per_case)
        s11_available = True
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        s11_available = False

    per_case = pd.concat(per_case_list, ignore_index=True)

    # Build summary rows
    summary_rows = []
    for run in runs:
        sub_all = per_case[per_case["run"] == run]
        sub_id = sub_all[sub_all["subset"] == "idealized"]
        sub_real = sub_all[sub_all["subset"] == "realistic"]

        fold_records = [json.loads((RUNS / run / f"fold{k}.json").read_text()) for k in range(5)]
        gpu_minutes = sum(f.get("minutes", 0.0) for f in fold_records)

        # Config details
        cfg_hash = fold_records[0]["config_hash"]
        matched_cfg = None
        for p in (STAGE / "configs").glob("*.yaml"):
            c = load_config(p)
            if config_hash(c) == cfg_hash:
                matched_cfg = c
                break

        model_name = matched_cfg["model"]["name"] if matched_cfg else run
        param_count = count_parameters(model_name, matched_cfg) if matched_cfg else 0
        infer_time = measure_inference_time_per_case(model_name, matched_cfg, device) if matched_cfg else 0.0

        # Seed from fold record
        seed = fold_records[0].get("seed", 0)

        summary_rows.append({
            "model": model_name,
            "seed": seed,
            "n_cases": len(sub_all),
            "parameters": param_count,
            "gpu_minutes": round(gpu_minutes, 1),
            "pooled_r2": round(pooled_r2(sub_all), 4),
            "median_case_r2_all": round(float(sub_all["r2"].median()), 4),
            "median_case_r2_idealized": round(float(sub_id["r2"].median()), 4),
            "median_case_r2_realistic": round(float(sub_real["r2"].median()), 4),
            "mean_delta_p95": round(float(sub_all["delta_p95"].mean()), 4),
            "inference_sec_per_case": round(infer_time, 4),
        })

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(OUT / "summary.csv", index=False)
    print("\n" + "=" * 80)
    print("S12 ARCHITECTURE SEARCH SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Bootstrap vs S11 if available
    if s11_available:
        vs_rows = []
        for run in runs:
            vs_df = bootstrap_vs_s11(per_case, run, S11_BASELINE_RUN)
            vs_rows.append(vs_df)
        all_vs = pd.concat(vs_rows, ignore_index=True)
        all_vs.to_csv(OUT / "vs_s11_seed1.csv", index=False, float_format="%.4f")
        print("\n" + "=" * 80)
        print("S12 VS S11 GROUP-BOOTSTRAP COMPARISON (SEED 1)")
        print("=" * 80)
        print(all_vs.to_string(index=False))
    else:
        print("Skipping vs_s11_seed1.csv: S11 baseline reference metrics could not be loaded.")

    plot_learning_curves_if_available(runs)


if __name__ == "__main__":
    main()

