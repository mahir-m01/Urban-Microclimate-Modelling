"""
train.py: one entry point for every deep-learning experiment.

    uv run python -m urbanwind.dl.train --config <yaml> --seed <n> [--folds 0 1 ...] [--repredict]

For each outer fold the network trains on the other four folds. Whole split groups of the training
folds (never the scored fold) are held back as an inner validation set, the same share of the
idealized and the realistic cases, with no single city or layout family above a quarter of it.
After every epoch a smoothed copy of the weights (an exponential moving average, EMA) predicts
those cases whole; training stops once their error, averaged per subset, has not improved for
``patience`` epochs. The best EMA weights then predict every case of the scored fold on its native
grid. With ``inner_val_fraction: 0`` there is no validation: the network trains on every training
case for the full schedule and keeps its final EMA weights.

Every epoch has the same number of steps (``steps_per_epoch``), so all folds get the same training.
Once all five folds are done, the out-of-fold predictions of all 482 development cases are scored
by ``urbanwind.evaluate``, which appends a row to ``results/runs.csv``.

A finished fold leaves ``<out_dir>/<exp_id>/fold<k>.json`` (with the code version and a hash of the
settings) and is skipped on a rerun; an unfinished fold resumes from its last epoch, and refuses to
if the settings changed.

Outputs
- results/preds/<exp_id>/<case>.npz          out-of-fold fields for the evaluator (not tracked)
- results/checkpoints/<exp_id>/fold<k>.pt    best EMA weights per fold (not tracked)
- <out_dir>/<exp_id>/fold<k>.json, .csv      per-fold summary and per-epoch history
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from urbanwind.data import REPO_ROOT, load_case
from urbanwind.dl.cache import CHANNELS, dev_cases, load_cached
from urbanwind.dl.datasets import PatchDataset, predict_arrays, predict_canonical
from urbanwind.dl.losses import LOSSES
from urbanwind.dl.nets import build_model
from urbanwind.evaluate import main as evaluate_main

PREDS = REPO_ROOT / "results" / "preds"
CHECKPOINTS = REPO_ROOT / "results" / "checkpoints"
RANDOM_SPLIT_SEED = 42  # the random-split folds stay the same for every training seed
NOT_HASHED = ("stage", "out_dir", "workers")  # settings that do not change what is trained

DEFAULTS: dict[str, Any] = {
    "stage": "S11", "out_dir": "pipeline/S11_unet_baseline/outputs/runs", "split": "grouped",
    "model": {"name": "unet", "kwargs": {}}, "loss": "l1",
    "lr": 1e-3, "min_lr": 1e-5, "weight_decay": 1e-4, "batch": 16, "clip": 1.0, "amp": True,
    "steps_per_epoch": 70, "max_epochs": 100, "patience": 30, "ema": 0.999,
    "inner_val_fraction": 0.15, "inner_group_cap": 0.25, "val_metric": "rmse",
    "patch": 256, "border": 16, "mirror": True, "transpose45": True,
    "mirror_tta": True, "native_upsample": "bilinear", "infer_window": 0, "infer_border": 64, "infer_margin": 128,
    "workers": 0,
}


def load_config(path: pathlib.Path) -> dict[str, Any]:
    cfg = DEFAULTS | yaml.safe_load(path.read_text())
    unknown = set(cfg) - set(DEFAULTS) - {"exp_id"}
    if unknown or "exp_id" not in cfg:
        raise ValueError(f"{path}: needs exp_id; unknown keys {sorted(unknown)}")
    return cfg


def config_hash(cfg: dict) -> str:
    kept = {k: v for k, v in cfg.items() if k not in NOT_HASHED}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()[:12]


def code_version() -> dict:
    """The git commit the code came from, and whether the code differed from it (a Kaggle bundle carries a file)."""
    bundled = REPO_ROOT / "code_version.json"
    if bundled.exists():
        return json.loads(bundled.read_text())
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    # only code and settings count; run outputs and figures written next to the code do not make it dirty
    changed = git("status", "--porcelain", "--untracked-files=all", "--", "src", "pipeline").splitlines()
    code = [line for line in changed if line.endswith((".py", ".yaml", ".yml", ".toml"))]
    return {"git_sha": git("rev-parse", "--short", "HEAD") or "unknown", "git_dirty": bool(code)}


def pick_device() -> torch.device:
    """CUDA, then Apple's GPU, then the CPU; URBANWIND_DEVICE (for example "cpu") overrides the choice."""
    if os.environ.get("URBANWIND_DEVICE"):
        return torch.device(os.environ["URBANWIND_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fold_table(split: str) -> pd.DataFrame:
    """Development cases with the outer fold each is scored in."""
    dev = dev_cases()
    if split == "random":
        # cases dealt into five folds at random, ignoring their city or layout family
        rng = np.random.default_rng(RANDOM_SPLIT_SEED)
        dev["fold"] = rng.permutation(np.arange(len(dev)) % dev["fold"].nunique())
    elif split != "grouped":
        raise ValueError(f"split must be 'grouped' or 'random', not {split!r}")
    return dev


def inner_split(train: pd.DataFrame, fraction: float, fold: int, by_group: bool,
                cap: float = 0.25) -> tuple[list[str], list[str]]:
    """Hold back whole groups (single cases for the random split) for early stopping.

    Each subset (idealized, realistic) gives the same share ``fraction`` of its cases, so model
    selection reflects both. Groups are taken in a fixed random order; a group larger than ``cap``
    of the planned validation set is passed over, so no single city dominates it. The choice
    depends on the fold only, not on the seed, so the seeds differ in training alone.
    """
    rng = np.random.default_rng(1000 + fold)
    planned = fraction * len(train)
    val: list[str] = []
    for subset in sorted(train["subset"].unique()):
        part = train[train["subset"] == subset]
        units = part["split_group"] if by_group else pd.Series(part.index, index=part.index)
        sizes = units.value_counts()
        held = 0
        for unit in rng.permutation(sorted(sizes.index)):
            if held >= fraction * len(part):
                break
            if sizes[unit] > cap * planned:
                continue
            val += part.index[units == unit].tolist()
            held += int(sizes[unit])
    return sorted(set(train.index) - set(val)), sorted(val)


def inference(cfg: dict) -> dict:
    return {"window": cfg["infer_window"], "border": cfg["infer_border"], "margin": cfg["infer_margin"]}


def train_epoch(model, ema, loader, opt, scaler, loss_fn, device, cfg) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu"):
            pred = model(x)
        loss = loss_fn(pred, y, mask)  # every loss works in float32, outside autocast
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        scaler.step(opt)
        scaler.update()
        ema.update_parameters(model)
        total, n = total + loss.item(), n + 1
    return total / max(n, 1)


def validate(model, val_cases: list[dict], device, cfg) -> float:
    """Each inner-validation case's error (RMSE or MAE, in u*), averaged within each subset, then over subsets."""
    model.eval()
    by_subset: dict[str, list[float]] = {}
    with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu"):
        for c in val_cases:
            err = (predict_canonical(model, c, device, fill=False, **inference(cfg)) - c["y"])[c["valid"]]
            value = np.sqrt(np.mean(err ** 2)) if cfg["val_metric"] == "rmse" else np.mean(np.abs(err))
            by_subset.setdefault(c["subset"], []).append(float(value))
    return float(np.mean([np.mean(v) for v in by_subset.values()]))


def _save_atomic(obj, path: pathlib.Path) -> None:
    """Write to a temporary name first, so a crash mid-save leaves the previous file intact."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _save_state(path: pathlib.Path, parts: dict, state: dict) -> None:
    _save_atomic({name: part.state_dict() for name, part in parts.items()} | state, path)


def _load_state(path: pathlib.Path, parts: dict, cfg: dict, device) -> dict:
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("config") != config_hash(cfg):
        raise ValueError(f"{path} was written with other settings; delete it to start this fold again")
    for name, part in parts.items():
        part.load_state_dict(saved.pop(name))
    return saved


def ema_average(decay: float) -> Callable:
    """Moving average of the weights with a warm-up: early on it averages over few steps, so the random
    starting weights fade within the first few hundred steps instead of lingering (0.999 ** 1750 = 17%)."""
    def average(averaged: torch.Tensor, current: torch.Tensor, num_averaged: torch.Tensor) -> torch.Tensor:
        n = float(num_averaged)
        d = min(decay, (1.0 + n) / (10.0 + n))
        return d * averaged + (1.0 - d) * current
    return average


def _setup(cfg, seed: int, fit: list[str], device) -> dict:
    """Model, its EMA copy, optimiser, schedule, loss scaler and patch loader for one fold."""
    torch.manual_seed(seed)
    model = build_model(cfg["model"]["name"], len(CHANNELS), **cfg["model"]["kwargs"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    data = PatchDataset(fit, patch=cfg["patch"], mirror=cfg["mirror"], seed=seed, border=cfg["border"],
                        length=cfg["steps_per_epoch"] * cfg["batch"], transpose45=cfg["transpose45"])
    return {"model": model, "opt": opt, "data": data,
            "ema": AveragedModel(model, avg_fn=ema_average(cfg["ema"])),
            "sched": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["min_lr"]),
            "scaler": torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type != "cpu"),
            "loader": DataLoader(data, batch_size=cfg["batch"], shuffle=False, drop_last=True,
                                 num_workers=cfg["workers"])}


def fit_fold(cfg, seed: int, fit: list[str], val: list[str], device, ckpt: pathlib.Path | None,
             report: Callable[[int, float], None] | None = None) -> tuple[pd.DataFrame, dict]:
    """Train on ``fit``, early-stop on ``val`` (if any), and keep the best EMA weights in ``ckpt``.

    With ``ckpt`` set, the full training state is also saved after every epoch (``<ckpt>_last.pt``),
    so a run that was stopped resumes from its last finished epoch. Patches depend only on (seed,
    epoch, index), so a resumed run sees the same data; only dropout draws differ. ``report(epoch,
    val)`` is called after every epoch; the S11 Optuna study uses it to stop unpromising trials early.
    """
    s = _setup(cfg, seed, fit, device)
    parts = {k: s[k] for k in ("model", "ema", "opt", "sched", "scaler")}
    val_cases = [load_cached(case) for case in val]
    last = None if ckpt is None else ckpt.with_name(f"{ckpt.stem}_last.pt")
    state = {"history": [], "best": math.inf, "since_best": 0, "config": config_hash(cfg)}
    if last is not None and last.exists():
        state = _load_state(last, parts, cfg, device)
        print(f"  resuming after epoch {len(state['history']) - 1}", flush=True)
    for epoch in range(len(state["history"]), cfg["max_epochs"]):
        if state["since_best"] >= cfg["patience"]:
            break
        t0 = time.time()
        s["data"].epoch = epoch
        train_loss = train_epoch(s["model"], s["ema"], s["loader"], s["opt"], s["scaler"], LOSSES[cfg["loss"]],
                                 device, cfg)
        val_loss = validate(s["ema"].module, val_cases, device, cfg) if val_cases else math.nan
        state["history"].append({"epoch": epoch, "train_loss": train_loss, "val": val_loss,
                                 "lr": s["opt"].param_groups[0]["lr"], "seconds": time.time() - t0})
        s["sched"].step()
        print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if report is not None:
            report(epoch, val_loss)
        improved = not val_cases or val_loss < state["best"]  # without validation the latest weights count
        state["best"], state["since_best"] = ((val_loss, 0) if improved else (state["best"], state["since_best"] + 1))
        if improved and ckpt is not None:
            _save_atomic(s["ema"].module.state_dict(), ckpt)
        if last is not None:
            _save_state(last, parts, state)
    hist = pd.DataFrame(state["history"])
    best_epoch = int(hist["val"].idxmin()) if val_cases else len(hist) - 1
    info = {"epochs_run": len(hist), "best_epoch": best_epoch, "best_val": state["best"] if val_cases else None,
            "val_metric": cfg["val_metric"], "minutes": round(hist["seconds"].sum() / 60, 1), "n_fit": len(fit),
            "n_val": len(val), "steps_per_epoch": len(s["loader"]), "device": device.type}
    return hist, info


def native_truth(case: str, cached: dict) -> np.ndarray:
    """U/u_tau on the native grid: from the cache, or from the raw files for a cache built before it stored it."""
    if "U_native" in cached:
        return cached["U_native"]
    return load_case(case)["U"].astype(np.float32)


def write_predictions(model, cases: list[str], pred_dir: pathlib.Path, device, cfg) -> None:
    """Native-grid fields in the evaluator's format: NaN outside the valid cells, mask True there."""
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu"):
        for case in cases:
            cached = load_cached(case)
            truth = native_truth(case, cached)
            valid = np.isfinite(truth)
            pred = predict_arrays(model, cached, device=device, mirror_tta=cfg["mirror_tta"],
                                  upsample=cfg["native_upsample"], **inference(cfg))
            np.savez_compressed(pred_dir / f"{case}.npz", y_true=truth.astype(np.float32),
                                y_pred=np.where(valid, pred, np.nan).astype(np.float32), mask=~valid)


def load_best(cfg, ckpt: pathlib.Path, device) -> torch.nn.Module:
    model = build_model(cfg["model"]["name"], len(CHANNELS), **cfg["model"]["kwargs"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def prediction_record(cfg: dict) -> dict:
    """What a fold's predictions depend on beyond its weights; the results script checks folds agree."""
    keys = ("mirror_tta", "native_upsample", "infer_window", "infer_border", "infer_margin")
    return {k: cfg[k] for k in keys} | {"predict_code": code_version()}


def repredict_fold(cfg, fold: int, folds: pd.DataFrame, exp_id: str, device) -> None:
    """Rewrite a finished fold's predictions from its best weights with the current prediction settings."""
    summary = REPO_ROOT / cfg["out_dir"] / exp_id / f"fold{fold}.json"
    ckpt = CHECKPOINTS / exp_id / f"fold{fold}.pt"
    if not (summary.exists() and ckpt.exists()):
        raise FileNotFoundError(f"fold {fold} of {exp_id} has not finished training")
    t0 = time.time()
    write_predictions(load_best(cfg, ckpt, device), folds.index[folds["fold"] == fold].tolist(),
                      PREDS / exp_id, device, cfg)
    info = json.loads(summary.read_text()) | prediction_record(cfg) | {
        "predict_minutes": round((time.time() - t0) / 60, 1)}
    summary.write_text(json.dumps(info, indent=2) + "\n")
    print(f"fold {fold}: predictions rewritten", flush=True)


def run_fold(cfg, seed: int, fold: int, folds: pd.DataFrame, exp_id: str, device) -> dict:
    out = REPO_ROOT / cfg["out_dir"] / exp_id
    summary = out / f"fold{fold}.json"
    if summary.exists():
        print(f"fold {fold}: already done, skipping ({summary})")
        return json.loads(summary.read_text())
    train, scored = folds[folds["fold"] != fold], folds.index[folds["fold"] == fold].tolist()
    fit, val = inner_split(train, cfg["inner_val_fraction"], fold, by_group=cfg["split"] == "grouped",
                           cap=cfg["inner_group_cap"])
    print(f"fold {fold}: {len(fit)} training cases, {len(val)} inner-validation cases, {len(scored)} scored", flush=True)
    ckpt = CHECKPOINTS / exp_id / f"fold{fold}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    train_code = code_version()
    hist, info = fit_fold(cfg, seed, fit, val, device, ckpt)
    t0 = time.time()
    write_predictions(load_best(cfg, ckpt, device), scored, PREDS / exp_id, device, cfg)
    info |= {"fold": fold, "seed": seed, "config_hash": config_hash(cfg), "train_code": train_code,
             "predict_minutes": round((time.time() - t0) / 60, 1), "inner_val_cases": val,
             "inner_val_by_subset": folds.loc[val, "subset"].value_counts().to_dict(),
             "n_scored": len(scored)} | prediction_record(cfg)
    hist.to_csv(out / f"fold{fold}.csv", index=False, float_format="%.6g")
    summary.write_text(json.dumps(info, indent=2) + "\n")
    ckpt.with_name(f"{ckpt.stem}_last.pt").unlink(missing_ok=True)  # the fold is done; only the best weights stay
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a network on the grouped folds and score it.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="only these outer folds (default: all)")
    parser.add_argument("--repredict", action="store_true",
                        help="no training: rewrite finished folds' predictions with the current settings, then score")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    exp_id = f"{cfg['exp_id']}_seed{args.seed}"
    device = pick_device()
    folds = fold_table(cfg["split"])
    todo = sorted(int(k) for k in folds["fold"].unique()) if args.folds is None else args.folds
    print(f"{exp_id} on {device}: folds {todo}; code {code_version()}", flush=True)
    for fold in todo:
        if args.repredict:
            repredict_fold(cfg, int(fold), folds, exp_id, device)
        else:
            run_fold(cfg, args.seed, int(fold), folds, exp_id, device)
    done = {p.stem for p in (PREDS / exp_id).glob("*.npz")}
    if done != set(folds.index):
        print(f"{len(done)} of {len(folds)} cases predicted; scoring waits for all five folds")
        return 0
    record = cfg | {"seed": args.seed, "config_hash": config_hash(cfg), "code": code_version()}
    return evaluate_main([str(PREDS / exp_id), "--stage", cfg["stage"], "--exp-id", exp_id,
                          "--config", json.dumps(record, sort_keys=True, default=str)])


if __name__ == "__main__":
    raise SystemExit(main())
