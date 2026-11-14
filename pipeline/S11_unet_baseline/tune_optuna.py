"""
S11 tuning: an Optuna search over the U-Net's training settings, inside a fixed time budget.

Every trial trains on fold 0's training cases and is judged on fold 0's inner-validation groups,
which are training data: no case that fold 0 scores is ever seen. (Those inner-validation cases
are scored in the other folds, so the five-fold result of the tuned settings is slightly
optimistic; the frozen test set gives the clean number.)

Trials are short (TRIAL_EPOCHS epochs of a fixed number of steps, with their own cosine schedule)
and a median pruner stops a trial early when its validation error is worse than the median of
earlier trials at the same epoch. A trial scores the mean validation error of its last SCORE_EPOCHS
epochs. The first trial is the S11 baseline, so every other trial is compared with it.

Nothing is lost when the time runs out:
- every finished trial is stored at once in an SQLite file, which a later run can resume;
- trials.csv and best.json are rewritten after every trial;
- no trial starts unless it can finish, and a running trial stops at the end of an epoch once
  the deadline has passed; it is recorded as pruned, so it is never ranked next to full trials.

Several workers can share one study (one per GPU): create it once with --init-only, then start
each worker with its own --worker number and the same --deadline.

Usage
  python pipeline/S11_unet_baseline/tune_optuna.py --hours 5.5 --out <dir>
  python pipeline/S11_unet_baseline/tune_optuna.py --init-only --out <dir>
  CUDA_VISIBLE_DEVICES=1 python pipeline/S11_unet_baseline/tune_optuna.py --worker 1 --deadline <t> --out <dir>
  python pipeline/S11_unet_baseline/tune_optuna.py --smoke --out <dir>     # tiny trials, to test the set-up
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import optuna
import torch

from urbanwind.dl.train import DEFAULTS, code_version, fit_fold, fold_table, inner_split, pick_device

TUNE_FOLD = 0
TRIAL_EPOCHS = 25  # 25 epochs of DEFAULTS["steps_per_epoch"] steps: every trial gets the same number of steps
SCORE_EPOCHS = 3  # a trial's score is its mean validation error over its last epochs, not one lucky epoch
STUDY = "s11_unet_fold0"
CONTEXTS = {"256/16": (256, 16), "384/64": (384, 64)}  # patch side / border left out of the loss
BASELINE = {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "base_width": 32, "depth": 4, "context": "256/16",
            "loss": "l1"}


def trial_config(trial: optuna.Trial, epochs: int) -> dict:
    """The settings Optuna varies; everything else stays at the S11 defaults."""
    p = {"lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
         "dropout": trial.suggest_float("dropout", 0.0, 0.3),
         "base_width": trial.suggest_categorical("base_width", [24, 32, 48]),
         "depth": trial.suggest_categorical("depth", [4, 5]),
         "context": trial.suggest_categorical("context", list(CONTEXTS)),
         "loss": trial.suggest_categorical("loss", ["l1", "mse"])}
    widths = [p["base_width"] * 2**i for i in range(p["depth"])]
    model = {"name": "unet", "kwargs": {"widths": widths, "bottleneck": 2 * widths[-1], "dropout": p["dropout"]}}
    patch, border = CONTEXTS[p["context"]]
    # patience above the epoch count: the pruner, not early stopping, ends weak trials
    return DEFAULTS | {"exp_id": STUDY, "lr": p["lr"], "weight_decay": p["weight_decay"], "loss": p["loss"],
                       "patch": patch, "border": border, "model": model, "max_epochs": epochs,
                       "patience": epochs + 1}


def shrink(cfg: dict) -> dict:
    """Smoke tests only: keep the sampled depth but make the network, patches and epochs tiny."""
    depth = len(cfg["model"]["kwargs"]["widths"])
    kwargs = cfg["model"]["kwargs"] | {"widths": [8 * 2**i for i in range(depth)], "bottleneck": 8 * 2**depth}
    return cfg | {"patch": 64, "border": 4, "batch": 4, "steps_per_epoch": 3, "model": cfg["model"] | {"kwargs": kwargs}}


class _Deadline(Exception):
    """Raised inside a trial when the time budget is used up."""


def make_objective(fit, val, device, epochs: int, deadline: float, seconds_per_trial: list[float], smoke: bool):
    def objective(trial: optuna.Trial) -> float:
        cfg = trial_config(trial, epochs)
        if smoke:
            cfg = shrink(cfg)
        trial.set_user_attr("code", code_version())
        history: list[float] = []

        def report(epoch: int, value: float) -> None:
            history.append(value)
            trial.report(value, epoch)
            if time.time() > deadline:
                raise _Deadline()
            if trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.time()
        try:
            fit_fold(cfg, seed=0, fit=fit, val=val, device=device, ckpt=None, report=report)
        except _Deadline as err:
            # an unfinished trial must not be ranked next to full ones: record it as pruned
            trial.set_user_attr("stopped_by_deadline_after_epochs", len(history))
            raise optuna.TrialPruned("time budget used up") from err
        except torch.cuda.OutOfMemoryError as err:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"out of memory: {cfg['model']['kwargs']}, patch {cfg['patch']}") from err
        finally:
            seconds_per_trial.append(time.time() - t0)
        return float(sum(history[-SCORE_EPOCHS:]) / len(history[-SCORE_EPOCHS:]))
    return objective


def _write(path: pathlib.Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")  # two workers write these files: never leave half a file
    tmp.write_text(text)
    tmp.replace(path)


def save_progress(study: optuna.Study, out: pathlib.Path) -> None:
    _write(out / "trials.csv", study.trials_dataframe().to_csv(index=False))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        best = min(done, key=lambda t: t.value)
        _write(out / "best.json", json.dumps({"trial": best.number, "val_rmse": best.value, "params": best.params,
                                              "n_trials": len(study.trials), "n_complete": len(done)}, indent=2))


def open_study(out: pathlib.Path, worker: int) -> optuna.Study:
    """The shared study in out/optuna.db.

    Each worker samples with its own seed, so two workers never propose the same trial; the seed also
    moves on with the number of trials already stored, so a resumed study does not replay old ones.
    """
    storage = optuna.storages.RDBStorage(f"sqlite:///{(out / 'optuna.db').resolve()}",
                                         engine_kwargs={"connect_args": {"timeout": 60}})
    done = len(optuna.load_study(study_name=STUDY, storage=storage).trials) if STUDY in [
        s.study_name for s in optuna.get_all_study_summaries(storage)] else 0
    return optuna.create_study(study_name=STUDY, storage=storage, load_if_exists=True, direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=worker + 1000 * done),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=6))


def wait_for_baseline(study: optuna.Study, timeout: float = 900.0) -> None:
    """Other workers start only once worker 0 has taken the queued baseline trial.

    Two workers that start together can otherwise both pick up the one queued trial, and the second
    to finish crashes when it tries to record a trial that is already complete.
    """
    waiting = optuna.trial.TrialState.WAITING
    end = time.time() + timeout
    while all(t.state == waiting for t in study.get_trials(deepcopy=False)) and time.time() < end:
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search for the S11 U-Net settings on fold 0.")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--hours", type=float, default=5.5, help="time budget, counted from now")
    parser.add_argument("--deadline", type=float, default=None, help="shared end time (Unix seconds); overrides --hours")
    parser.add_argument("--worker", type=int, default=0, help="worker number; with several GPUs, one worker per GPU")
    parser.add_argument("--init-only", action="store_true", help="create the study and queue the baseline, then exit")
    parser.add_argument("--smoke", action="store_true", help="2 tiny trials on a few cases, to test the set-up")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    study = open_study(args.out, args.worker)
    if not study.trials:
        study.enqueue_trial(BASELINE)
    if args.init_only:
        return
    start = time.time()
    deadline = args.deadline or start + args.hours * 3600
    folds = fold_table("grouped")
    fit, val = inner_split(folds[folds["fold"] != TUNE_FOLD], DEFAULTS["inner_val_fraction"], TUNE_FOLD,
                           by_group=True, cap=DEFAULTS["inner_group_cap"])
    epochs = TRIAL_EPOCHS
    if args.smoke:
        fit, val, epochs, deadline = fit[:6], val[:2], 2, start + 600
    device = pick_device()
    print(f"worker {args.worker} on {device}; {len(fit)} training and {len(val)} validation cases; "
          f"{len(study.trials)} trials so far", flush=True)
    if args.worker > 0:
        wait_for_baseline(study)
    seconds: list[float] = []
    objective = make_objective(fit, val, device, epochs, deadline, seconds, args.smoke)
    while True:
        # start another trial only if one as long as this worker's longest so far would end before the deadline
        if time.time() + (max(seconds) if seconds else 0.0) > deadline or (args.smoke and len(seconds) >= 2):
            break
        study.optimize(objective, n_trials=1, catch=(RuntimeError,))
        save_progress(study, args.out)
        print(f"worker {args.worker}: trial done in {seconds[-1] / 60:.1f} min; "
              f"{(deadline - time.time()) / 3600:.2f} h left", flush=True)
    save_progress(study, args.out)
    print(f"worker {args.worker} finished after {(time.time() - start) / 3600:.2f} h; "
          f"{len(study.trials)} trials in the study", flush=True)


if __name__ == "__main__":
    main()
