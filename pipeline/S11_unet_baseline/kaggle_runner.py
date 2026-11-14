"""Run a queue of trainings on a Kaggle GPU machine, one job per GPU at a time.

Used for the S11 final runs; later stages can reuse it with their own JOBS.

Each job is one `python -m urbanwind.dl.train --config <c> --seed <s> --folds <f>` on its own GPU.
A job only starts if it can finish before CAP_HOURS from the start of the session (the estimate
is the longest finished job so far, or FIRST_GUESS_MIN before any finished), so the session ends
on its own, inside Kaggle's limit. At the end, only the small results are kept in the output:
the fold records and histories, the best weights, the job logs and a summary. The predictions
are rebuilt later from the weights with `--repredict`.

Paths come from environment variables, so the same script can be tested on a laptop.

On Kaggle: attach the private datasets mahirm01/urbanwind-code (the code at one commit, with a
code_version.json) and mahirm01/urbantales-dl-cache (the S10 cache), set JOBS, and run the
notebook with "Save & Run All" on a GPU machine.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

JOBS = []  # (config file, seed, fold), set per notebook; S11_JOBS (JSON) overrides it for a local test
CAP_HOURS = 8.0
FIRST_GUESS_MIN = 100.0
POLL_S = int(os.environ.get("S11_POLL", "30"))

INPUT = pathlib.Path(os.environ.get("S11_INPUT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("S11_WORK", "/kaggle/working"))
REPO = pathlib.Path(os.environ.get("S11_REPO", "/tmp/repo"))  # outside the output folder: not saved


def unpack() -> None:
    bundle = next(INPUT.rglob("urbanwind_code.tar.gz"), None)
    if bundle is not None:
        with tarfile.open(bundle) as tar:
            tar.extractall(REPO)
    else:  # Kaggle unpacked the archive on upload: copy the tree, since inputs are read-only
        src = next(p.parent for p in INPUT.rglob("pyproject.toml") if (p.parent / "src" / "urbanwind").is_dir())
        shutil.copytree(src, REPO)
    cache = next(INPUT.rglob("UA25_d45.npz")).parent
    (REPO / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "processed" / "dl").symlink_to(cache)
    print("code:", json.loads((REPO / "code_version.json").read_text()), "| cache:", cache,
          f"({len(list(cache.glob('*.npz')))} cases)", flush=True)


def gpu_count() -> int:
    if os.environ.get("S11_FAKE_GPUS"):
        return int(os.environ["S11_FAKE_GPUS"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return 1
    return max(out.count("GPU "), 1)


def launch(job, gpu: int, logs: pathlib.Path) -> subprocess.Popen:
    cfg, seed, fold = job
    log = open(logs / f"{pathlib.Path(cfg).stem}_seed{seed}_fold{fold}.log", "w")
    cmd = [sys.executable, "-m", "urbanwind.dl.train", "--config", cfg, "--seed", str(seed), "--folds", str(fold)]
    env = os.environ | {"PYTHONPATH": f"{REPO / 'src'}:{REPO}", "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_queue(jobs: list, gpus: int, logs: pathlib.Path) -> dict:
    start, queue, running, done, durations = time.time(), list(jobs), {}, [], []
    while queue or running:
        for gpu in range(gpus):
            if gpu in running or not queue:
                continue
            expected = (max(durations) * 1.05 if durations else FIRST_GUESS_MIN * 60)
            if time.time() + expected > start + CAP_HOURS * 3600:
                continue
            job = queue.pop(0)
            running[gpu] = (job, time.time(), launch(job, gpu, logs))
            print(f"{time.strftime('%H:%M')} gpu {gpu} started {job}", flush=True)
        for gpu, (job, t0, proc) in list(running.items()):
            if proc.poll() is not None:
                minutes = (time.time() - t0) / 60
                durations.append(minutes * 60)
                done.append({"job": job, "gpu": gpu, "minutes": round(minutes, 1), "returncode": proc.returncode})
                print(f"{time.strftime('%H:%M')} gpu {gpu} finished {job} in {minutes:.0f} min, code {proc.returncode}",
                      flush=True)
                del running[gpu]
        if not running and queue:
            expected = max(durations) * 1.05 if durations else FIRST_GUESS_MIN * 60
            if time.time() + expected > start + CAP_HOURS * 3600:
                break  # nothing running and nothing else fits in the time left
        time.sleep(POLL_S)
    return {"done": done, "not_started": queue, "hours": round((time.time() - start) / 3600, 2)}


def save_outputs(summary: dict) -> None:
    out = WORK / "s11_final"
    # every stage's run records (each config names its own out_dir), then the weights
    sources = [(runs, out / "runs") for runs in sorted(REPO.glob("pipeline/*/outputs/runs"))]
    for src, dst in (*sources, (REPO / "results" / "checkpoints", out / "checkpoints")):
        if src.exists():
            # finished folds have already deleted their resume files; an unfinished fold keeps its own
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp"))
    runs_csv = REPO / "results" / "runs.csv"
    if runs_csv.exists():
        shutil.copy(runs_csv, out / "runs_on_kaggle.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    unpack()
    logs = WORK / "s11_final" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    gpus = gpu_count()
    jobs = [tuple(j) for j in json.loads(os.environ["S11_JOBS"])] if os.environ.get("S11_JOBS") else JOBS
    print(f"{gpus} GPU(s); {len(jobs)} jobs; cap {CAP_HOURS} h", flush=True)
    summary = run_queue(jobs, gpus, logs)
    save_outputs(summary)


if __name__ == "__main__":
    main()
