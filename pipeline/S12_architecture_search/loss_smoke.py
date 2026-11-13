"""
S12b smoke run: every loss must fall and stay finite when it trains the S11 U-Net.

Trains a fresh copy of the S11 tuned U-Net for a few hundred steps with each loss on the same 20
development cases (10 idealized and 10 realistic, never a test case), and checks that the loss
stays finite and that the mean of the last 20 steps is below the mean of the first 20. This only
shows that a loss can train the network; whether it helps is measured later in S15 (A4).

Writes outputs/loss_smoke.json with every loss curve.

Usage:
    uv run python pipeline/S12_architecture_search/loss_smoke.py                 # small patches, fine on a laptop CPU
    uv run python pipeline/S12_architecture_search/loss_smoke.py --patch 256     # training-size patches, on a GPU
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from urbanwind.dl.cache import CHANNELS, dev_cases
from urbanwind.dl.datasets import PatchDataset
from urbanwind.dl.losses import LOSSES
from urbanwind.dl.nets import build_model
from urbanwind.dl.train import load_config, pick_device

HERE = pathlib.Path(__file__).resolve().parent
S11_CONFIG = HERE.parent / "S11_unet_baseline" / "configs" / "unet_tuned.yaml"
OUT = HERE / "outputs" / "loss_smoke.json"
SMOKE_LOSSES = ("l1", "sobel", "tail_w1", "tail_w2", "fft", "mean_consistency")
CASES_PER_SUBSET = 10
WINDOW = 20  # steps averaged at the start and at the end of each curve


def smoke_cases() -> list[str]:
    """10 idealized and 10 realistic development cases, the same every time."""
    dev = dev_cases()
    picked = [dev[dev["subset"] == s].sample(CASES_PER_SUBSET, random_state=0).index for s in ("idealized", "realistic")]
    return sorted(c for idx in picked for c in idx)


def run_one(name: str, cases: list[str], cfg: dict, args, device: torch.device) -> dict:
    torch.manual_seed(0)
    data = PatchDataset(cases, patch=args.patch, mirror=True, seed=0, length=args.steps * args.batch)
    loader = DataLoader(data, batch_size=args.batch, shuffle=False, num_workers=0)
    model = build_model(cfg["model"]["name"], len(CHANNELS), **cfg["model"]["kwargs"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = cfg["amp"] and device.type != "cpu"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn, curve, t0 = LOSSES[name], [], time.time()
    model.train()
    for x, y, mask in loader:  # same patches, in the same order, for every loss
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(x)
        loss = loss_fn(pred, y, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        scaler.step(opt)
        scaler.update()
        curve.append(loss.item())
    first, last = sum(curve[:WINDOW]) / WINDOW, sum(curve[-WINDOW:]) / WINDOW
    finite = all(math.isfinite(v) for v in curve)
    result = {"loss": name, "first_mean": first, "last_mean": last, "finite": finite,
              "falls": finite and last < first, "seconds": round(time.time() - t0, 1), "curve": curve}
    print(f"{name:16s} first {first:.4f}  last {last:.4f}  finite {finite}  falls {result['falls']}  "
          f"({result['seconds']} s)", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patch", type=int, default=128, help="training uses 256; 128 keeps a CPU run short")
    args = parser.parse_args()
    if args.steps < 2 * WINDOW:
        parser.error(f"--steps must be at least {2 * WINDOW}")

    device, cfg, cases = pick_device(), load_config(S11_CONFIG), smoke_cases()
    print(f"device {device.type}; {len(cases)} cases; {args.steps} steps of {args.batch} patches of {args.patch}",
          flush=True)
    results = [run_one(name, cases, cfg, args, device) for name in SMOKE_LOSSES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"device": device.type, "cases": cases, "steps": args.steps, "batch": args.batch,
                               "patch": args.patch, "results": results}, indent=1) + "\n")
    failed = [r["loss"] for r in results if not r["falls"]]
    print(f"wrote {OUT}" + (f"; FAILED: {failed}" if failed else "; every loss fell and stayed finite"))


if __name__ == "__main__":
    main()
