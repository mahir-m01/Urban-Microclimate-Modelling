"""
S11 sanity check: the U-Net must be able to memorise 4 training patches.

If the loss on 4 fixed patches does not fall close to zero, something is wrong with shapes, the
mask or the target scaling, and a long training run would only hide it. Writes
outputs/overfit_check.json with the loss curve.

Usage: uv run python pipeline/S11_unet_baseline/overfit_check.py
"""

from __future__ import annotations

import json
import pathlib
import time

import torch

from urbanwind.dl.cache import CHANNELS, dev_cases
from urbanwind.dl.datasets import PatchDataset
from urbanwind.dl.losses import masked_l1
from urbanwind.dl.nets import build_model
from urbanwind.dl.train import pick_device

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
STEPS = 2000
N_PATCHES = 4


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()
    cases = dev_cases().query("fold != 0").index.tolist()
    data = PatchDataset(cases, mirror=False, seed=0)
    x, y, mask = (torch.stack(t).to(device) for t in zip(*(data[i] for i in range(N_PATCHES))))
    model = build_model("unet", len(CHANNELS), dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type != "cpu")
    curve, t0 = [], time.time()
    for step in range(STEPS):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type != "cpu"):
            pred = model(x)
        loss = masked_l1(pred, y, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        curve.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  masked L1 {curve[-1]:.4f}", flush=True)
    target_mean = float((y * mask).sum() / mask.sum())
    result = {"device": device.type, "patches": N_PATCHES, "steps": STEPS, "seconds": round(time.time() - t0, 1),
              "first_loss": curve[0], "final_loss": curve[-1], "mean_target": target_mean,
              "final_loss_relative_to_mean_target": curve[-1] / target_mean, "curve_every_10": curve[::10]}
    OUT.mkdir(exist_ok=True)
    (OUT / "overfit_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "curve_every_10"})


if __name__ == "__main__":
    main()
