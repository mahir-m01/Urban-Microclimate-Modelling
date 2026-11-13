"""
S12 sanity check: Candidate architectures must be able to memorise 4 training patches.

Adapted from S11 overfit check to accept any registered model architecture name via CLI.
If the loss on 4 fixed patches does not fall close to zero, something is wrong with shapes,
padding, or gradient flow, and a long training run would only hide it. Writes
outputs/overfit_check_<model>.json with the loss curve and summary metrics.

Usage:
    uv run python pipeline/S12_architecture_search/overfit_check.py --model res_unet
    uv run python pipeline/S12_architecture_search/overfit_check.py --model attn_unet
    uv run python pipeline/S12_architecture_search/overfit_check.py --model fno
    uv run python pipeline/S12_architecture_search/overfit_check.py --model all --steps 50  # fast CPU check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
import yaml

from urbanwind.dl.cache import CACHE_DIR, CHANNELS, dev_cases
from urbanwind.dl.datasets import PatchDataset
from urbanwind.dl.losses import masked_l1
from urbanwind.dl.nets import build_model
from urbanwind.dl.train import pick_device

STAGE_DIR = pathlib.Path(__file__).resolve().parent
CONFIGS_DIR = STAGE_DIR / "configs"
OUT = STAGE_DIR / "outputs"
STEPS = 2000
N_PATCHES = 4
MODELS = ("res_unet", "attn_unet", "fno")


def get_patches(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve 4 patches from local cache if present, otherwise generate reproducible synthetic patches."""
    if (CACHE_DIR / "UA25_d45.npz").exists():
        cases = dev_cases().query("fold != 0").index.tolist()
        data = PatchDataset(cases, mirror=False, seed=0)
        x, y, mask = (torch.stack(t).to(device) for t in zip(*(data[i] for i in range(N_PATCHES))))
    else:
        # Fallback for environments without local raw/dl cache
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(N_PATCHES, len(CHANNELS), 256, 256, generator=rng, device=device)
        y = torch.rand(N_PATCHES, 1, 256, 256, generator=rng, device=device) + 0.5
        mask = torch.ones_like(y, dtype=torch.bool, device=device)
    return x, y, mask


def run_overfit_check(model_name: str, steps: int = STEPS, device: torch.device | None = None) -> dict:
    torch.manual_seed(0)
    if device is None:
        device = pick_device()
    x, y, mask = get_patches(device)

    cfg_file = CONFIGS_DIR / f"{model_name}.yaml"
    model_kwargs = {}
    if cfg_file.exists():
        with open(cfg_file) as f:
            raw_cfg = yaml.safe_load(f) or {}
            model_kwargs = dict(raw_cfg.get("model", {}).get("kwargs", {}))
    # Dropout is set to 0.0 for the overfit memorisation check (matching S11)
    model_kwargs["dropout"] = 0.0
    model = build_model(model_name, len(CHANNELS), **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    use_amp = (device.type != "cpu")
    scaler = (
        torch.amp.GradScaler(device.type, enabled=use_amp)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
        else None
    )

    curve, t0 = [], time.time()
    print(f"\n--- Overfit check for '{model_name}' ({steps} steps on {device.type}) ---", flush=True)

    for step in range(steps):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x)
        loss = masked_l1(pred, y, mask)
        opt.zero_grad(set_to_none=True)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        curve.append(loss.item())
        interval = max(1, steps // 10)
        if step % interval == 0 or step == steps - 1:
            print(f"step {step:4d}  masked L1 {curve[-1]:.4f}", flush=True)

    target_mean = float((y * mask).sum() / mask.sum())
    result = {
        "model": model_name,
        "config_file": str(cfg_file.relative_to(STAGE_DIR)) if cfg_file.exists() else None,
        "config": model_kwargs,
        "device": device.type,
        "patches": N_PATCHES,
        "steps": steps,
        "seconds": round(time.time() - t0, 1),
        "first_loss": curve[0],
        "final_loss": curve[-1],
        "mean_target": target_mean,
        "final_loss_relative_to_mean_target": curve[-1] / target_mean,
        "curve_every_10": curve[::10] if len(curve) >= 10 else curve,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"overfit_check_{model_name}.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n")
    # Also write canonical overfit_check.json
    (OUT / "overfit_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {out_file}")
    print({k: v for k, v in result.items() if k != "curve_every_10"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check: overfit 4 patches with candidate DL architecture")
    parser.add_argument("--model", default="res_unet", choices=[*MODELS, "all"], help="Architecture name to test")
    parser.add_argument("--steps", type=int, default=STEPS, help="Number of gradient steps")
    args = parser.parse_args()

    targets = list(MODELS) if args.model == "all" else [args.model]
    for m in targets:
        run_overfit_check(m, steps=args.steps)


if __name__ == "__main__":
    main()

