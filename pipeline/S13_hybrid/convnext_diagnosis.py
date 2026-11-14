"""
S13: why does the ConvNeXt U-Net fit the training data worse than the U-Net?

Short controlled runs: every variant trains from the same seed on the same fold-0 training patches,
in the same order, with the S11 tuned optimiser settings (AdamW, learning rate 3.6e-4, weight decay
6.5e-5, gradient clipping at 1.0, mixed precision), and changes one thing. 400 steps of 16 patches of
128×128 cells are a short proxy for the full training, so only the ranking is meant to transfer.

Reports, per variant, the parameter count, the mean training loss over the first and last 100 steps,
and the median gradient norm before clipping. Writes outputs/convnext_diagnosis.csv.

Usage: uv run python -m pipeline.S13_hybrid.convnext_diagnosis
"""

from __future__ import annotations

import math
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from urbanwind.dl.cache import CHANNELS
from urbanwind.dl.datasets import PatchDataset
from urbanwind.dl.losses import masked_l1
from urbanwind.dl.nets import ConvNeXtBlock, build_model, conv_block
from urbanwind.dl.train import fold_table, pick_device

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
STEPS, BATCH, PATCH, BORDER, WINDOW = 400, 16, 128, 16, 100
LR, WD, CLIP = 3.6e-4, 6.5e-5, 1.0


def unet() -> nn.Module:
    return build_model("unet", len(CHANNELS), widths=[24, 48, 96, 192], bottleneck=384, dropout=0.08)


def convnext(**kwargs) -> nn.Module:
    return build_model("convnext_unet", len(CHANNELS), dropout=0.08, **kwargs)


def layer_scale_one() -> nn.Module:
    model = convnext()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma"):
                p.fill_(1.0)
    return model


def no_output_norm() -> nn.Module:
    model = convnext()
    model.head[0] = nn.Identity()
    return model


def groupnorm_blocks() -> nn.Module:
    model = convnext()
    for block in model.modules():
        if isinstance(block, ConvNeXtBlock):
            block.norm = nn.GroupNorm(8, block.dw.out_channels)
    return model


def dense_skip_merge() -> nn.Module:
    """The 1x1 convolutions that merge each skip are replaced by the U-Net's two dense 3x3 convolutions."""
    model = convnext()
    model.merge = nn.ModuleList(conv_block(2 * conv.out_channels, conv.out_channels) for conv in model.merge)
    return model


# name: (model builder, learning rate, clipping threshold, what the variant tests)
VARIANTS = {
    "unet_s11_tuned": (unet, LR, CLIP, "reference"),
    "convnext_as_trained": (convnext, LR, CLIP, "the S13 ConvNeXt U-Net"),
    "convnext_layer_scale_1": (layer_scale_one, LR, CLIP, "blocks start fully on instead of near the identity"),
    "convnext_no_output_norm": (no_output_norm, LR, CLIP, "no LayerNorm before the output layer"),
    "convnext_lr_x3": (convnext, 3 * LR, CLIP, "three times the learning rate"),
    "convnext_no_clipping": (convnext, LR, math.inf, "no gradient clipping"),
    "convnext_groupnorm": (groupnorm_blocks, LR, CLIP, "GroupNorm instead of LayerNorm inside the blocks"),
    "convnext_2_decoder_blocks": (lambda: convnext(up_depth=2), LR, CLIP, "two ConvNeXt blocks per decoder level"),
    "convnext_dense_skip_merge": (dense_skip_merge, LR, CLIP, "dense 3x3 convolutions to merge the skips"),
}


def run(make, lr: float, clip: float, batches: list, device: torch.device) -> dict:
    torch.manual_seed(0)
    model = make().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    amp = device.type != "cpu"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    losses, norms, t0 = [], [], time.time()
    model.train()
    for x, y, mask in batches:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(x)
        loss = masked_l1(pred, y, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)))
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "train_loss_first_100": float(np.mean(losses[:WINDOW])),
            "train_loss_last_100": float(np.mean(losses[-WINDOW:])),
            "median_grad_norm": float(np.median([n for n in norms if math.isfinite(n)])),
            "seconds": round(time.time() - t0)}


def main() -> None:
    device = pick_device()
    fit = fold_table("grouped").query("fold != 0").index.tolist()
    data = PatchDataset(fit, patch=PATCH, border=BORDER, mirror=True, seed=0, length=STEPS * BATCH)
    batches = list(DataLoader(data, batch_size=BATCH, shuffle=False, num_workers=0))
    rows = []
    for name, (make, lr, clip, tests) in VARIANTS.items():
        rows.append({"variant": name, "tests": tests, "device": device.type} | run(make, lr, clip, batches, device))
        print({k: rows[-1][k] for k in ("variant", "parameters", "train_loss_last_100", "median_grad_norm")}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "convnext_diagnosis.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
