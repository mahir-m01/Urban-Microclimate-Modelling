"""
param_counts.py: Count and compare parameters of S12 candidate architectures against S11 U-Net baseline.

Baseline:
    Tuned S11 U-Net: 4,860,313 parameters (widths [24, 48, 96, 192], bottleneck 384)
Equal budget rule:
    Each model should be within ±25% of baseline (3,645,235 to 6,075,391 parameters).

Usage:
    uv run python pipeline/S12_architecture_search/param_counts.py
"""

from __future__ import annotations

import pandas as pd

from urbanwind.dl.nets import UNet, build_model

BASELINE_PARAMS = 4_860_313
MODELS = {
    "s11_unet_tuned (baseline)": lambda: UNet(5, widths=(24, 48, 96, 192), bottleneck=384, dropout=0.08),
    "res_unet": lambda: build_model("res_unet", 5, widths=(24, 48, 96, 192), bottleneck=384, dropout=0.08),
    "attn_unet": lambda: build_model("attn_unet", 5, widths=(24, 48, 96, 192), bottleneck=384, dropout=0.08),
    "fno": lambda: build_model("fno", 5, width=48, modes1=16, modes2=16, n_layers=4, dropout=0.08),
}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_param_table() -> pd.DataFrame:
    rows = []
    lo, hi = 0.75 * BASELINE_PARAMS, 1.25 * BASELINE_PARAMS
    for name, builder in MODELS.items():
        m = builder()
        params = count_parameters(m)
        ratio = params / BASELINE_PARAMS
        delta = params - BASELINE_PARAMS
        within_25pct = (lo <= params <= hi)
        rows.append({
            "model": name,
            "parameters": params,
            "delta_vs_baseline": delta,
            "ratio_vs_baseline": f"{ratio:.2%}",
            "multiple": getattr(m, "multiple", 1),
            "within_25pct": within_25pct,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = get_param_table()
    print("=" * 80)
    print("S12 ARCHITECTURE SEARCH: PARAMETER COUNT AUDIT (EQUAL BUDGET VERIFICATION)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Target window (±25%): {int(0.75 * BASELINE_PARAMS):,} to {int(1.25 * BASELINE_PARAMS):,} parameters\n")


if __name__ == "__main__":
    main()

