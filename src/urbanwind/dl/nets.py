"""
nets.py: the network architectures, and the registry every stage builds its models through.

Every layer that looks at neighbouring cells wraps around the domain edges (circular padding,
periodic upsampling), because the simulated city repeats like a tiled floor. A 4-level network
with this padding is exactly shift-equivariant for shifts that are multiples of 16 cells.

Public API
----------
UNet(in_ch, ...)                  : the S11 baseline
ResUNet, AttnUNet, FNO            : S12a: residual, attention-gated and Fourier-operator candidates
ConvNeXtUNet(in_ch, ...)          : S13: the same U-Net layout built from ConvNeXt blocks
register(name)                    : decorator that adds an architecture to the registry
build_model(name, in_ch, **kw)    : build any registered architecture
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

MODELS: dict[str, Callable[..., nn.Module]] = {}


def register(name: str) -> Callable:
    """Add an architecture to the registry so configs can name it (S12a adds its own this way)."""
    def add(cls):
        if name in MODELS:
            raise ValueError(f"model {name!r} is already registered")
        MODELS[name] = cls
        return cls
    return add


def build_model(name: str, in_ch: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](in_ch=in_ch, **kwargs)


def conv_block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by GroupNorm and ReLU, with circular padding."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
    )


def upsample_periodic(x: torch.Tensor) -> torch.Tensor:
    """Bilinear 2x upsampling that wraps around the edges.

    Plain bilinear upsampling clamps at the border; padding one cell circularly first and cropping
    afterwards makes the edge cells blend with their periodic neighbours instead.
    """
    up = F.interpolate(F.pad(x, (1, 1, 1, 1), mode="circular"), scale_factor=2, mode="bilinear", align_corners=False)
    return up[..., 2:-2, 2:-2]


@register("unet")
class UNet(nn.Module):
    """Periodic U-Net: 4 levels, GroupNorm, spatial dropout, and a softplus output so speeds stay positive."""

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (32, 64, 128, 256), bottleneck: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(conv_block(c, w))
            c = w
        self.mid = conv_block(c, bottleneck)
        self.up_conv, self.up_block = nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_conv.append(nn.Conv2d(c, w, 3, padding=1, padding_mode="circular"))
            self.up_block.append(conv_block(2 * w, w))
            c = w
        # spatial dropout drops whole feature maps; S14 keeps it on at prediction time for MC dropout
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, 1, 1)
        self.multiple = 2 ** len(widths)
        for m in self.modules():  # He initialisation, suited to ReLU layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(f"input sides {tuple(x.shape[-2:])} must be multiples of {self.multiple}")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.mid(x))
        last = len(self.up_block) - 1
        for i, (conv, block, skip) in enumerate(zip(self.up_conv, self.up_block, reversed(skips))):
            x = block(torch.cat([conv(upsample_periodic(x)), skip], dim=1))
            if i < last:  # no dropout right before the output layer, which only maps features to speed
                x = self.drop(x)
        return F.softplus(self.head(x))


class ResBlock(nn.Module):
    """Residual block: two circular 3x3 convolutions with GroupNorm and ReLU, plus 1x1 projection.

    The scale (gamma) of the final GroupNorm (gn2) is zero-initialised so that at the start of
    training the residual branch contributes nothing and each block behaves like an identity mapping.
    This is a standard technique (ReZero / FixUp family) that dramatically improves early gradient
    flow in deep residual networks and prevents the collapse-to-zero pathology observed without it.
    The He-init loop in ResUNet.__init__ only touches nn.Conv2d, so it does not overwrite this.
    """

    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular")
        self.gn1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular")
        self.gn2 = nn.GroupNorm(groups, cout)
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else None
        # Zero-initialise the scale of the last normalisation in the residual branch so the block
        # starts as a near-identity mapping.  This is applied here (not in ResUNet.__init__) so that
        # the invariant holds regardless of where ResBlock is used.
        nn.init.zeros_(self.gn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x) if self.proj is not None else x
        h = F.relu(self.gn1(self.conv1(x)), inplace=True)
        h = self.gn2(self.conv2(h))
        return F.relu(h + res, inplace=True)


@register("res_unet")
class ResUNet(nn.Module):
    """Periodic Res-U-Net: S11 layout with residual blocks and softplus output.

    Two-part initialisation strategy for stable early training:
    1. Each ResBlock zero-initialises its final GroupNorm scale (gn2.weight), so the residual
       branch contributes nothing at step 0 and each block starts as a near-identity mapping.
    2. The head (1x1 output conv) weight is scaled to 0.01× He-init. Without this, the stacked
       projection shortcuts (which change channel count and cannot be identity) produce initial
       outputs in the range [26, 57] on typical inputs, far above the target speed range (≈1 m/s).
       Two gradient steps from such a large initial loss drive the head weight strongly negative and
       the model collapses to predicting ≈0 (confirmed by the 30-step CPU overfit check). Scaling
       the head to 0.01 brings the initial output to the correct order of magnitude (≈0.5–1 m/s)
       and prevents this collapse. This is standard practice (BigGAN, FixUp) and does not conflict
       with the He-init loop because the loop runs first and the scaling is applied after.
    """

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (24, 48, 96, 192), bottleneck: int = 384,
                 dropout: float = 0.08):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(ResBlock(c, w))
            c = w
        self.mid = ResBlock(c, bottleneck)
        self.up_conv, self.up_block = nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_conv.append(nn.Conv2d(c, w, 3, padding=1, padding_mode="circular"))
            self.up_block.append(ResBlock(2 * w, w))
            c = w
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, 1, 1)
        self.multiple = 2 ** len(widths)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Scale head weight to 0.01× so initial predictions are O(1) not O(50).
        # Applied AFTER the He-init loop so it is not overwritten.
        self.head.weight.data.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(f"input sides {tuple(x.shape[-2:])} must be multiples of {self.multiple}")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.mid(x))
        last = len(self.up_block) - 1
        for i, (conv, block, skip) in enumerate(zip(self.up_conv, self.up_block, reversed(skips))):
            x = block(torch.cat([conv(upsample_periodic(x)), skip], dim=1))
            if i < last:
                x = self.drop(x)
        return F.softplus(self.head(x))


class AttentionGate(nn.Module):
    """Attention gate on skip connections (Oktay et al. 2018).

    Gating signal from the coarser level is brought to the skip's size with periodic upsampling.
    1x1 convolutions compute linear embeddings, followed by GroupNorm, ReLU, and sigmoid coefficient map.
    """

    def __init__(self, in_skip: int, in_g: int, int_ch: int, groups: int = 8):
        super().__init__()
        self.w_x = nn.Sequential(nn.Conv2d(in_skip, int_ch, 1), nn.GroupNorm(groups, int_ch))
        self.w_g = nn.Sequential(nn.Conv2d(in_g, int_ch, 1), nn.GroupNorm(groups, int_ch))
        self.psi = nn.Conv2d(int_ch, 1, 1)

    def forward(self, skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g_up = upsample_periodic(g)
        act = F.relu(self.w_x(skip) + self.w_g(g_up), inplace=True)
        coeff = torch.sigmoid(self.psi(act))
        return skip * coeff


@register("attn_unet")
class AttnUNet(nn.Module):
    """Periodic Attention U-Net: S11 layout with attention gates on skip connections."""

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (24, 48, 96, 192), bottleneck: int = 384,
                 dropout: float = 0.08):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(conv_block(c, w))
            c = w
        self.mid = conv_block(c, bottleneck)
        self.up_conv, self.up_block, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_conv.append(nn.Conv2d(c, w, 3, padding=1, padding_mode="circular"))
            self.gates.append(AttentionGate(in_skip=w, in_g=c, int_ch=w))
            self.up_block.append(conv_block(2 * w, w))
            c = w
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, 1, 1)
        self.multiple = 2 ** len(widths)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(f"input sides {tuple(x.shape[-2:])} must be multiples of {self.multiple}")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.mid(x))
        last = len(self.up_block) - 1
        for i, (conv, block, gate, skip) in enumerate(zip(self.up_conv, self.up_block, self.gates, reversed(skips))):
            gated_skip = gate(skip, x)
            x = block(torch.cat([conv(upsample_periodic(x)), gated_skip], dim=1))
            if i < last:
                x = self.drop(x)
        return F.softplus(self.head(x))


class SpectralBlock(nn.Module):
    """2D Spectral convolution block for Fourier Neural Operator (FNO).

    Runs 2D real FFT, retains low frequency modes, applies learned complex weights, and maps back
    with 2D inverse real FFT. The FFT and spectral multiplication execute in float32 with autocast
    disabled to ensure compatibility with mixed precision training and arbitrary spatial tile sizes.
    Adds a 1x1 convolution bypass followed by GELU activation.
    """

    def __init__(self, width: int, modes1: int = 16, modes2: int = 16):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (width * width)
        self.weights1 = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))
        self.bypass = nn.Conv2d(width, width, 1)

    def _apply(self, fn):
        super()._apply(fn)
        if self.bypass.weight.dtype == torch.float64:
            self.weights1.data = self.weights1.data.to(torch.cdouble)
            self.weights2.data = self.weights2.data.to(torch.cdouble)
        elif self.bypass.weight.dtype == torch.float32:
            self.weights1.data = self.weights1.data.to(torch.cfloat)
            self.weights2.data = self.weights2.data.to(torch.cfloat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        bypass = self.bypass(x)
        with torch.autocast(x.device.type, enabled=False):
            c_dtype = self.weights1.dtype
            r_dtype = torch.float64 if c_dtype == torch.cdouble else torch.float32
            x_in = x.to(dtype=r_dtype)
            x_ft = torch.fft.rfft2(x_in)
            out_ft = torch.zeros(b, self.width, h, w // 2 + 1, device=x.device, dtype=c_dtype)
            m1 = min(self.modes1, h // 2)
            m2 = min(self.modes2, w // 2)
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
            )
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
            )
            spec = torch.fft.irfft2(out_ft, s=(h, w)).to(dtype=bypass.dtype)
        return F.gelu(spec + bypass)


@register("fno")
class FNO(nn.Module):
    """Fourier Neural Operator (FNO): 1x1 lifting, 4 spectral layers with bypass, 1x1 projection.

    Modes are counted in grid cells (for example, mode 16 corresponds to a 64 m wave on a 1,024-cell
    domain but a 16 m wave on a 256-cell patch). To preserve physical wavelength correspondence,
    inference should use 256x256 wrap-around tiles matching training (infer_window: 256, infer_border: 16).
    Contains no positional or coordinate channels and no domain padding. Multiple is 1.
    """

    def __init__(self, in_ch: int = 5, width: int = 48, modes1: int = 16, modes2: int = 16, n_layers: int = 4,
                 dropout: float = 0.08):
        super().__init__()
        self.multiple = 1
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral_layers = nn.ModuleList([
            SpectralBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        self.drop = nn.Dropout2d(dropout)
        self.proj = nn.Conv2d(width, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        last = len(self.spectral_layers) - 1
        for i, layer in enumerate(self.spectral_layers):
            x = layer(x)
            if i < last:
                x = self.drop(x)
        return F.softplus(self.proj(x))


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of each cell separately, as in ConvNeXt.

    Unlike GroupNorm it never averages over space, so a cell's output does not depend on how large
    the patch or the domain is.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # statistics in float32 even under mixed precision; written out rather than permuting to
        # channels-last for F.layer_norm, which copies the whole tensor twice
        x32 = x.float()
        mean = x32.mean(dim=1, keepdim=True)
        var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
        out = (x32 - mean) * torch.rsqrt(var + self.eps)
        return (out * self.weight[:, None, None] + self.bias[:, None, None]).to(x.dtype)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al. 2022): a 7x7 depthwise convolution with circular padding, LayerNorm,
    a 1x1 expansion to 4x the channels with GELU, a 1x1 projection back, and a residual connection.

    The 7x7 filter sees a wider neighbourhood than the U-Net's 3x3 ones at every level. The layer
    scale starts small, so each block begins close to the identity.
    """

    def __init__(self, channels: int, kernel: int = 7, expansion: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels, padding_mode="circular")
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, expansion * channels, 1)
        self.pw2 = nn.Conv2d(expansion * channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


def convnext_stage(channels: int, depth: int) -> nn.Sequential:
    return nn.Sequential(*(ConvNeXtBlock(channels) for _ in range(depth)))


@register("convnext_unet")
class ConvNeXtUNet(nn.Module):
    """Periodic U-Net built from ConvNeXt blocks, with the S11 U-Net's layout and output.

    - Encoder: a 3x3 circular stem, then at each level ConvNeXt blocks, a skip connection, and a
      LayerNorm + 2x2 stride-2 convolution that halves the grid (no padding needed).
    - Decoder: periodic bilinear upsampling, a 1x1 convolution to the level's width, concatenation
      with the skip, a 1x1 convolution that merges them, and ConvNeXt blocks.
    - Output: LayerNorm, 1x1 convolution and softplus, so speeds stay positive. Spatial dropout
      sits where the U-Net has it, so MC dropout works the same way.
    No positional or coordinate input, and every layer that looks at neighbours wraps around.
    """

    def __init__(self, in_ch: int = 5, widths: tuple[int, ...] = (32, 64, 128, 256), bottleneck: int = 384,
                 depths: tuple[int, ...] = (2, 2, 2, 2), mid_depth: int = 2, up_depth: int = 1,
                 dropout: float = 0.08):
        super().__init__()
        if len(depths) != len(widths):
            raise ValueError("depths needs one entry per level")
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1, padding_mode="circular")
        self.down_blocks, self.downsample = nn.ModuleList(), nn.ModuleList()
        nxt = [*widths[1:], bottleneck]
        for w, d, w_next in zip(widths, depths, nxt):
            self.down_blocks.append(convnext_stage(w, d))
            self.downsample.append(nn.Sequential(LayerNorm2d(w), nn.Conv2d(w, w_next, 2, stride=2)))
        self.mid = convnext_stage(bottleneck, mid_depth)
        self.up_proj, self.merge, self.up_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up_proj.append(nn.Conv2d(c, w, 1))
            self.merge.append(nn.Conv2d(2 * w, w, 1))
            self.up_blocks.append(convnext_stage(w, up_depth))
            c = w
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Sequential(LayerNorm2d(c), nn.Conv2d(c, 1, 1))
        self.multiple = 2 ** len(widths)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(f"input sides {tuple(x.shape[-2:])} must be multiples of {self.multiple}")
        x, skips = self.stem(x), []
        for blocks, down in zip(self.down_blocks, self.downsample):
            x = blocks(x)
            skips.append(x)
            x = down(x)
        x = self.drop(self.mid(x))
        last = len(self.up_blocks) - 1
        for i, (proj, merge, blocks, skip) in enumerate(zip(self.up_proj, self.merge, self.up_blocks, reversed(skips))):
            x = blocks(merge(torch.cat([proj(upsample_periodic(x)), skip], dim=1)))
            if i < last:
                x = self.drop(x)
        return F.softplus(self.head(x))
