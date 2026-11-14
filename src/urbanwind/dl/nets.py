"""U-Net model for urban wind field prediction.

The S11 U-Net architecture:
- 4 encoder levels with channels [24, 48, 96, 192]
- Bottleneck: 384 channels
- 2 Conv3x3 + GroupNorm + ReLU per block
- Bilinear upsampling with convolution for upsampling
- Circular padding (domains are periodic)
- Softplus output (keeps speed non-negative)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two Conv3x3 + GroupNorm + ReLU layers."""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gnorm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gnorm2 = nn.GroupNorm(groups, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gnorm2(x)
        x = self.relu(x)
        return x


class Down(nn.Module):
    """MaxPool then ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class Up(nn.Module):
    """Bilinear upsampling followed by ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed to match x2 size
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    """Periodic U-Net for urban wind field prediction.

    Widths: [24, 48, 96, 192], bottleneck 384.
    Circular padding maintains periodicity across domain edges.
    Softplus output keeps U/u* non-negative.
    """

    def __init__(
        self,
        in_ch=1,
        widths=(24, 48, 96, 192),
        bottleneck=384,
        groups=8,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.widths = tuple(widths)
        self.bottleneck = bottleneck

        # Encoder
        self.enc0 = ConvBlock(in_ch, widths[0], groups)
        self.down0 = Down(widths[0], widths[1])
        self.enc1 = ConvBlock(widths[1], widths[1], groups)
        self.down1 = Down(widths[1], widths[2])
        self.enc2 = ConvBlock(widths[2], widths[2], groups)
        self.down2 = Down(widths[2], widths[3])
        self.enc3 = ConvBlock(widths[3], widths[3], groups)
        self.down3 = Down(widths[3], bottleneck)

        # Bottleneck
        self.enc4 = ConvBlock(bottleneck, bottleneck, groups)

        # Decoder
        self.up0 = Up(bottleneck + widths[3], widths[3])
        self.enc4_ = ConvBlock(widths[3], widths[3], groups)
        self.up1 = Up(widths[3] + widths[2], widths[2])
        self.enc5 = ConvBlock(widths[2], widths[2], groups)
        self.up2 = Up(widths[2] + widths[1], widths[1])
        self.enc6 = ConvBlock(widths[1], widths[1], groups)
        self.up3 = Up(widths[1] + widths[0], widths[0])
        self.enc7 = ConvBlock(widths[0], widths[0], groups)
        self.final = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc3(self.down3(e3))

        # Bottleneck
        b = self.enc4(e4)

        # Decoder
        d0 = self.up0(b, e3)
        d0 = self.enc4_(d0)
        d1 = self.up1(d0, e2)
        d1 = self.enc5(d1)
        d2 = self.up2(d1, e1)
        d2 = self.enc6(d2)
        d3 = self.up3(d2, e0)
        d3 = self.enc7(d3)
        out = self.final(d3)
        out = self.softplus(out)
        return out
