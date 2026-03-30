# model.py
# Custom U-Net CNN built completely from scratch (no pretrained weights)
# Architecture: Encoder (downsampling) + Bottleneck + Decoder (upsampling)
# Output: 2-class segmentation mask (Drivable=1, Non-Drivable=0)

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Basic building block: two Conv layers in a row
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    """
    Two convolution layers, each followed by BatchNorm and ReLU.
    This is the core repeated unit of U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),   # stabilises training
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# Encoder block: DoubleConv → MaxPool (halves resolution)
# ─────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)      # save this for skip connection
        pooled   = self.pool(features)
        return features, pooled      # return both


# ─────────────────────────────────────────────
# Decoder block: Upsample → concat skip → DoubleConv
# ─────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bilinear upsampling doubles the spatial resolution
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat with skip, channels are doubled → halve them
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch (can happen with odd dimensions)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)   # concatenate on channel axis
        return self.conv(x)


# ─────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net with 4 encoder levels, a bottleneck, and 4 decoder levels.
    Input:  (batch, 3, H, W)   — RGB image
    Output: (batch, num_classes, H, W)  — per-pixel class scores
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # Encoder (going down, doubling channels)
        self.enc1 = EncoderBlock(in_channels, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        # Bottleneck (deepest representation)
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (going up, halving channels)
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)

        # Final 1×1 conv: map to class scores
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # ── Encoder ──
        s1, x = self.enc1(x)   # skip1: (B, 32, H,   W)
        s2, x = self.enc2(x)   # skip2: (B, 64, H/2, W/2)
        s3, x = self.enc3(x)   # skip3: (B,128, H/4, W/4)
        s4, x = self.enc4(x)   # skip4: (B,256, H/8, W/8)

        # ── Bottleneck ──
        x = self.bottleneck(x) # (B, 512, H/16, W/16)

        # ── Decoder ──
        x = self.dec4(x, s4)   # (B, 256, H/8,  W/8)
        x = self.dec3(x, s3)   # (B, 128, H/4,  W/4)
        x = self.dec2(x, s2)   # (B,  64, H/2,  W/2)
        x = self.dec1(x, s1)   # (B,  32, H,    W)

        return self.final_conv(x)  # (B, num_classes, H, W)


# ─────────────────────────────────────────────
# Quick test — run this file directly to verify
# ─────────────────────────────────────────────
if __name__ == '__main__':
    model = UNet(in_channels=3, num_classes=2)
    dummy = torch.randn(2, 3, 256, 512)   # batch=2, RGB, 256x512
    out   = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")   # should be (2, 2, 256, 512)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
