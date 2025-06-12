#!/usr/bin/env python3
"""
Fine-Grained Style Aggregation (FGSA)
=====================================

Implementation of the FGSA block described in the paper.

Processing steps
----------------
1. **Input**
   * `content_feats` `(B, C, H, W)` – content feature map (query)
   * `style_feats`   `(B, C, H, W)` – style feature map  (key / value)
     If multiple style reference glyphs are available you may aggregate them
     beforehand (e.g. averaging) to the same shape.
2. Interleave the two tensors along the channel dimension →
   `F_cs ∈ ℝ^{B×2C×H×W}`.
3. Apply a **1×1 grouped convolution** (`groups=C`, 2→1 per group) to obtain
   `F_a ∈ ℝ^{B×C×H×W}`.
4. Pass through a **7×7 depth-wise separable convolution** (DW + PW) followed by
   `sigmoid` to get the attention map `A ∈ ℝ^{B×C×H×W}`.
5. Element-wise multiply: `V' = A ⊗ style_feats`.
6. Residual add: `F_r = content_feats + V'` and return.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to have symmetric padding"
        padding = kernel_size // 2

        # 1×1 分组卷积：输入 2C，输出 C，groups=C（每组2个通道→1）
        self.group_conv = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=1,
            groups=channels,
            bias=True,
        )

        # Depthwise Separable Conv
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # depthwise
            bias=True,
        )
        self.pw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _interleave_channels(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Interleave two tensors of shape `(B,C,H,W)` into one `(B,2C,H,W)` along the channel dimension."""
        B, C, H, W = a.shape
        # stack -> (B, C, 2, H, W)
        stack = torch.stack([a, b], dim=2)
        # reshape to (B, 2C, H, W) with interleaved channels
        interleaved = stack.view(B, 2 * C, H, W)
        return interleaved

    def forward(self, content_feats: torch.Tensor, style_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            content_feats: (B, C, H, W) Content feature map (Query)
            style_feats:   (B, C, H, W) Style feature map  (Key/Value)

        Returns:
            out_feats: (B, C, H, W) Aggregated features F_r
        """
        # 1. interleave content & style features
        f_cs = self._interleave_channels(content_feats, style_feats)

        # 2. grouped 1×1 convolution
        f_a = self.group_conv(f_cs)

        # 3. depth-wise separable convolution (DW + PW)
        x = self.dw_conv(f_a)
        x = self.pw_conv(x)

        # 4. sigmoid → attention map A
        attn = self.sigmoid(x)

        # 5. weight style features
        fused = attn * style_feats

        # 6. residual addition
        out = fused + content_feats
        return out


if __name__ == "__main__":
    # quick sanity test
    B, C, H, W = 2, 64, 32, 32
    fgsa = FGSA(C)
    q = torch.randn(B, C, H, W)
    k = torch.randn(B, C, H, W)
    out = fgsa(q, k)
    print(out.shape)  # Expected (2, 64, 32, 32) 