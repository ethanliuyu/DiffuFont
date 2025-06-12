#!/usr/bin/env python3
"""
FontDiffusionUNet
=================

Implementation of the main network described in the paper.  The network
integrates the following modules:

1. **Content encoder**  – a stack of Conv-BN-ReLU blocks that extract content features.
2. **Style encoder**    – identical architecture for style feature extraction.
3. **U-Net backbone**   – DACA blocks are inserted in the down-sampling path, FGSA
   blocks in the up-sampling path.
4. **AdaLN**            – Adaptive LayerNorm injecting the diffusion timestep *t*.
5. **Sinusoidal timestep embedding** passed through an MLP.

Forward signature
-----------------
```python
out = model(
    x_t,         # noisy image  (B,C,H,W)
    t,           # timestep tensor (B,)
    content_img, # content image (B,C,H,W)
    style_img,   # reference style images (B,C*k,H,W)
)
# returns  \hat x_0   (B,C,H,W)
```
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.daca import DACA
from models.fgsa import FGSA


# ------------------------- 基础工具 ------------------------- #


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal timestep embeddings (same formulation as in the Transformer).

    Args:
        timesteps:  `(B,)` timestep indices
        dim:        embedding dimension

    Returns
    -------
    Tensor of shape `(B, dim)`
    """
    half_dim = dim // 2
    exponent = -math.log(10000.0) / (half_dim - 1)
    exponents = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * exponent)
    angles = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization.
    GroupNorm over spatial dimensions followed by a timestep-conditioned affine transform.
    """

    def __init__(self, channels: int, t_embed_dim: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(1, channels, eps=1e-6, affine=False)  # LayerNorm over (H, W)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, channels * 2),  # γ, β
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W), t_emb: (B, t_embed_dim)"""
        B, C, _, _ = x.shape
        h = self.norm(x)
        params = self.mlp(t_emb).view(B, 2, C, 1, 1)  # (B,2,C,1,1)
        gamma, beta = params[:, 0], params[:, 1]
        return gamma * h + beta + x  # 残差加入，以稳健训练


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ------------------------- ResNet Block ------------------------- #


class ResBlock(nn.Module):
    """2×Conv + (optional) AdaLN residual block"""

    def __init__(self, channels: int, t_embed_dim: int, use_adaln: bool = True):
        super().__init__()
        self.use_adaln = use_adaln
        self.conv1 = ConvBNReLU(channels, channels)
        if use_adaln:
            self.adaln1 = AdaLN(channels, t_embed_dim)
        self.conv2 = ConvBNReLU(channels, channels)
        if use_adaln:
            self.adaln2 = AdaLN(channels, t_embed_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        if self.use_adaln:
            x = self.adaln1(x, t_emb)
        x = self.conv2(x)
        if self.use_adaln:
            x = self.adaln2(x, t_emb)
        return x + residual


# Decoder-side ResNet block with FGSA + AdaLN
class DecoderResBlock(nn.Module):
    def __init__(self, channels: int, t_embed_dim: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.fgsa = FGSA(channels)
        self.adaln = AdaLN(channels, t_embed_dim)
        self.conv2 = ConvBNReLU(channels, channels)

    def forward(self, x: torch.Tensor, style_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.fgsa(x, style_feat)
        x = self.adaln(x, t_emb)
        x = self.conv2(x)
        return x + residual


# ------------------------- 编码器 ------------------------- #

class Encoder(nn.Module):
    """A simple Conv-BN-ReLU stack.  Each block downsamples with `stride=2`.  All
    intermediate feature maps are returned for skip connections.
    """

    def __init__(self, in_ch: int, base_ch: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            out_ch = base_ch * (2 ** i)
            layers.append(ConvBNReLU(ch, out_ch, stride=2))
            ch = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats  # list len==num_layers, 分辨率从 H/2 到 H/2^{n}


# ------------------------- U-Net Blocks ------------------------- #

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_embed_dim: int, with_daca: bool, use_adaln: bool):
        super().__init__()
        self.with_daca = with_daca
        self.use_adaln = use_adaln
        self.conv1 = ConvBNReLU(in_ch, out_ch, stride=2)
        if with_daca:
            self.daca = DACA(in_channels=out_ch)
        if use_adaln:
            self.adaln = AdaLN(out_ch, t_embed_dim)

    def forward(self, x: torch.Tensor, content_feat: torch.Tensor, t: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.with_daca:
            x, _, _ = self.daca(x, content_feat, t)
        if self.use_adaln:
            x = self.adaln(x, t_emb)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_embed_dim: int,
                 use_fgsa: bool, use_adaln: bool, need_upsample: bool):
        super().__init__()
        self.need_upsample = need_upsample
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.use_fgsa = use_fgsa
        self.use_adaln = use_adaln
        if use_fgsa:
            self.fgsa = FGSA(out_ch)
        if use_adaln:
            self.adaln = AdaLN(out_ch, t_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        style_feat: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # concat first (assumes same spatial size)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        if self.use_fgsa:
            x = self.fgsa(x, style_feat)
        if self.use_adaln:
            x = self.adaln(x, t_emb)
        # upsample for next scale if needed
        if self.need_upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return x


# ------------------------- Font Diffusion U-Net ------------------------- #

class FontDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        time_embed_dim: int = 256,
        style_k: int = 1,  # 参考风格字符数量
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.style_k = style_k
        # timestep embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Encoders for content and style (style 输入通道 = in_channels * style_k)
        self.content_encoder = Encoder(in_channels, base_channels, num_layers)
        self.style_encoder = Encoder(in_channels * style_k, base_channels, num_layers)

        # U-Net Down & Up
        down_blocks = []
        up_blocks = []
        ch = in_channels  # 直接从原始输入开始
        down_channels: List[int] = []  # record feature dims for skip connections
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            # 最后一层 down block 不使用 DACA/AdaLN
            is_last = i == num_layers - 1
            down_blocks.append(DownBlock(ch, out_ch, time_embed_dim, with_daca=not is_last, use_adaln=not is_last))
            down_channels.append(out_ch)
            ch = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)

        # 编码器底部 ResBlock，无 AdaLN
        self.bottom_resblock = ResBlock(ch, time_embed_dim, use_adaln=False)

        # decoder first ResNet layer with FGSA & AdaLN
        self.decoder_resblock = DecoderResBlock(ch, time_embed_dim)

        # build up blocks (reverse order)
        rev_channels = list(reversed(down_channels))
        in_ch = ch  # after decoder_resblock channels unchanged
        for i, skip_ch in enumerate(rev_channels):
            out_ch = skip_ch
            is_last = i == len(rev_channels) - 1
            up_blocks.append(UpBlock(in_ch + skip_ch, out_ch, time_embed_dim,
                                     use_fgsa=not is_last, use_adaln=not is_last,
                                     need_upsample=True))  # 最后一层也上采样
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)

        # final output conv
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    # -------------------- forward -------------------- #

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
    ) -> torch.Tensor:
        """推断一个时间步的 \hat{x}_0。
        Args:
            x_t:         (B,C,H,W) 噪声输入
            t:           (B,)       时间步
            content_img: (B,C,H,W) 内容图像
            style_img:   (B,C,H,W) 风格参考图像
        Returns:
            \hat{x}_0:   (B,C,H,W) 预测的干净图像
        """
        # 1) 生成时间步嵌入
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        # 2) 提取 content / style 特征
        content_feats = self.content_encoder(content_img)
        style_feats = self.style_encoder(style_img)

        # 3) U-Net encoder path (含 DACA)
        skips: List[torch.Tensor] = []
        h = x_t
        for i, down in enumerate(self.down_blocks):
            h = down(h, content_feats[i], t, t_emb)
            skips.append(h)

        # 4) bottom without AdaLN/DACA
        h = self.bottom_resblock(h, t_emb)

        # decoder first ResNet layer with FGSA & AdaLN
        style_feat_bottom = style_feats[-1]
        h = self.decoder_resblock(h, style_feat_bottom, t_emb)

        # 5) decoder path (含 FGSA)
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            style_feat = style_feats[-(i + 1)]
            h = up(h, skip, style_feat, t_emb)
        # 6) output
        x0_hat = self.out_conv(h)

        return x0_hat


# ------------------------- quick sanity test ------------------------- #

if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256
    K = 3  # 风格字符数量
    net = FontDiffusionUNet(in_channels=C, style_k=K)
    x_t = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    content = torch.randn(B, C, H, W)
    style = torch.randn(B, C*K, H, W)  # 3 张风格图像拼通道
    out = net(x_t, t, content, style)
    print("Output:", out.shape)  # Expected (B, C, H, W)
