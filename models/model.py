#!/usr/bin/env python3
"""
Training utilities for FontDiffusionUNet.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from .font_diffusion_unet import FontDiffusionUNet
from .daca import DACA


class NoiseScheduler:
    """Creates beta / alpha_bar schedules and provides utilities for the forward (noise adding) process."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps
        self.register_schedule(beta_start, beta_end)

    def register_schedule(self, beta_start: float, beta_end: float):
        betas = torch.linspace(beta_start, beta_end, self.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1 - alpha_bars))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def to(self, device):
        for name in ["betas", "alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """Return noisy sample `x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε`."""
        device = x0.device
        t = t.long()
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1).to(device)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1).to(device)
        eps = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_1m * eps
        return x_t, eps


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss based on the first few layers of a pretrained VGG19 network."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16]  # till relu3_1
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_norm = self.norm(input)
        target_norm = self.norm(target)
        feat_in = self.vgg(input_norm)
        feat_tg = self.vgg(target_norm)
        return F.l1_loss(feat_in, feat_tg)


class DiffusionTrainer:
    def __init__(
        self,
        model: FontDiffusionUNet,
        device: torch.device,
        lr: float = 2e-4,
        lambda_mse: float = 1.0,
        lambda_off: float = 0.1,
        lambda_cp: float = 0.1,
        T: int = 1000,
        sample_every_steps: int | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # cosine LR schedule over total training steps (T epochs assumed)
        self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T)
        self.scheduler = NoiseScheduler(T).to(device)
        self.lambda_mse = lambda_mse
        self.lambda_off = lambda_off
        self.lambda_cp = lambda_cp
        self.perc_loss_fn = VGGPerceptualLoss().to(device)
        self.global_step = 0
        self.local_step = 0  # batch idx within current epoch
        self.current_epoch = 0

        # sampling config
        self.sample_every_steps = sample_every_steps  # if None → no automatic batch sampling
        self.sample_batch: Dict[str, torch.Tensor] | None = None
        self.sample_dir: Path | None = None

    def _offset_loss(self) -> torch.Tensor:
        off_losses: List[torch.Tensor] = []
        for m in self.model.modules():
            if isinstance(m, DACA) and hasattr(m, "last_offset_loss"):
                off_losses.append(m.last_offset_loss)
        if len(off_losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(off_losses).mean()

    def train_step(self, batch: Dict[str, torch.Tensor]):
        self.model.train()
        self._clear_offset()
        x0 = batch["target"].to(self.device)  # clean target image
        content = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        B = x0.size(0)
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device)

        # forward diffusion
        x_t, _ = self.scheduler.add_noise(x0, t)

        # prediction
        x0_hat = self.model(x_t, t, content, style)

        # losses
        loss_mse = F.mse_loss(x0_hat, x0)
        loss_off = self._offset_loss()
        loss_cp = self.perc_loss_fn(x0_hat, x0)

        loss = self.lambda_mse * loss_mse + self.lambda_off * loss_off + self.lambda_cp * loss_cp

        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.lr_schedule.step()

        self.global_step += 1
        self.local_step += 1

        # automatic sampling triggered by batch counter
        if (
            self.sample_every_steps
            and self.sample_batch is not None
            and self.sample_dir is not None
            and self.local_step % self.sample_every_steps == 0
        ):
            self.sample_and_save(self.sample_batch, self.sample_dir)

        return {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_off": loss_off.item(),
            "loss_cp": loss_cp.item(),
            "lr": self.lr_schedule.get_last_lr()[0],
        }

    # --------------------- sampling (DDIM简化) --------------------- #
    def ddim_sample(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        c: int = 10,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM 采样实现，默认使用 η=0 的确定性路径。

        Args:
            content_img: (B,C,H,W)
            style_img:   (B,C*k,H,W)
            c:           子序列线性间隔因子
            eta:         噪声比例，0 为确定性 DDIM
        """
        self.model.eval()

        B, C, H, W = content_img.shape
        device = self.device

        # 构造子序列 tau
        steps = max(2, self.scheduler.T // c)
        tau = torch.linspace(self.scheduler.T - 1, 0, steps).long()  # descending

        x_t = torch.randn(B, C, H, W, device=device)

        for i in range(len(tau) - 1):
            t_i = tau[i]
            t_prev = tau[i + 1]  # tau_{i-1}

            t_cond = torch.full((B,), t_i, device=device, dtype=torch.long)
            x0_hat = self.model(x_t, t_cond, content_img.to(device), style_img.to(device))

            alphabar_i = self.scheduler.alpha_bars[t_i].to(device)
            alphabar_prev = self.scheduler.alpha_bars[t_prev].to(device)

            a_i = torch.sqrt(alphabar_prev / alphabar_i)
            b_i = torch.sqrt(1 - alphabar_prev) - torch.sqrt((1 - alphabar_i) / alphabar_i)

            x_t = a_i * x0_hat + b_i * x_t

            if eta > 0 and i < len(tau) - 2:
                z = torch.randn_like(x_t)
                sigma = eta * torch.sqrt((1 - alphabar_prev) / (1 - alphabar_i) * (1 - alphabar_i / alphabar_prev))
                x_t += sigma * z

        return x0_hat.clamp(-1, 1)

    # --------------------- checkpoint --------------------- #
    def save(self, path: str | Path):
        chkpt = {
            "model_state": self.model.state_dict(),
            "opt_state": self.opt.state_dict(),
            "step": self.global_step,
        }
        torch.save(chkpt, path)

    def load(self, path: str | Path):
        chkpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chkpt["model_state"])
        self.opt.load_state_dict(chkpt["opt_state"])
        self.global_step = chkpt.get("step", 0)

    # ------------------ internal helpers ------------------ #
    def _clear_offset(self):
        for m in self.model.modules():
            if isinstance(m, DACA):
                m.last_offset_loss = torch.tensor(0.0, device=self.device)

    # --------------------- epoch / fit --------------------- #
    def train_epoch(self, dataloader, epoch: int, grad_clip: float | None = 1.0):
        logs: List[Dict] = []
        self.current_epoch = epoch
        self.local_step = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            out = self.train_step(batch)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            logs.append(out)

            # update progress bar metrics
            pbar.set_postfix({
                'loss': f"{out['loss']:.3f}",
                'mse': f"{out['loss_mse']:.3f}",
                'off': f"{out['loss_off']:.3f}",
                'cp': f"{out['loss_cp']:.3f}"
            })

        keys = logs[0].keys()
        return {k: sum(d[k] for d in logs) / len(logs) for k in keys}

    def fit(
        self,
        dataloader,
        epochs: int,
        save_every: int | None = None,
        save_dir: str | Path | None = None,
        sample_every: int | None = None,
        sample_callback=None,
    ):
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            log = self.train_epoch(dataloader, epoch)
            print(f"\nEpoch {epoch}:", {k: round(v,4) for k,v in log.items()})

            if save_every and epoch % save_every == 0 and save_dir is not None:
                self.save(save_dir / f"ckpt_{epoch}.pt")

            if sample_every and epoch % sample_every == 0 and sample_callback is not None:
                sample_callback(self)

    # --------------------- convenience sample & save --------------------- #
    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path, steps: int = 50):
        """使用固定 batch 采样并保存到 out_dir."""
        out_dir.mkdir(parents=True, exist_ok=True)
        content = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        sample = self.ddim_sample(content, style, steps)
        filename = f"sample_ep{self.current_epoch}_step{self.local_step}.png"
        save_image((sample + 1) / 2, out_dir / filename)
