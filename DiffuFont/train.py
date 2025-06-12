#!/usr/bin/env python3
"""Simple training script for FontDiffusionUNet.

Expected data structure (when *not* using FontImageDataset):

root/
    content/  xxx.png                # content glyphs
    style/    xxx_0.png, xxx_1.png   # reference style glyphs
    target/   xxx.png                # target glyphs of the training font

Files with the same base name belong to the same character.  `--style-k` determines
how many reference images are concatenated along the channel dimension.
When `FontImageDataset` is available (default in this project), the LMDB-based loader
is used instead and the folders above are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

from models.font_diffusion_unet import FontDiffusionUNet
from models.model import DiffusionTrainer

# Use the LMDB-based dataset provided by the project
from dataset import FontImageDataset


def collate_fn(samples, style_k: int):
    """Convert a list of samples returned by FontImageDataset to a batched dict."""
    # 将列表样本转换为 batch tensor
    contents, styles, targets = [], [], []
    for s in samples:
        contents.append(s["content"])
        targets.append(s["input"])  # take "input" image as training target

        style_imgs = s["styles"][:style_k]
        if len(style_imgs) < style_k:
            style_imgs = style_imgs * style_k  # repeat if not enough references
            style_imgs = style_imgs[:style_k]
        styles.append(torch.cat(style_imgs, dim=0))

    return {
        "content": torch.stack(contents),
        "style": torch.stack(styles),
        "target": torch.stack(targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--style-k", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    DATA_ROOT = Path("./")  # TODO: 根据实际数据集路径修改

    dataset = FontImageDataset(project_root=DATA_ROOT, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,  # lmdb Environment 不可被多进程 pickle
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, args.style_k),
    )

    model = FontDiffusionUNet(in_channels=3, style_k=args.style_k)
    trainer = DiffusionTrainer(model, device, lr=args.lr)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}")

    fixed_batch = next(iter(loader))
    out_dir = Path(args.save_dir) / "samples"

    # 配置按 batch 采样
    trainer.sample_every_steps = 3  # 每 100 个 batch 采样一次，可改
    trainer.sample_batch = fixed_batch
    trainer.sample_dir = out_dir

    trainer.fit(
        loader,
        epochs=args.epochs,
        save_every=5,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
