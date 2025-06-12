# Few-Shot Font Generation via Denoising Diffusion and Component-Level Fine-Grained Style

## Key Components

* `models/font_diffusion_unet.py` – U-Net generator containing DACA / FGSA / AdaLN modules.
* `models/model.py` – training utilities (losses, DDPM training, DDIM sampling, automatic image logging).
* `train.py` – single-GPU training script.
* `dataset.py` – `FontImageDataset` loader used by the project.

## 1. Installation
```bash
# Python ≥ 3.8 is recommended
conda create -n DiffFont python=3.9 -y
conda activate DiffFont

# Core dependencies (choose the CUDA / CPU build that matches your system)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install tqdm lmdb pillow
```
A CUDA-enabled GPU is strongly recommended for training.

## 2. Data Preparation
The directory layout expected by `dataset.py` is shown below.
```text
ProjectRoot/
├── CharacterData/
│   ├── CharList.json          # list of content characters
│   └── mapping.json           # mapping: content char → list[style chars]
├── DataPreparation/
│   ├── FontList.json          # list of candidate .ttf files
│   └── LMDB/
│       ├── ContentFont.lmdb   # LMDB for the content font
│       └── TrainFont.lmdb     # LMDB for all style / target fonts
└── ...
```
### 2.1 Image Generation & LMDB Construction
Two helper scripts are provided:
* **`generate_font_images.py`** – render characters of a given TrueType font to individual PNG files.
* **`images_to_lmdb.py`** – convert a folder of PNGs into an LMDB database.

Typical workflow
1. **Generate content-font images**
   ```bash
   # Render only the content font (e.g. SourceHanSans)
   python generate_font_images.py \
       --font-path path/to/SourceHanSans.ttf \
       --charset CharacterData/CharList.json \
       --out-dir DataPreparation/Generated/ContentFont

   # Convert to LMDB
   python images_to_lmdb.py \
       --img-root DataPreparation/Generated/ContentFont \
       --lmdb-path DataPreparation/LMDB/ContentFont.lmdb
   ```
2. **Generate style / target font images**
   ```bash
   # Render every font listed in FontList.json
   python generate_font_images.py \
       --font-list DataPreparation/FontList.json \
       --charset CharacterData/CharList.json \
       --out-dir DataPreparation/Generated/TrainFonts

   # Convert to a single LMDB (sub-directories are supported)
   python images_to_lmdb.py \
       --img-root DataPreparation/Generated/TrainFonts \
       --lmdb-path DataPreparation/LMDB/TrainFont.lmdb
   ```
> The default filename pattern is `FontName@<char>.png`; `images_to_lmdb.py` relies on this convention.

### 2.2 Font Files
The **300** Chinese font files used in our experiments can be downloaded from Google Drive:
<https://drive.google.com/file/d/17rajeJz53RnCOEv9B4X6tDKaIwpz2bIb/view?usp=drive_link>

After downloading, extract the archive to `DataPreparation/Fonts/` and list the extracted `.ttf` paths in `DataPreparation/FontList.json`, then follow Section 2.1 to render the images.

### 2.3 Content–Style Reference Mapping

decomposition.json  is the character structure decomposition table.  The first thing you need to do is to construct a content-style reference mapping table. 

```bash
{content1: [ref1, ref2, ref3, ...],content2: [ref1, ref2, ref3, ...],...}
```

## 3. Training
```bash
python train.py \
  --epochs 50 \
  --batch 1 \
  --style-k 3 \
  --lr 2e-4 \
  --save-dir checkpoints
```
Edit `DATA_ROOT` at the top of `train.py` to point to your data directory.

During training
* A checkpoint is saved every **5** epochs in `checkpoints/ckpt_*.pt`.
* A DDIM sample is saved every **100** mini-batches in `checkpoints/samples/` with filename `sample_ep{epoch}_step{batch}.png`.

Resume training
```bash
python train.py --resume checkpoints/ckpt_10.pt
```

## 4. Inference / Sampling
```python
import torch
from PIL import Image
from torchvision import transforms as T
from models.font_diffusion_unet import FontDiffusionUNet
from models.model import DiffusionTrainer

net = FontDiffusionUNet(in_channels=3, style_k=3)
trainer = DiffusionTrainer(net, torch.device('cuda'), sample_every_steps=None)
trainer.load('checkpoints/ckpt_50.pt')

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])
content_img = transform(Image.open('content.png')).unsqueeze(0)
style_imgs = [transform(Image.open(f'style_{i}.png')) for i in range(3)]
style_img = torch.cat(style_imgs, dim=0).unsqueeze(0)

sample = trainer.ddim_sample(content_img, style_img, c=10, eta=0)
T.functional.to_pil_image((sample[0] + 1) / 2).save('result.png')
```

---
