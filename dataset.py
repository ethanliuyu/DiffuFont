#!/usr/bin/env python3
"""
FontImageDataset
================
Dataset that returns a content glyph, the corresponding glyph of a training font
(*input image*), and a list of style reference glyphs that share the same
structure/components with the content glyph.

Data sources
+------------
* **CharacterData/CharList.json**   : list of content characters **A**
* **CharacterData/mapping.json**    : mapping **A → [C, D, E, …]** of style characters
* **DataPreparation/FontList.json** : list of available *.ttf* fonts – the selected
  *font_index* denotes training font **B**
* **DataPreparation/LMDB/ContentFont.lmdb** : key `ContentFont@<char>`
* **DataPreparation/LMDB/TrainFont.lmdb**   : key `<FontName>@<char>`

Returned sample
```
{
    "char":   str,                    # character code
    "content": Image/Tensor,          # glyph rendered in ContentFont
    "input":   Image/Tensor,          # same char rendered in font B (training target)
    "styles":  List[Image/Tensor]     # reference glyphs from font B
}
```
If *transform* is provided, it is applied to **every** image.
"""

from pathlib import Path
import json
import io
from typing import List, Dict, Any, Optional, Union

import lmdb
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Fallback stub when torch is not installed.
    class Dataset:  # type: ignore
        pass


class FontImageDataset(Dataset):
    def __init__(
        self,
        project_root: Union[str, Path] = ".",
        font_index: int = 0,
        content_lmdb: Optional[Union[str, Path]] = None,
        train_lmdb: Optional[Union[str, Path]] = None,
        transform=None,
    ) -> None:
        self.root = Path(project_root).resolve()

        # Paths
        char_list_path = self.root / "CharacterData" / "CharList.json"
        mapping_path = self.root / "CharacterData" / "mapping.json"
        font_list_path = self.root / "DataPreparation" / "FontList.json"

        # Load JSON files
        self.char_list: List[str] = json.load(char_list_path.open("r", encoding="utf-8"))
        self.mapping: Dict[str, List[str]] = json.load(mapping_path.open("r", encoding="utf-8"))
        font_list: List[str] = json.load(font_list_path.open("r", encoding="utf-8"))

        if font_index < 0 or font_index >= len(font_list):
            raise IndexError(f"font_index {font_index} out of range of FontList.json")
        self.font_name = Path(font_list[font_index]).stem  # remove .ttf extension

        # LMDB paths
        if content_lmdb is None:
            content_lmdb = self.root / "DataPreparation" / "LMDB" / "ContentFont.lmdb"
        if train_lmdb is None:
            train_lmdb = self.root / "DataPreparation" / "LMDB" / "TrainFont.lmdb"

        # Open LMDB in read-only mode (lock=False to allow multi-process readers)
        self.content_env = lmdb.open(str(content_lmdb), readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(str(train_lmdb), readonly=True, lock=False, readahead=False)

        # Optional transform function
        self.transform = transform

        # Pre-scan to build a list of valid indices where every required key exists
        self.valid_indices: List[int] = []
        with self.content_env.begin() as c_txn, self.train_env.begin() as t_txn:
            for idx, ch in enumerate(self.char_list):
                # content glyph key
                content_key = f"ContentFont@{ch}".encode("utf-8")
                if c_txn.get(content_key) is None:
                    continue

                # same character rendered in training font
                input_key = f"{self.font_name}@{ch}".encode("utf-8")
                if t_txn.get(input_key) is None:
                    continue

                # style reference keys
                style_chars = self.mapping.get(ch, [])
                if not style_chars:
                    continue
                missing_style = False
                for sc in style_chars:
                    style_key = f"{self.font_name}@{sc}".encode("utf-8")
                    if t_txn.get(style_key) is None:
                        missing_style = True
                        break
                if missing_style:
                    continue

                # all keys exist → keep this index
                self.valid_indices.append(idx)

        if not self.valid_indices:
            raise RuntimeError("No valid samples found – please check LMDB integrity and paths.")

    def __len__(self):
        return len(self.valid_indices)

    def _bytes_to_img(self, b: bytes):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return self.transform(img) if self.transform else img

    def __getitem__(self, index: int):
        real_idx = self.valid_indices[index]
        ch = self.char_list[real_idx]

        content_key = f"ContentFont@{ch}".encode("utf-8")
        input_key = f"{self.font_name}@{ch}".encode("utf-8")
        style_chars = self.mapping.get(ch, [])

        with self.content_env.begin() as c_txn, self.train_env.begin() as t_txn:
            content_bytes = c_txn.get(content_key)
            input_bytes = t_txn.get(input_key)
            if content_bytes is None or input_bytes is None:
                raise KeyError("Missing required glyph images")

            style_imgs = []
            for sc in style_chars:
                style_key = f"{self.font_name}@{sc}".encode("utf-8")
                sb = t_txn.get(style_key)
                if sb is None:
                    raise KeyError(f"Missing style glyph: {sc}")
                style_imgs.append(self._bytes_to_img(sb))

        sample: Dict[str, Any] = {
            "char": ch,
            "content": self._bytes_to_img(content_bytes),
            "input": self._bytes_to_img(input_bytes),
            "styles": style_imgs,
        }
        return sample

    def close(self):
        if hasattr(self, "content_env"):
            self.content_env.close()
        if hasattr(self, "train_env"):
            self.train_env.close()

    def __del__(self):
        self.close()


# Quick sanity test
if __name__ == "__main__":
    ds = FontImageDataset()
    print(f"Valid samples: {len(ds)}")
    sample = ds[0]
    print(sample["char"], len(sample["styles"]))

    # Visualize loaded images (uncomment to view)
