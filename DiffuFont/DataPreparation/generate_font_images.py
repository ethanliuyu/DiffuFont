#!/usr/bin/env python3
"""
Generate 256×256 monochrome glyph images for every (font, character) pair, based on:

* Character list  : `CharacterData/CharList.json`
* Font list       : `DataPreparation/FontList.json`

File-name pattern
-----------------
<font-file-name>@<character>.png  (the character is kept as-is without escaping)

Command-line flags (all optional)
--------------------------------
1. --char_size   (default: 240)   – glyph pixel size
2. --canvas_size (default: 256)   – canvas size
3. --x_offset    (default: 0)     – horizontal offset
4. --y_offset    (default: 0)     – vertical offset
5. --out_dir     (default: DataPreparation/Generated)

Example
~~~~~~~
python generate_font_images.py --char_size 200 --x_offset 10 --y_offset -20
"""

import json
import os
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

# Constant paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root folder
CHAR_LIST_PATH = PROJECT_ROOT / "CharacterData" / "CharList.json"
FONT_LIST_PATH = PROJECT_ROOT / "DataPreparation" / "FontList.json"
FONT_DIR = PROJECT_ROOT / "DataPreparation" / "Font"

# === configurable parameters ===
CHAR_SIZE = 240        # glyph pixel size
CANVAS_SIZE = 256      # canvas size
X_OFFSET = 0           # horizontal offset
Y_OFFSET = 0           # vertical offset
# Font indices to process:
#   None  -> all fonts
#   int   -> a single font
#   list[int] -> multiple fonts
FONT_INDEX = 1

OUT_DIR = None          # output directory; None = default path
# ==================


def load_json_list(path: Path) -> List[str]:
    """Load a JSON list file and return the contained list."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def draw_char(
    ch: str,
    font_path: Path,
    char_size: int,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
) -> Image.Image:
    """Render a single character with the specified font and return a PIL.Image"""
    # 加载字体
    font = ImageFont.truetype(str(font_path), size=char_size)

    # 创建白底画布
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 绘制字符（黑色）
    draw.text((x_offset, y_offset), ch, fill=(0, 0, 0), font=font)

    return img


def char_supported(ch: str, font_path: Path) -> bool:
    """Check whether the font file supports the given character."""
    try:
        font = ImageFont.truetype(str(font_path), size=10)
        return font.getmask(ch).getbbox() is not None
    except Exception:
        return False


def generate_images(
    char_list: List[str],
    font_list: List[str],
    char_size: int,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
    out_dir: Path,
):
    # make sure the output root directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    for font_name in font_list:
        font_path = FONT_DIR / font_name
        if not font_path.exists():
            print(f"[Warning] Font file not found: {font_path}")
            continue

        # create a sub-folder (font name without extension) for each font
        font_out_dir = out_dir / font_path.stem
        font_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing font: {font_name} -> saving to {font_out_dir.relative_to(PROJECT_ROOT)}")

        for ch in char_list:
            if not char_supported(ch, font_path):
                print(f"  [skip] Font does not support character: {ch}")
                continue

            img = draw_char(ch, font_path, char_size, canvas_size, x_offset, y_offset)

            # file-name pattern: <fontName>@<character>.png
            char_safe = ch  # 如有兼容性问题可替换为 f"U+{ord(ch):04X}"
            filename = f"{font_path.stem}@{char_safe}.png"
            save_path = font_out_dir / filename

            # 保存图片
            img.save(save_path, dpi=(300, 300))
            print(f"    Saved: {save_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    # 如果 OUT_DIR 未手动指定，则使用默认目录
    effective_out_dir = OUT_DIR or PROJECT_ROOT / "DataPreparation" / "Generated"

    chars = load_json_list(CHAR_LIST_PATH)
    all_fonts = load_json_list(FONT_LIST_PATH)

    # 根据 FONT_INDEX 筛选字体
    if FONT_INDEX is None:
        selected_fonts = all_fonts
    elif isinstance(FONT_INDEX, int):
        try:
            selected_fonts = [all_fonts[FONT_INDEX]]
        except IndexError:
            raise ValueError(f"FONT_INDEX {FONT_INDEX} is out of range of FontList.json")
    else:  # 假定为可迭代
        selected_fonts = []
        for idx in FONT_INDEX:
            try:
                selected_fonts.append(all_fonts[idx])
            except IndexError:
                raise ValueError(f"FONT_INDEX {idx} is out of range of FontList.json")

    generate_images(
        char_list=chars,
        font_list=selected_fonts,
        char_size=CHAR_SIZE,
        canvas_size=CANVAS_SIZE,
        x_offset=X_OFFSET,
        y_offset=Y_OFFSET,
        out_dir=Path(effective_out_dir),
    )
