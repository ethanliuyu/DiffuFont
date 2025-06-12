#!/usr/bin/env python3
"""
Convert glyph images inside the specified directories into an LMDB database.

Default paths
-------------
* Input images: `DataPreparation/Generated/ContentFont/`
* Output LMDB:  `DataPreparation/LMDB/ContentFont.lmdb`

Edit `INPUT_DIRS` and `LMDB_PATH` below to suit your needs.
"""

from pathlib import Path
import lmdb

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === configurable parameters ===
INPUT_DIRS = [  # one or more directories containing images to be written
    PROJECT_ROOT / "DataPreparation" / "Generated" / "FZSJ-ZHUZT",
]
LMDB_PATH = PROJECT_ROOT / "DataPreparation" / "LMDB" / "TrainFont.lmdb"   # LMDB output path
MAP_SIZE = 2 ** 30  # 1 GiB
BATCH_SIZE = 1000   # commit transaction every N images
# Skip existing keys (True) or overwrite (False)
DUPLICATE_SKIP = True
# ==================


def main():
    # collect all image files
    img_paths = []
    for d in INPUT_DIRS:
        img_paths.extend([p for p in Path(d).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    img_paths = sorted(img_paths)
    if not img_paths:
        print("[Warning] No images found – please check INPUT_DIRS")
        return

    # ensure LMDB parent directory exists
    LMDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(LMDB_PATH),
        map_size=MAP_SIZE,
        subdir=True,
        meminit=False,
        map_async=True,
    )

    txn = env.begin(write=True)
    for idx, img_path in enumerate(img_paths):
        with img_path.open("rb") as f:
            img_bytes = f.read()

        key = img_path.stem.encode("utf-8")

        if DUPLICATE_SKIP and txn.get(key) is not None:
            continue

        txn.put(key, img_bytes)

        if (idx + 1) % BATCH_SIZE == 0:
            txn.commit()
            print(f"Written {idx + 1}/{len(img_paths)} images …")
            txn = env.begin(write=True)

    # commit the remaining images
    txn.commit()
    print(f"Done: {len(img_paths)} images written to {LMDB_PATH}")

    env.sync()
    env.close()


if __name__ == "__main__":
    main()
