"""
prerender_cache.py

Renders every row of a metadata CSV (char, codepoint, pinyin, tone) to an
image ONCE and saves everything into a single .pt file: a stacked image
tensor plus integer label tensors plus the label vocabularies. Loading this
cache at training time avoids re-rasterizing glyphs on every epoch.

Usage:
    python prerender_cache.py --csv data/labels_simplified.csv  --out data/cache_simplified.pt
    python prerender_cache.py --csv data/labels_traditional.csv --out data/cache_traditional.pt

Cache contents (torch.save'd dict):
    images         : uint8 tensor [N, 1, H, W]   (0-255, 0=black ... 255=white)
    tone_labels    : int64 tensor [N]
    pinyin_labels  : int64 tensor [N]
    tone_classes   : list[int]      index -> tone value (1-5)
    pinyin_classes : list[str]      index -> pinyin syllable string
    chars          : list[str]      chars[i] is the source character for row i
    image_size     : int
    font_path      : str
"""

import argparse
import time
from pathlib import Path

import torch

from chinese_char_dataset import ChineseCharDataset, render_char

DATA_DIR = Path(__file__).resolve().parent / "data"


def build_cache(csv_path, image_size, font_path, invert=False):
    ds = ChineseCharDataset(csv_path, image_size=image_size, font_paths=[font_path])
    n = len(ds)
    images = torch.empty((n, 1, image_size, image_size), dtype=torch.uint8)
    tone_labels = torch.empty(n, dtype=torch.int64)
    pinyin_labels = torch.empty(n, dtype=torch.int64)
    chars = []

    t0 = time.time()
    for i in range(n):
        row = ds.rows[i]
        img = render_char(row["char"], font_path, image_size=image_size, invert=invert)
        images[i, 0] = torch.from_numpy(
            __import__("numpy").array(img, dtype="uint8")
        )
        tone_labels[i] = ds.tone_to_idx[row["tone"]]
        pinyin_labels[i] = ds.pinyin_to_idx[row["pinyin"]]
        chars.append(row["char"])
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  rendered {i + 1}/{n} ({elapsed:.1f}s)")

    return {
        "images": images,
        "tone_labels": tone_labels,
        "pinyin_labels": pinyin_labels,
        "tone_classes": ds.tone_classes,
        "pinyin_classes": ds.pinyin_classes,
        "chars": chars,
        "image_size": image_size,
        "font_path": font_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DATA_DIR / "labels_simplified.csv"), help="Input metadata CSV")
    parser.add_argument("--out", default=str(DATA_DIR / "cache_simplified.pt"), help="Output cache .pt path")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--font",
        default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        help="Font file to render with",
    )
    parser.add_argument("--invert", action="store_true", help="White-on-black instead of black-on-white")
    args = parser.parse_args()

    print(f"Building cache from {args.csv} ...")
    cache = build_cache(args.csv, args.image_size, args.font, invert=args.invert)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)

    n = cache["images"].shape[0]
    size_mb = cache["images"].element_size() * cache["images"].nelement() / (1024 ** 2)
    print(f"Saved {n} images ({args.image_size}x{args.image_size}) to {args.out}")
    print(f"  image tensor: ~{size_mb:.1f} MB")
    print(f"  tone classes: {len(cache['tone_classes'])}")
    print(f"  pinyin classes: {len(cache['pinyin_classes'])}")


if __name__ == "__main__":
    main()
