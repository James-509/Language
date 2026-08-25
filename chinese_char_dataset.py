"""
chinese_char_dataset.py

A PyTorch Dataset that renders Chinese characters (from Unicode code points)
into images on the fly, and returns:
    image        : a tensor (rendered glyph)
    tone_label   : int, tone class index (0..num_tones-1)
    pinyin_label : int, pinyin-syllable class index (0..num_pinyin-1)

Each row of the metadata CSV (produced by build_metadata.py) is one
(character, pronunciation) pair, so a polyphonic character contributes one
dataset entry per distinct pronunciation, as requested.

Two label vocabularies are built automatically from the CSV and are exposed
on the dataset instance:
    dataset.tone_classes    -> list, index -> tone (e.g. [1, 2, 3, 4, 5])
    dataset.pinyin_classes  -> list, index -> pinyin syllable string (e.g. ["a", "ai", ...])
    dataset.tone_to_idx / dataset.idx_to_tone
    dataset.pinyin_to_idx / dataset.idx_to_pinyin

Example
-------
    from torch.utils.data import DataLoader
    from chinese_char_dataset import ChineseCharDataset

    ds = ChineseCharDataset("labels.csv", image_size=64)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    images, tones, pinyins = next(iter(loader))
"""

import csv
import functools
import glob
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset

# Default fonts to try, in order of preference. All are present on systems
# with `fonts-noto-cjk` installed (apt-get install fonts-noto-cjk).
DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]


def _find_default_fonts():
    fonts = [f for f in DEFAULT_FONT_CANDIDATES if os.path.exists(f)]
    if fonts:
        return fonts
    # fall back to searching common font directories for anything CJK-capable
    candidates = glob.glob("/usr/share/fonts/**/*CJK*.tt[cf]", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/*cjk*.tt[cf]", recursive=True)
    if not candidates:
        raise FileNotFoundError(
            "No CJK-capable font found. Install one, e.g.:\n"
            "  sudo apt-get install fonts-noto-cjk\n"
            "or pass font_paths=[...] explicitly to ChineseCharDataset."
        )
    return sorted(set(candidates))[:1]


@functools.lru_cache(maxsize=None)
def _load_font(font_path, font_index, size):
    return ImageFont.truetype(font_path, size=size, index=font_index)


def render_char(
    char,
    font_path,
    image_size=64,
    font_index=0,
    invert=False,
):
    """Render a single character to a square, grayscale PIL Image."""
    font = _load_font(font_path, font_index, int(image_size * 0.8))
    img = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image_size - w) / 2 - bbox[0]
    y = (image_size - h) / 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)

    if invert:
        img = ImageOps.invert(img)
    return img


class ChineseCharDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to the metadata CSV produced by build_metadata.py, with columns
        char, codepoint, pinyin, tone.
    image_size : int
        Output image is (image_size x image_size).
    font_paths : list[str] or None
        One or more .ttf/.ttc font files to render with. If more than one is
        given, a font is chosen per-sample (see `font_selection`), which is a
        cheap way to add visual variety (regular vs serif, etc).
    font_selection : {"random", "fixed", "cycle"}
        "random": pick a random font for every __getitem__ call (data augmentation).
        "fixed":  always use font_paths[0].
        "cycle":  deterministic, based on the sample index (reproducible).
    transform : callable or None
        Optional torchvision-style transform applied to the PIL image before
        converting to tensor. If None, the image is converted to a
        [1, H, W] float tensor in [0, 1] via a default ToTensor-equivalent.
    invert : bool
        If True, render white-on-black instead of black-on-white.
    seed : int or None
        Seed for the "random" font_selection mode reproducibility (per-worker).
    """

    def __init__(
        self,
        csv_path,
        image_size=64,
        font_paths=None,
        font_selection="fixed",
        transform=None,
        invert=False,
        seed=None,
    ):
        self.csv_path = csv_path
        self.image_size = image_size
        self.font_paths = font_paths or _find_default_fonts()
        self.font_selection = font_selection
        self.transform = transform
        self.invert = invert
        self._rng = random.Random(seed)

        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    {
                        "char": row["char"],
                        "codepoint": row["codepoint"],
                        "pinyin": row["pinyin"],
                        "tone": int(row["tone"]),
                    }
                )
        if not self.rows:
            raise ValueError(f"No rows found in {csv_path}")

        # Build label vocabularies (sorted for reproducibility across runs).
        self.tone_classes = sorted(set(r["tone"] for r in self.rows))
        self.tone_to_idx = {t: i for i, t in enumerate(self.tone_classes)}
        self.idx_to_tone = {i: t for t, i in self.tone_to_idx.items()}

        self.pinyin_classes = sorted(set(r["pinyin"] for r in self.rows))
        self.pinyin_to_idx = {p: i for i, p in enumerate(self.pinyin_classes)}
        self.idx_to_pinyin = {i: p for p, i in self.pinyin_to_idx.items()}

    def __len__(self):
        return len(self.rows)

    def _pick_font(self, idx):
        if len(self.font_paths) == 1:
            return self.font_paths[0]
        if self.font_selection == "random":
            return self._rng.choice(self.font_paths)
        if self.font_selection == "cycle":
            return self.font_paths[idx % len(self.font_paths)]
        return self.font_paths[0]  # "fixed"

    def __getitem__(self, idx):
        row = self.rows[idx]
        font_path = self._pick_font(idx)

        img = render_char(
            row["char"],
            font_path,
            image_size=self.image_size,
            invert=self.invert,
        )

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = _default_to_tensor(img)

        tone_label = self.tone_to_idx[row["tone"]]
        pinyin_label = self.pinyin_to_idx[row["pinyin"]]

        return image_tensor, tone_label, pinyin_label

    def get_meta(self, idx):
        """Return the raw (non-tensor) metadata for a sample: char, codepoint,
        pinyin string, tone int. Useful for debugging/visualization."""
        return dict(self.rows[idx])


def _default_to_tensor(pil_img):
    """Minimal PIL->tensor conversion (avoids a hard torchvision dependency)."""
    import numpy as np

    arr = np.array(pil_img, dtype="float32") / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]  # add channel dim -> [1, H, W]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


if __name__ == "__main__":
    # Quick smoke test / demo.
    import sys
    from pathlib import Path

    default_csv = Path(__file__).resolve().parent / "data" / "labels.csv"
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(default_csv)
    ds = ChineseCharDataset(csv_path, image_size=64)
    print(f"Dataset size: {len(ds)}")
    print(f"Num tone classes: {len(ds.tone_classes)} -> {ds.tone_classes}")
    print(f"Num pinyin classes: {len(ds.pinyin_classes)}")

    img_t, tone_l, pinyin_l = ds[0]
    meta = ds.get_meta(0)
    print(f"Sample 0: char={meta['char']} pinyin={meta['pinyin']} tone={meta['tone']}")
    print(f"  image tensor shape: {tuple(img_t.shape)}, dtype={img_t.dtype}")
    print(f"  tone_label={tone_l} (-> tone {ds.idx_to_tone[tone_l]})")
    print(f"  pinyin_label={pinyin_l} (-> '{ds.idx_to_pinyin[pinyin_l]}')")

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch_imgs, batch_tones, batch_pinyins = next(iter(loader))
    print(f"Batch shapes: images={tuple(batch_imgs.shape)}, "
          f"tones={tuple(batch_tones.shape)}, pinyins={tuple(batch_pinyins.shape)}")
