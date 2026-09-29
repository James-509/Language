"""
cached_dataset.py

A PyTorch Dataset that loads a pre-rendered tensor cache (produced by
prerender_cache.py) instead of rasterizing glyphs on every __getitem__ call.
This is the fast path for training loops that will iterate over the data
many times (many epochs).

Usage:
    from cached_dataset import CachedCharDataset
    ds = CachedCharDataset("data/cache_all.pt")
    image, tone_label, pinyin_label = ds[0]   # image is float32 [1, H, W] in [0, 1]
"""

import torch
from torch.utils.data import Dataset


class CachedCharDataset(Dataset):
    def __init__(self, cache_path=None, cache_dict=None, transform=None):
        """
        Provide either cache_path (loads the .pt file) or a pre-loaded
        cache_dict (as returned by torch.load on that file). Passing an
        already-loaded cache_dict lets you build two views of the same data
        with different transforms (e.g. train vs val) without loading the
        file from disk twice or duplicating the image tensor in memory --
        see train.py for how this is used to augment only the train split.
        """
        if cache_dict is None:
            if cache_path is None:
                raise ValueError("Provide either cache_path or cache_dict")
            cache_dict = torch.load(cache_path, weights_only=False)
        self.images = cache_dict["images"]  # uint8 [N, 1, H, W]
        self.tone_labels = cache_dict["tone_labels"]  # int64 [N]
        self.pinyin_labels = cache_dict["pinyin_labels"]  # int64 [N]
        self.tone_classes = cache_dict["tone_classes"]
        self.pinyin_classes = cache_dict["pinyin_classes"]
        self.chars = cache_dict["chars"]
        self.image_size = cache_dict["image_size"]

        self.tone_to_idx = {t: i for i, t in enumerate(self.tone_classes)}
        self.idx_to_tone = {i: t for t, i in self.tone_to_idx.items()}
        self.pinyin_to_idx = {p: i for i, p in enumerate(self.pinyin_classes)}
        self.idx_to_pinyin = {i: p for p, i in self.pinyin_to_idx.items()}

        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0  # [1, H, W] in [0, 1]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.tone_labels[idx], self.pinyin_labels[idx]

    def get_meta(self, idx):
        return {
            "char": self.chars[idx],
            "pinyin": self.pinyin_classes[self.pinyin_labels[idx].item()],
            "tone": self.tone_classes[self.tone_labels[idx].item()],
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    default_cache = Path(__file__).resolve().parent / "data" / "cache_all.pt"
    cache_path = sys.argv[1] if len(sys.argv) > 1 else str(default_cache)
    ds = CachedCharDataset(cache_path)
    print(f"Dataset size: {len(ds)}")
    print(f"Num tone classes: {len(ds.tone_classes)}")
    print(f"Num pinyin classes: {len(ds.pinyin_classes)}")

    img, tone_l, pinyin_l = ds[0]
    print(f"Sample 0: {ds.get_meta(0)}")
    print(f"  image shape: {tuple(img.shape)}, dtype={img.dtype}, "
          f"min={img.min():.2f}, max={img.max():.2f}")

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=64, shuffle=True)
    images, tones, pinyins = next(iter(loader))
    print(f"Batch: images={tuple(images.shape)}, tones={tuple(tones.shape)}, "
          f"pinyins={tuple(pinyins.shape)}")