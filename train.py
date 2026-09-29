"""
train.py

Trains a CNN with two output heads on the pre-rendered character cache:
  - pinyin head: classifies the pinyin syllable (no tone), ~420 classes
  - tone head:   classifies the tone (1-5, 5=neutral), 5 classes

Usage:
    python train.py --epochs 20
    python train.py --cache data/cache_traditional.pt --epochs 20 --batch-size 128

Progress is logged to TensorBoard by default (loss, tone/pinyin/both
accuracy, learning rate, weight/gradient histograms, for train and val).
View it with:
    tensorboard --logdir runs
Disable with --no-tensorboard / --no-weight-histograms.

Training images are augmented by default with slight random rotation and
rescaling (torchvision transforms.RandomAffine) to improve generalization;
validation images are never augmented, so val accuracy always reflects
performance on clean, canonical glyphs. Tune with --augment-degrees /
--augment-scale-min / --augment-scale-max, or disable with --no-augment.

The cache's own train/val split is random per run unless --seed is fixed.
Checkpoints (best val accuracy) are saved to --out, and won't overwrite an
existing checkpoint with a worse one.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from cached_dataset import CachedCharDataset

DATA_DIR = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class GlyphNet(nn.Module):
    """Shared conv trunk, two linear classification heads."""

    def __init__(self, num_tones, num_pinyin, image_size=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # /8

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
        )
        self.tone_head = nn.Linear(128, num_tones)
        self.pinyin_head = nn.Linear(128, num_pinyin)

    def forward(self, x):
        feats = self.trunk(x)
        return self.tone_head(feats), self.pinyin_head(feats)


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None, pinyin_loss_weight=1.0):
    """If optimizer is given, trains; otherwise evaluates (no grad)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_tone_correct = 0
    total_pinyin_correct = 0
    total_both_correct = 0
    n = 0

    with torch.set_grad_enabled(is_train):
        for images, tones, pinyins in loader:
            images = images.to(device, non_blocking=True)
            tones = tones.to(device, non_blocking=True)
            pinyins = pinyins.to(device, non_blocking=True)

            tone_logits, pinyin_logits = model(images)
            tone_loss = F.cross_entropy(tone_logits, tones)
            pinyin_loss = F.cross_entropy(pinyin_logits, pinyins)
            loss = tone_loss + pinyin_loss_weight * pinyin_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            tone_pred = tone_logits.argmax(dim=1)
            pinyin_pred = pinyin_logits.argmax(dim=1)
            tone_correct = tone_pred == tones
            pinyin_correct = pinyin_pred == pinyins
            total_tone_correct += tone_correct.sum().item()
            total_pinyin_correct += pinyin_correct.sum().item()
            total_both_correct += (tone_correct & pinyin_correct).sum().item()
            n += batch_size

    return {
        "loss": total_loss / n,
        "tone_acc": total_tone_correct / n,
        "pinyin_acc": total_pinyin_correct / n,
        "both_acc": total_both_correct / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DATA_DIR / "cache_all.pt"),
                         help="Path to a prerender_cache.py .pt file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--pinyin-loss-weight", type=float, default=1.0,
                         help="Relative weight of the pinyin loss vs tone loss (both start at 1.0)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                         help="Where to save the best checkpoint. Default: "
                              "checkpoints/<cache_name>_<timestamp>.pt (a fresh file per run)")
    parser.add_argument("--device", default=None, help="cuda / cpu / mps (auto-detected if omitted)")
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "runs"),
                         help="TensorBoard log directory (a timestamped subfolder is created inside it)")
    parser.add_argument("--run-name", default=None,
                         help="Name for this run's TensorBoard subfolder / checkpoint stem "
                              "(default: cache name + timestamp)")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--no-weight-histograms", action="store_true",
                         help="Skip logging weight/gradient histograms (they add some overhead per epoch)")
    parser.add_argument("--no-augment", action="store_true",
                         help="Disable data augmentation (rotation + rescaling) on the training split")
    parser.add_argument("--augment-degrees", type=float, default=10.0,
                         help="Max +/- rotation angle in degrees for training augmentation")
    parser.add_argument("--augment-scale-min", type=float, default=0.9,
                         help="Min random scale factor for training augmentation")
    parser.add_argument("--augment-scale-max", type=float, default=1.1,
                         help="Max random scale factor for training augmentation")
    args = parser.parse_args()

    if args.out is None:
        checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        stem = args.run_name or f"{Path(args.cache).stem}_{time.strftime('%Y%m%d-%H%M%S')}"
        args.out = str(checkpoints_dir / f"{stem}.pt")

    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the cache ONCE, then build two thin dataset views over the same
    # underlying tensors (one with augmentation, one without) so train and
    # val can have different transforms without duplicating the image data
    # in memory or reading the file from disk twice.
    raw_cache = torch.load(args.cache, weights_only=False)

    train_transform = None
    if not args.no_augment:
        train_transform = transforms.RandomAffine(
            degrees=args.augment_degrees,
            scale=(args.augment_scale_min, args.augment_scale_max),
            fill=1.0,  # background is white (1.0) after ToTensor-style scaling; avoid black corners
        )
        print(f"Augmenting training split with: RandomAffine(degrees={args.augment_degrees}, "
              f"scale=({args.augment_scale_min}, {args.augment_scale_max}))")

    train_view = CachedCharDataset(cache_dict=raw_cache, transform=train_transform)
    val_view = CachedCharDataset(cache_dict=raw_cache, transform=None)
    # val_view has no transform on purpose, so val accuracy reflects
    # performance on clean, canonical glyphs.

    n = len(train_view)
    n_val = int(n * args.val_fraction)
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices, val_indices = perm[:n_train], perm[n_train:]

    train_ds = Subset(train_view, train_indices)
    val_ds = Subset(val_view, val_indices)
    print(f"Train: {n_train}  Val: {n_val}")
    print(f"Tone classes: {len(train_view.tone_classes)}  Pinyin classes: {len(train_view.pinyin_classes)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = GlyphNet(
        num_tones=len(train_view.tone_classes),
        num_pinyin=len(train_view.pinyin_classes),
        image_size=train_view.image_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    writer = None
    if not args.no_tensorboard:
        run_name = args.run_name or f"{Path(args.cache).stem}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = Path(args.log_dir) / run_name
        writer = SummaryWriter(log_dir=str(run_dir))
        print(f"TensorBoard logging to: {run_dir}")
        print(f"  view with: tensorboard --logdir {args.log_dir}")

    best_both_acc = -1.0
    if Path(args.out).exists():
        try:
            existing = torch.load(args.out, weights_only=False)
            best_both_acc = existing.get("val_both_acc", -1.0)
            print(f"Found existing checkpoint at {args.out} (both_acc={best_both_acc*100:.1f}%); "
                  f"will only overwrite it if this run does better.")
        except Exception as e:
            print(f"Could not read existing checkpoint at {args.out} ({e}); starting fresh.")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            pinyin_loss_weight=args.pinyin_loss_weight,
        )
        val_stats = run_epoch(
            model, val_loader, device, optimizer=None,
            pinyin_loss_weight=args.pinyin_loss_weight,
        )
        scheduler.step(val_stats["both_acc"])
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"train loss {train_stats['loss']:.3f} tone {train_stats['tone_acc']*100:5.1f}% "
            f"pinyin {train_stats['pinyin_acc']*100:5.1f}% both {train_stats['both_acc']*100:5.1f}% || "
            f"val loss {val_stats['loss']:.3f} tone {val_stats['tone_acc']*100:5.1f}% "
            f"pinyin {val_stats['pinyin_acc']*100:5.1f}% both {val_stats['both_acc']*100:5.1f}%"
        )

        if writer is not None:
            writer.add_scalars("loss", {"train": train_stats["loss"], "val": val_stats["loss"]}, epoch)
            writer.add_scalars("tone_acc", {"train": train_stats["tone_acc"], "val": val_stats["tone_acc"]}, epoch)
            writer.add_scalars("pinyin_acc", {"train": train_stats["pinyin_acc"], "val": val_stats["pinyin_acc"]}, epoch)
            writer.add_scalars("both_acc", {"train": train_stats["both_acc"], "val": val_stats["both_acc"]}, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            if not args.no_weight_histograms:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"weights/{name}", param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"grads/{name}", param.grad, epoch)

        if val_stats["both_acc"] > best_both_acc:
            best_both_acc = val_stats["both_acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tone_classes": train_view.tone_classes,
                    "pinyin_classes": train_view.pinyin_classes,
                    "image_size": train_view.image_size,
                    "epoch": epoch,
                    "val_both_acc": best_both_acc,
                },
                args.out,
            )
            print(f"  -> new best (both_acc={best_both_acc*100:.1f}%), saved to {args.out}")

    if writer is not None:
        writer.close()

    print(f"Done. Best val both-correct accuracy: {best_both_acc*100:.1f}% (checkpoint: {args.out})")


if __name__ == "__main__":
    main()