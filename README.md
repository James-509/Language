# Chinese Character Image Dataset (pinyin + tone labels)

Renders every Chinese character in a chosen Unicode range into an image, and
pairs it with two labels:

- **tone**: 1, 2, 3, 4, or 5 (5 = neutral tone)
- **pinyin**: the syllable with no tone mark/number, e.g. `"zhong"`

Characters with multiple pronunciations (heteronyms, e.g. 中 = zhong1/zhong4,
重 = zhong4/chong2/tong2) get one row per distinct pronunciation, as requested.

## Files

```
chinese_char_dataset/
├── build_metadata.py       # scans Unicode range -> data/labels.csv
├── filter_by_script.py     # splits labels.csv -> simplified/traditional CSVs
├── chinese_char_dataset.py # ChineseCharDataset: renders on the fly from a CSV
├── prerender_cache.py      # renders a CSV once -> data/cache_*.pt
├── cached_dataset.py       # CachedCharDataset: loads a prebuilt .pt cache (fast path)
├── train.py                # trains GlyphNet (CNN, tone + pinyin heads) on a cache
├── README.md
└── data/
    ├── labels.csv                # all readings, both scripts mixed (~29.7k rows)
    ├── labels_simplified.csv     # simplified-only + shared characters
    ├── labels_traditional.csv    # traditional-only + shared characters
    ├── cache_simplified.pt       # pre-rendered images + labels (simplified)
    └── cache_traditional.pt      # pre-rendered images + labels (traditional)
```

Every script takes its input/output paths as CLI arguments (or constructor
arguments, for the two Dataset classes) — none of them hardcode a location.
Defaults point at the `data/` subfolder next to the script itself (resolved
via the script's own file path, not your current working directory), so the
commands below work whether you run them from inside `chinese_char_dataset/`
or from anywhere else on the filesystem.

## Setup

```bash
pip install pypinyin torch pillow
# a CJK-capable font must be installed, e.g.:
sudo apt-get install fonts-noto-cjk
```

## 1. Regenerate the metadata (optional — labels.csv is already included)

```bash
# default: CJK Unified Ideographs block (most common ~20,900 characters)
python build_metadata.py --out data/labels.csv

# add rarer characters from CJK Extension A too
python build_metadata.py --out data/labels.csv --ranges 4E00-9FFF 3400-4DBF

# use literal ü instead of ASCII 'v' in pinyin (lü vs lv)
python build_metadata.py --out data/labels.csv --v-to-u
```

## 2. Use the dataset in PyTorch

```python
from torch.utils.data import DataLoader
from chinese_char_dataset import ChineseCharDataset

ds = ChineseCharDataset("data/labels.csv", image_size=64)

print(len(ds))                  # 29743
print(ds.tone_classes)          # [1, 2, 3, 4, 5]
print(len(ds.pinyin_classes))   # 420

image, tone_label, pinyin_label = ds[0]
# image: FloatTensor [1, 64, 64], values in [0, 1]
# tone_label: int, index into ds.tone_classes (use ds.idx_to_tone[tone_label] to recover 1-5)
# pinyin_label: int, index into ds.pinyin_classes (use ds.idx_to_pinyin[pinyin_label] to recover the string)

loader = DataLoader(ds, batch_size=64, shuffle=True)
for images, tones, pinyins in loader:
    ...
```

To inspect the original character/pinyin string/tone for any sample:

```python
ds.get_meta(0)   # {'char': '一', 'codepoint': '4E00', 'pinyin': 'yi', 'tone': 1}
```

### Training two heads (tone classifier + pinyin classifier)

Since `pinyin_label` is a single class index over the closed set of ~420
Mandarin syllables, both tasks are plain multi-class classification — e.g. a
shared CNN trunk with two output heads:

```python
import torch.nn as nn

class GlyphNet(nn.Module):
    def __init__(self, num_tones, num_pinyin):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.tone_head = nn.Linear(64, num_tones)
        self.pinyin_head = nn.Linear(64, num_pinyin)

    def forward(self, x):
        feats = self.trunk(x)
        return self.tone_head(feats), self.pinyin_head(feats)

model = GlyphNet(len(ds.tone_classes), len(ds.pinyin_classes))
```

### Data augmentation / multiple fonts

Pass several font files and a selection strategy to vary the rendering style:

```python
ds = ChineseCharDataset(
    "data/labels.csv",
    image_size=64,
    font_paths=[
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ],
    font_selection="random",  # "fixed" | "cycle" | "random"
)
```

You can also pass any torchvision `transform` (rotation, noise, resizing,
normalization, etc.) via the `transform=` argument — it receives the PIL
image before the default tensor conversion.

## Pre-rendered tensor caches (fast path for training)

`ChineseCharDataset` renders glyphs on the fly, which is fine for a quick
look but wastes CPU if you're iterating over the same data for many epochs.
For actual training, pre-render once into a cache file and load from that
instead:

```bash
# 1. Split into simplified-only / traditional-only metadata
python filter_by_script.py --in data/labels.csv \
    --out-simplified data/labels_simplified.csv \
    --out-traditional data/labels_traditional.csv

# 2. Render each set once into a single .pt tensor cache
python prerender_cache.py --csv data/labels_simplified.csv  --out data/cache_simplified.pt
python prerender_cache.py --csv data/labels_traditional.csv --out data/cache_traditional.pt
```

`filter_by_script.py` uses OpenCC to classify each character: characters
identical in both scripts (一, 人, 中, ...) are kept in both sets;
characters that differ (语 vs 語) go only into the set matching their form.
On the full CJK Unified Ideographs block this currently gives:

| set | rows | unique chars |
|---|---|---|
| simplified | 24,338 | 17,381 |
| traditional | 26,542 | 18,325 |

Each cache is a single `.pt` file (`torch.save`'d dict) containing:

- `images`: `uint8` tensor `[N, 1, H, W]`
- `tone_labels`, `pinyin_labels`: `int64` tensors `[N]`
- `tone_classes`, `pinyin_classes`: label vocabularies (index -> value)
- `chars`: source character for each row
- `image_size`, `font_path`: metadata

Load it with the lightweight `CachedCharDataset`:

```python
from torch.utils.data import DataLoader
from cached_dataset import CachedCharDataset

ds = CachedCharDataset("data/cache_simplified.pt")   # or data/cache_traditional.pt
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)

for images, tones, pinyins in loader:
    # images: float32 [B, 1, 64, 64] in [0, 1]
    # tones, pinyins: int64 [B]
    ...
```

This is a drop-in replacement for `ChineseCharDataset` — same
`(image, tone_label, pinyin_label)` item shape, same `idx_to_tone` /
`idx_to_pinyin` / `get_meta()` API — just backed by an in-memory tensor
instead of live rendering, so `__getitem__` is a tensor slice instead of a
font-rasterization call.

Regenerate a cache anytime with a different `--image-size` or `--font`, or
pass `--invert` for white-on-black glyphs.

## Training a convnet

`train.py` trains a small CNN (`GlyphNet`) with a shared conv trunk and two
linear heads — one for tone (5-way), one for pinyin (~420-way) — directly
against a cache file:

```bash
python train.py --cache data/cache_simplified.pt --epochs 20
python train.py --cache data/cache_traditional.pt --epochs 20 --batch-size 128
```

It does a random train/val split of the cache (`--val-fraction`, default
10%), trains with Adam + `ReduceLROnPlateau`, and saves the checkpoint with
the best validation "both correct" accuracy (tone AND pinyin both right on
the same sample) to `--out`. Per-epoch it reports loss and accuracy for
tone, pinyin, and both-correct, on train and val.

**Every run gets its own checkpoint file by default** — `--out` defaults to
`checkpoints/<cache_name>_<timestamp>.pt`, so running `train.py` multiple
times won't clobber earlier results. If you do point `--out` (or
`--run-name`) at an existing checkpoint on purpose, the script reads its
saved accuracy first and only overwrites it if the new run actually beats
that number — it won't silently replace a good checkpoint with a worse one.

Key flags:
- `--pinyin-loss-weight`: relative weight of the pinyin cross-entropy loss
  vs the tone loss (both default to 1.0 — pinyin is the harder task with
  ~420 classes vs tone's 5, so upweighting tone slightly, e.g. `--pinyin-loss-weight 1.0`
  with a lower LR just for the tone head, is one thing to experiment with
  if tone accuracy lags).
- `--device`: auto-detects CUDA / Apple MPS / CPU if omitted.
- `--seed`: controls both the train/val split and model init for reproducibility.

### Visualizing training progress with TensorBoard

Progress is logged to TensorBoard automatically — no extra flags needed.
Each run gets its own timestamped subfolder under `runs/`:

```bash
python train.py --epochs 20
# TensorBoard logging to: .../chinese_char_dataset/runs/cache_simplified_20260816-143012
#   view with: tensorboard --logdir runs
```

In another terminal:

```bash
pip install tensorboard   # if not already installed
tensorboard --logdir runs
```

Then open the URL it prints (usually `http://localhost:6006`). You'll see,
for both train and val, on the same plot: loss, tone accuracy, pinyin
accuracy, both-correct accuracy, plus the learning rate over time (useful
for seeing when `ReduceLROnPlateau` kicks in). The Histograms/Distributions
tab also shows every layer's weight and gradient distribution evolving
epoch by epoch (disable with `--no-weight-histograms` if it's too much
overhead/clutter).

Useful flags:
- `--run-name my_experiment`: name the subfolder yourself instead of the
  auto-generated `<cache_name>_<timestamp>`, handy for comparing runs side
  by side in TensorBoard (it overlays every subfolder under `--logdir` on
  the same charts).
- `--log-dir other/path`: change where the `runs/` folder lives.
- `--no-tensorboard`: skip logging entirely.

### How many DataLoader workers should I use?

Benchmark it rather than guess — `bench_workers.py` times real batches at
several `num_workers` settings and reports images/sec for each:

```bash
python bench_workers.py --cache data/cache_simplified.pt --workers 0 1 2 4 8
```

```
workers | sec/batch |   imgs/sec |  total (s)
------------------------------------------------
      0 |     0.004 |    14104.4 |       0.02
      2 |     0.007 |     8831.0 |       0.04
...
```

Pick the smallest `num_workers` within ~10% of the fastest result — beyond
that point more workers just burn RAM/CPU without further speedup.

**Why `num_workers=0` is often *fastest* for this specific dataset:** worker
processes only help when each `__getitem__` call does real CPU work worth
parallelizing (e.g. `ChineseCharDataset`, which rasterizes a font glyph
every call). `CachedCharDataset` just slices an already-loaded tensor and
divides by 255 — that's cheaper than the overhead of shipping the result
between processes. Extra workers there add cost (process startup, and each
worker gets its own copy of the in-memory image tensor, multiplying RAM use)
without saving any real compute. Rule of thumb:
- **Live rendering (`ChineseCharDataset`)**: more workers usually helps,
  up to roughly your CPU core count.
- **Pre-rendered cache (`CachedCharDataset`, what `train.py` uses)**: try
  `0` first. Only add workers if `bench_workers.py` actually shows a win —
  which is more likely once you're on GPU and the CPU-side pipeline has to
  keep up with a much faster training step.

General signs to watch for beyond the benchmark:
- **GPU sitting idle between batches** (check `nvidia-smi` utilization
  while training) → data loading is the bottleneck → try more workers.
- **High RAM usage that scales with `num_workers`** → each worker is
  holding its own copy of data → fewer workers, or restructure the dataset
  so large arrays are loaded lazily instead of held per-worker.
- **`DataLoader` warning about worker count exceeding CPU cores** → PyTorch
  itself will tell you when `num_workers` is set higher than your machine's
  core count; that's a hard ceiling worth respecting.

Loading a trained checkpoint later:

```python
import torch
from train import GlyphNet

ckpt = torch.load("checkpoints/cache_simplified_20260816-150233.pt", weights_only=False)
model = GlyphNet(len(ckpt["tone_classes"]), len(ckpt["pinyin_classes"]), ckpt["image_size"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tone_logits, pinyin_logits = model(some_image_batch)
tone_pred = ckpt["tone_classes"][tone_logits.argmax(1).item()]
pinyin_pred = ckpt["pinyin_classes"][pinyin_logits.argmax(1).item()]
```

### Visualizing layer weights directly

The weight/gradient histograms above are automatic (logged every epoch by
`train.py`), but for actually *looking at* the filters, `visualize_weights.py`
renders them as an image grid:

```bash
# see what conv layers exist and their shapes
python visualize_weights.py --list-layers

# first conv layer (1 input channel) -> each tile IS the literal 3x3 filter
python visualize_weights.py --checkpoint checkpoints/cache_simplified_20260816-150233.pt --layer 0

# a deeper layer (many input channels) -> each tile is a summary
# (mean |weight| across input channels), not the literal weights
python visualize_weights.py --checkpoint checkpoints/cache_simplified_20260816-150233.pt --layer 4

# log into TensorBoard's Images tab instead of / as well as a PNG
python visualize_weights.py --checkpoint ... --layer 0 --log-dir runs/weights_demo
```

Omit `--checkpoint` to look at freshly-initialized (untrained) weights —
useful as a sanity check on initialization, or to see what "no signal yet"
looks like before comparing against a trained model.

Only the **first** conv layer is directly interpretable this way, since its
filters are `[out_channels, 1, 3, 3]` — one small grayscale image per
filter. Every deeper layer's filters are `[out_channels, in_channels, 3, 3]`
— a 3D block, not a picture — so the script shows a summary (average
absolute weight across input channels) rather than the literal values.
That's a real limitation, not a display quirk: genuinely understanding what
a deep filter "looks for" usually means activation maximization (optimizing
an input image to maximally excite that neuron) rather than staring at its
raw numbers, which is a heavier technique not implemented here.

### Visualizing a neuron's receptive field

TensorBoard doesn't compute receptive fields directly — RF is a fixed
geometric property of the architecture (kernel sizes/strides/padding), not
something that changes over training, so it doesn't fit TensorBoard's
scalar/image-over-time model. `GlyphNet`'s theoretical RF, worked out by
hand (each conv is 3x3 stride 1, each pool is 2x2 stride 2):

| after layer | receptive field |
|---|---|
| conv+conv (stage 1) | 5 px |
| pool 1 | 6 px |
| conv+conv (stage 2) | 14 px |
| pool 2 | 16 px |
| conv+conv (stage 3) | 32 px |
| pool 3 | 36 px |
| `AdaptiveAvgPool2d(1)` | entire 64x64 image (averages every spatial location) |

`visualize_receptive_field.py` verifies this empirically by backpropagating
a gradient from one chosen neuron in the last conv feature map back to the
input pixels, and renders which pixels actually influenced it as a heatmap
overlaid on the character:

```bash
python visualize_receptive_field.py --char 中 --channel 0
# Empirical receptive field: rows 18-48, cols 22-53 (31x32 px, image is 64x64)
# Saved overlay to receptive_field.png
```

Pass `--row`/`--col` to inspect a different neuron in the 8x8 feature grid
(defaults to the center one) — a corner neuron shows a visibly smaller,
boundary-clipped receptive field than a center one, which is expected: the
theoretical numbers above assume no boundary clipping, real neurons near
the edge see less.

Optionally log the heatmap into TensorBoard's Images tab instead of just
saving a PNG:

```bash
python visualize_receptive_field.py --char 中 --channel 0 --log-dir runs/rf_demo
tensorboard --logdir runs
```

Pass `--checkpoint path/to/checkpoint.pt` to inspect a trained model's
neurons instead of a freshly-initialized one — note the receptive field
*size* is identical either way (it only depends on architecture), but which
input pixels actually get a strong gradient (vs a near-zero one) does
depend on the learned weights.

## Notes / design choices

- **Tone 5 = neutral tone** (轻声), not "no tone" — this keeps tone as a
  clean 5-way classification problem instead of a special-cased optional field.
- **Pinyin excludes tone marks/numbers entirely** (e.g. `"zhong"`, not
  `"zhong1"` or `"zhōng"`), per your request, and is returned as a class
  index (`ds.pinyin_classes` holds the ~420-syllable vocabulary; use
  `ds.idx_to_pinyin[idx]` to get the string back).
- **One row per pronunciation**: a heteronym like 了 (le / liǎo / liào)
  produces 3 separate dataset rows, each with the same character but a
  different (pinyin, tone) label pair.
- Readings come from pypinyin's built-in dictionary (based on cc-cedict /
  the classic Mandarin phonetic tables); a handful of very obscure or
  disputed characters may have incomplete data — this mirrors any
  dictionary-based pinyin source.