# Chinese Character Image Dataset (pinyin + tone labels)

Renders Chinese characters from Unicode into images, labeled with **tone** (1-5, 5 = neutral)
and **pinyin syllable without tone** (e.g. `zhong`). Heteronyms get one row per pronunciation.

## Layout
```
build_metadata.py        scan Unicode range -> data/labels.csv
filter_by_script.py      labels.csv -> labels_simplified.csv / labels_traditional.csv (via OpenCC)
chinese_char_dataset.py  ChineseCharDataset: renders glyphs on the fly from a CSV
prerender_cache.py       render a CSV once -> data/cache_*.pt
cached_dataset.py        CachedCharDataset: fast loader for a .pt cache
train.py                 GlyphNet CNN (tone + pinyin heads), TensorBoard, augmentation
bench_workers.py         benchmark DataLoader num_workers
visualize_weights.py     conv filter grids (PNG / TensorBoard)
visualize_receptive_field.py   gradient-based receptive field heatmap
data/                    labels*.csv and cache_*.pt
```
Setup: `pip install pypinyin opencc-python-reimplemented torch torchvision pillow tensorboard`
and a CJK font (`apt-get install fonts-noto-cjk`).

## Pipeline
```bash
python build_metadata.py                                   # data/labels.csv
python filter_by_script.py                                 # simplified / traditional CSVs
python prerender_cache.py --csv data/labels.csv --out data/cache_all.pt
python prerender_cache.py --csv data/labels_simplified.csv  --out data/cache_simplified.pt
python prerender_cache.py --csv data/labels_traditional.csv --out data/cache_traditional.pt
```

## Training
```bash
python train.py --epochs 20                                # defaults to data/cache_all.pt
python train.py --cache data/cache_traditional.pt --epochs 20
tensorboard --logdir runs
```
Each run saves to its own `checkpoints/<cache>_<timestamp>.pt`; an existing checkpoint is only
overwritten if the new run beats its saved accuracy.

### Data augmentation
Training images get slight random rotation and rescaling via torchvision
`RandomAffine(degrees=10, scale=(0.9, 1.1), fill=1.0)`. `fill=1.0` keeps the exposed
corners white (matching the white background). **Validation is never augmented**, so val
accuracy reflects clean glyphs. The cache is loaded once and shared between a train view
(augmented) and a val view (clean), so memory isn't doubled.

Flags: `--no-augment`, `--augment-degrees 10`, `--augment-scale-min 0.9`, `--augment-scale-max 1.1`.

## Notes
- `--num-workers`: with the in-memory cache, `0` is often fastest; run `bench_workers.py` to check.
  Augmentation adds per-sample CPU work, so workers may now help more than before.
- `visualize_weights.py`: only the first conv layer is directly interpretable; deeper layers show
  a channel-averaged summary.
- `visualize_receptive_field.py`: GlyphNet's theoretical RF grows to 36px before the global pool
  (which makes every unit depend on the full 64x64 image).