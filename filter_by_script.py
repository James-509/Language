"""
filter_by_script.py

Splits labels.csv (char, codepoint, pinyin, tone) into two CSVs:
  labels_simplified.csv  - characters valid in Simplified Chinese
  labels_traditional.csv - characters valid in Traditional Chinese

Classification per character (via OpenCC):
  - unchanged by s2t/t2s conversion (e.g. 一, 人, 中) -> included in BOTH
  - changes under t2s but not s2t (i.e. this IS the traditional form,
    e.g. 語) -> traditional only
  - changes under s2t but not t2s (i.e. this IS the simplified form,
    e.g. 语) -> simplified only

Usage:
    python filter_by_script.py --in labels.csv \
        --out-simplified labels_simplified.csv \
        --out-traditional labels_traditional.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from opencc import OpenCC

DATA_DIR = Path(__file__).resolve().parent / "data"


def classify_chars(chars):
    """Return dict: char -> 'both' | 'simplified' | 'traditional'."""
    s2t = OpenCC("s2t")
    t2s = OpenCC("t2s")
    result = {}
    for c in chars:
        to_trad = s2t.convert(c)
        to_simp = t2s.convert(c)
        if to_trad == c and to_simp == c:
            result[c] = "both"
        elif to_trad != c and to_simp == c:
            # converting to traditional changes it -> this char IS simplified
            result[c] = "simplified"
        elif to_simp != c and to_trad == c:
            # converting to simplified changes it -> this char IS traditional
            result[c] = "traditional"
        else:
            # both conversions change it (rare / ambiguous multi-mapping) -
            # treat conservatively as belonging to neither exclusive set,
            # but still usable in both since it round-trips through neither
            # cleanly. We fall back to 'both' so no data is silently dropped.
            result[c] = "both"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(DATA_DIR / "labels.csv"))
    parser.add_argument("--out-simplified", default=str(DATA_DIR / "labels_simplified.csv"))
    parser.add_argument("--out-traditional", default=str(DATA_DIR / "labels_traditional.csv"))
    args = parser.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unique_chars = sorted(set(r["char"] for r in rows))
    print(f"Classifying {len(unique_chars)} unique characters...")
    classification = classify_chars(unique_chars)

    counts = defaultdict(int)
    for v in classification.values():
        counts[v] += 1
    print(f"  simplified-only:  {counts['simplified']}")
    print(f"  traditional-only: {counts['traditional']}")
    print(f"  shared (both):    {counts['both']}")

    simp_rows = [r for r in rows if classification[r["char"]] in ("simplified", "both")]
    trad_rows = [r for r in rows if classification[r["char"]] in ("traditional", "both")]

    fieldnames = ["char", "codepoint", "pinyin", "tone"]
    Path(args.out_simplified).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_traditional).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_simplified, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(simp_rows)
    with open(args.out_traditional, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trad_rows)

    print(f"Wrote {len(simp_rows)} rows ({len(set(r['char'] for r in simp_rows))} chars) "
          f"-> {args.out_simplified}")
    print(f"Wrote {len(trad_rows)} rows ({len(set(r['char'] for r in trad_rows))} chars) "
          f"-> {args.out_traditional}")


if __name__ == "__main__":
    main()
