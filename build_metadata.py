"""
build_metadata.py

Builds a CSV of (character, codepoint, pinyin-without-tone, tone) entries
covering every character in the chosen Unicode range, expanded so that a
character with multiple pronunciations (heteronyms) gets one row per
distinct (pinyin, tone) reading.

Usage:
    python build_metadata.py --out labels.csv
    python build_metadata.py --out labels.csv --ranges 4E00-9FFF 3400-4DBF
"""

import argparse
import csv
from pathlib import Path

from pypinyin import Style, pinyin

DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_ranges(range_strs):
    """Turn ['4E00-9FFF', '3400-4DBF'] into a list of (start, end) ints."""
    ranges = []
    for r in range_strs:
        start_s, end_s = r.split("-")
        ranges.append((int(start_s, 16), int(end_s, 16)))
    return ranges


def iter_codepoints(ranges):
    for start, end in ranges:
        for cp in range(start, end + 1):
            yield cp


def build_rows(ranges, v_to_u=False):
    """
    Yields dict rows: char, codepoint (hex str), pinyin (no tone), tone (int 1-5).

    Tone 5 is used for the neutral tone (pypinyin's neutral_tone_with_five=True),
    tones 1-4 are the four standard Mandarin tones.
    """
    seen_chars = 0
    for cp in iter_codepoints(ranges):
        char = chr(cp)
        result = pinyin(
            char,
            style=Style.TONE3,
            heteronym=True,
            neutral_tone_with_five=True,
            v_to_u=v_to_u,
            errors="ignore",  # characters with no known reading are skipped
        )
        if not result or not result[0]:
            continue
        readings = result[0]  # list of readings like 'zhong1', 'zhong4'
        seen_pairs = set()
        seen_chars += 1
        for reading in readings:
            # split trailing tone digit (1-5) from the syllable letters
            if not reading or not reading[-1].isdigit():
                # no digit means pypinyin couldn't determine a tone; skip
                continue
            tone = int(reading[-1])
            syllable = reading[:-1]
            if tone not in (1, 2, 3, 4, 5):
                continue
            key = (syllable, tone)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            yield {
                "char": char,
                "codepoint": f"{cp:04X}",
                "pinyin": syllable,
                "tone": tone,
            }
    print(f"Characters with at least one reading: {seen_chars}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "labels.csv"), help="Output CSV path")
    parser.add_argument(
        "--ranges",
        nargs="+",
        default=["4E00-9FFF"],
        help=(
            "Unicode codepoint ranges (hex, inclusive) to scan, e.g. "
            "4E00-9FFF (CJK Unified Ideographs, default) or "
            "3400-4DBF (CJK Extension A, rarer characters)."
        ),
    )
    parser.add_argument(
        "--v-to-u",
        action="store_true",
        help="Render u-umlaut as literal 'ü' instead of ASCII 'v' (e.g. lü vs lv).",
    )
    args = parser.parse_args()

    ranges = parse_ranges(args.ranges)

    rows = list(build_rows(ranges, v_to_u=args.v_to_u))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "codepoint", "pinyin", "tone"])
        writer.writeheader()
        writer.writerows(rows)

    n_chars = len(set(r["char"] for r in rows))
    n_pinyin = len(set(r["pinyin"] for r in rows))
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"  unique characters: {n_chars}")
    print(f"  unique pinyin syllables (no tone): {n_pinyin}")
    print(f"  tone distribution:")
    from collections import Counter

    tone_counts = Counter(r["tone"] for r in rows)
    for tone in sorted(tone_counts):
        label = "neutral" if tone == 5 else str(tone)
        print(f"    tone {label}: {tone_counts[tone]}")


if __name__ == "__main__":
    main()
