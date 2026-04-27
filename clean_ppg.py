"""
clean_ppg.py  —  PPG cleaning stage (pre-filter)

Study: MAX30101 melanin-bias study, AGC bypassed, ~25 Hz sampling.
Input CSVs (no header):  timestamp_ms, red, ir, green
Output CSVs (header):    timestamp_ms, red, ir, green, outlier_flag

What this script does, in order:
  1. Read each raw CSV (headerless).
  2. Drop rows containing any NaN / missing values; count how many.
  3. Trim by timestamp: keep samples with timestamp_ms >= 2000 ms, to
     discard MAX32664 bootloader-exit artifacts (static high values in
     the first ~2 s). Trimming by timestamp rather than row index is
     important because P6 has jitter and a gap.
  4. Detect outliers at |x - rolling_mean| > 5 * rolling_std, computed
     independently per LED channel on a centered 51-sample window
     (~2.04 s @ 25 Hz — larger than one cardiac cycle). Rows are FLAGGED
     (outlier_flag = 1) but NOT deleted.
  5. Write cleaned/<same_name>.csv and append a row to cleaning_summary.txt.

No filters (bandpass, detrend, etc.) are applied here — that is the
next stage's job.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- configuration ----------
HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE
OUTPUT_DIR = HERE / "cleaned"
SUMMARY_PATH = HERE / "cleaning_summary.txt"

TRIM_MS = 2000               # discard first 2 s (bootloader artifacts)
SAMPLING_HZ = 25             # nominal; used only for window sizing
ROLL_WINDOW = 51             # centered, ~2.04 s — spans a cardiac cycle
OUTLIER_Z = 5.0              # |x - local_mean| > 5 * local_std -> flag
MIN_WINDOW_PERIODS = 11      # ~0.44 s: enough to have a real std

CHANNELS = ["red", "ir", "green"]
COLUMNS = ["timestamp_ms"] + CHANNELS


def clean_one(path: Path) -> dict:
    """Clean one participant file. Returns a dict of summary stats."""
    warnings: list[str] = []

    # (1) read headerless
    raw = pd.read_csv(path, header=None, names=COLUMNS)
    n_original = len(raw)

    # (2) drop NaN / missing
    n_nan = int(raw[COLUMNS].isna().any(axis=1).sum())
    df = raw.dropna(subset=COLUMNS).copy()

    # (3) timestamp-based trim of first 2 s
    #     Use relative time (t - t0) so we strip a fixed wall-clock
    #     window from the START of the record regardless of what the
    #     first timestamp actually is.
    t0 = df["timestamp_ms"].iloc[0]
    rel_ms = df["timestamp_ms"] - t0
    n_before_trim = len(df)
    df = df.loc[rel_ms >= TRIM_MS].reset_index(drop=True)
    n_trimmed = n_before_trim - len(df)

    if len(df) == 0:
        warnings.append("EMPTY_AFTER_TRIM: no samples survived the 2 s trim")
        _write_empty(path)
        return {
            "file": path.name,
            "n_original": n_original,
            "n_nan_dropped": n_nan,
            "n_trimmed_first_2s": n_trimmed,
            "n_final": 0,
            "duration_s": 0.0,
            "outliers_any": 0,
            "outliers_red": 0,
            "outliers_ir": 0,
            "outliers_green": 0,
            "warnings": warnings,
        }

    # (4) outlier flagging (per-channel rolling 5σ)
    flag_any = pd.Series(False, index=df.index)
    per_channel = {}
    for ch in CHANNELS:
        roll = df[ch].rolling(ROLL_WINDOW, center=True,
                              min_periods=MIN_WINDOW_PERIODS)
        mu, sigma = roll.mean(), roll.std()
        # guard against zero std at the very edges where only duplicates
        # were used to seed the window
        sigma = sigma.replace(0, np.nan)
        z = (df[ch] - mu).abs() / sigma
        is_out = (z > OUTLIER_Z).fillna(False)
        per_channel[ch] = int(is_out.sum())
        flag_any |= is_out

    df["outlier_flag"] = flag_any.astype(int)

    # duration
    duration_s = (df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0
    if duration_s < 60.0:
        warnings.append(
            f"SHORT_RECORDING: usable window {duration_s:.2f} s < 60 s target"
        )

    # (5) write cleaned output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / path.name
    df[["timestamp_ms"] + CHANNELS + ["outlier_flag"]].to_csv(
        out_path, index=False
    )

    return {
        "file": path.name,
        "n_original": n_original,
        "n_nan_dropped": n_nan,
        "n_trimmed_first_2s": n_trimmed,
        "n_final": len(df),
        "duration_s": duration_s,
        "outliers_any": int(flag_any.sum()),
        "outliers_red": per_channel["red"],
        "outliers_ir": per_channel["ir"],
        "outliers_green": per_channel["green"],
        "warnings": warnings,
    }


def _write_empty(path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS + ["outlier_flag"]).to_csv(
        OUTPUT_DIR / path.name, index=False
    )


def write_summary(rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PPG CLEANING SUMMARY")
    lines.append("Stage: pre-filter (trim + NaN-drop + outlier-flag)")
    lines.append(f"Trim: first {TRIM_MS} ms  |  Outlier rule: |x - local_mean| > "
                 f"{OUTLIER_Z}*local_std, rolling window {ROLL_WINDOW} samples "
                 f"(~{ROLL_WINDOW/SAMPLING_HZ:.2f} s)")
    lines.append("Outliers are FLAGGED (outlier_flag column), NOT deleted.")
    lines.append("=" * 78)
    lines.append("")
    for r in rows:
        lines.append(f"### {r['file']}")
        lines.append(f"  original rows       : {r['n_original']}")
        lines.append(f"  rows dropped (NaN)  : {r['n_nan_dropped']}")
        lines.append(f"  rows dropped (2s trim): {r['n_trimmed_first_2s']}")
        lines.append(f"  final rows          : {r['n_final']}")
        lines.append(f"  final duration      : {r['duration_s']:.2f} s")
        lines.append(f"  outliers flagged    : "
                     f"any={r['outliers_any']}  "
                     f"red={r['outliers_red']}  "
                     f"ir={r['outliers_ir']}  "
                     f"green={r['outliers_green']}")
        if r["warnings"]:
            for w in r["warnings"]:
                lines.append(f"  WARNING             : {w}")
        else:
            lines.append("  WARNING             : (none)")
        lines.append("")
    lines.append("=" * 78)
    lines.append("END OF SUMMARY")
    lines.append("=" * 78)
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    files = sorted(p for p in INPUT_DIR.glob("P*.csv")
                   if not p.name.startswith("cleaning_"))
    if not files:
        raise SystemExit(f"No CSV files found in {INPUT_DIR}")

    rows = [clean_one(p) for p in files]
    write_summary(rows)
    print(f"Cleaned {len(rows)} files -> {OUTPUT_DIR}")
    print(f"Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
