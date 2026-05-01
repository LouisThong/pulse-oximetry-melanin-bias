"""
filter_ppg.py
--------------
Stage 2 of PPG pipeline: apply a zero-phase 4th-order Butterworth bandpass
(0.6-3.3 Hz) to cleaned PPG signals to isolate the cardiac AC component.

Inputs:
    cleaned/*.csv          -> columns: timestamp_ms, red, ir, green, outlier_flag
Outputs:
    filtered/*.csv         -> columns: timestamp_ms, red, ir, green
    plots/02_filtered/     -> before/after + FFT plots (one random participant)
    filtering_summary.txt  -> detected cardiac peak per participant/channel
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE          = Path("/home/claude/oximeter")
CLEAN_DIR     = BASE / "cleaned"
FILT_DIR      = BASE / "filtered"
PLOT_DIR      = BASE / "plots" / "02_filtered"
SUMMARY_PATH  = BASE / "filtering_summary.txt"

FS_HZ         = 200.0          # confirmed sampling rate
LOW_HZ, HIGH_HZ = 0.6, 3.3    # bandpass cut-offs
ORDER         = 4             # Butterworth order
CHANNELS      = ["red", "ir", "green"]
PEAK_CH       = "ir"          # primary channel for cardiac peak
PEAK_BAND_HZ  = (0.8, 2.5)    # expected HR band (48-150 BPM)
SEED          = 42            # for reproducible "random" pick

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def design_butter_bandpass(low_hz: float, high_hz: float,
                           fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter in (b, a) form.

    Normalised cut-offs = f / (fs/2).
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return b, a


def bandpass(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase filtfilt application."""
    return filtfilt(b, a, x)


def fft_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided magnitude spectrum, DC removed first."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    mag = np.abs(np.fft.rfft(x)) / n
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, mag


def peak_in_band(freq: np.ndarray, mag: np.ndarray,
                 lo: float, hi: float) -> Tuple[float, float]:
    """Return (peak_hz, peak_mag) inside [lo, hi]. Returns (nan, nan) if empty."""
    mask = (freq >= lo) & (freq <= hi)
    if not mask.any():
        return float("nan"), float("nan")
    sub_f, sub_m = freq[mask], mag[mask]
    i = int(np.argmax(sub_m))
    return float(sub_f[i]), float(sub_m[i])


def clear_peak(freq: np.ndarray, mag: np.ndarray, band: Tuple[float, float],
               prominence_ratio: float = 2.0) -> bool:
    """Heuristic: the in-band peak must be >= prominence_ratio * the median
    magnitude in the rest of the analysable range (0.1 - fs/2 Hz)."""
    in_band = (freq >= band[0]) & (freq <= band[1])
    analysable = (freq >= 0.1)  # skip DC drift bin
    off_band = analysable & ~in_band
    if not in_band.any() or not off_band.any():
        return False
    peak = mag[in_band].max()
    baseline = np.median(mag[off_band]) + 1e-12
    return peak >= prominence_ratio * baseline


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def filter_all() -> Tuple[List[str], Dict[str, Dict[str, Tuple[float, float]]], List[str]]:
    """Filter every CSV in cleaned/. Returns (participants, peaks, flagged)."""
    FILT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    b, a = design_butter_bandpass(LOW_HZ, HIGH_HZ, FS_HZ, ORDER)

    files = sorted(p for p in CLEAN_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {CLEAN_DIR}")

    participants: List[str] = []
    peaks: Dict[str, Dict[str, Tuple[float, float]]] = {}
    flagged: List[str] = []

    for f in files:
        pid = f.stem  # e.g. P1_MT2
        df = pd.read_csv(f)
        # Keep timestamp as-is; filter each channel independently.
        out = pd.DataFrame({"timestamp_ms": df["timestamp_ms"].values})
        per_channel_peaks: Dict[str, Tuple[float, float]] = {}
        for ch in CHANNELS:
            x = df[ch].to_numpy(dtype=float)
            y = bandpass(x, b, a)
            out[ch] = y

            freq, mag = fft_spectrum(y, FS_HZ)
            pk_hz, _ = peak_in_band(freq, mag, *PEAK_BAND_HZ)
            pk_bpm = pk_hz * 60.0 if np.isfinite(pk_hz) else float("nan")
            per_channel_peaks[ch] = (pk_hz, pk_bpm)

        out.to_csv(FILT_DIR / f.name, index=False)

        # Flag if the primary channel (IR) has no clear peak.
        y_ir = out[PEAK_CH].to_numpy()
        f_ir, m_ir = fft_spectrum(y_ir, FS_HZ)
        if not clear_peak(f_ir, m_ir, PEAK_BAND_HZ):
            flagged.append(pid)

        participants.append(pid)
        peaks[pid] = per_channel_peaks

    return participants, peaks, flagged


def make_plots_for(pid: str) -> None:
    """Before/after time-domain and FFT spectra for one participant."""
    raw_df = pd.read_csv(CLEAN_DIR / f"{pid}.csv")
    filt_df = pd.read_csv(FILT_DIR / f"{pid}.csv")

    t = raw_df["timestamp_ms"].to_numpy() / 1000.0
    t = t - t[0]

    # --- Time-domain: 3 channels, raw vs filtered ---
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=False)
    colors = {"red": "#d1495b", "ir": "#2e4057", "green": "#4caf50"}
    for i, ch in enumerate(CHANNELS):
        axes[i, 0].plot(t, raw_df[ch], color=colors[ch], lw=0.9)
        axes[i, 0].set_title(f"{ch.upper()}  —  RAW (cleaned)")
        axes[i, 0].set_ylabel("counts")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(t, filt_df[ch], color=colors[ch], lw=0.9)
        axes[i, 1].set_title(f"{ch.upper()}  —  FILTERED (0.6–3.3 Hz BP)")
        axes[i, 1].set_ylabel("counts (AC)")
        axes[i, 1].grid(alpha=0.3)
        axes[i, 1].axhline(0, color="k", lw=0.5)

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{pid}: raw vs band-pass filtered PPG", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_DIR / f"{pid}_time_raw_vs_filtered.png", dpi=130)
    plt.close(fig)

    # --- FFT: 3 channels, raw vs filtered ---
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for i, ch in enumerate(CHANNELS):
        for j, src in enumerate((raw_df, filt_df)):
            freq, mag = fft_spectrum(src[ch].to_numpy(dtype=float), FS_HZ)
            label = "RAW" if j == 0 else "FILTERED"
            axes[i, j].plot(freq, mag, color=colors[ch], lw=1.0)
            axes[i, j].axvspan(*PEAK_BAND_HZ, color="orange", alpha=0.15,
                                label="expected HR band (0.8–2.5 Hz)")
            axes[i, j].axvspan(LOW_HZ, HIGH_HZ, color="green", alpha=0.08,
                                label="filter pass-band (0.6–3.3 Hz)")
            pk_hz, _ = peak_in_band(freq, mag, *PEAK_BAND_HZ)
            if np.isfinite(pk_hz):
                axes[i, j].axvline(pk_hz, ls="--", color="k", lw=0.8,
                                    label=f"peak @ {pk_hz:.2f} Hz ({pk_hz*60:.1f} BPM)")
            axes[i, j].set_title(f"{ch.upper()} — {label}")
            axes[i, j].set_ylabel("|X(f)|")
            axes[i, j].grid(alpha=0.3)
            axes[i, j].set_xlim(0, FS_HZ / 2)
            if i == 0 and j == 1:
                axes[i, j].legend(loc="upper right", fontsize=7)

    axes[-1, 0].set_xlabel("frequency (Hz)")
    axes[-1, 1].set_xlabel("frequency (Hz)")
    fig.suptitle(f"{pid}: FFT spectra  —  raw vs filtered", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_DIR / f"{pid}_fft_raw_vs_filtered.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

EXPLANATION = """\
FILTERING — IN PLAIN ENGLISH
----------------------------
A PPG (photoplethysmography) signal is the light reflected from your skin as
your heart pumps blood. It contains three things mixed together:

  1. A large DC component (how much light comes back on average).
  2. A slow drift (breathing, hand movement, sensor warm-up) — below ~0.5 Hz.
  3. The small pulse we care about, roughly one beat per second — 0.8–2.5 Hz
     for adults (48–150 BPM).
  4. High-frequency noise from the sensor and ambient light — above ~4 Hz.

A BAND-PASS FILTER is a "frequency gate": it only lets through frequencies
inside a chosen range and suppresses everything outside. We use cut-offs of
0.6 Hz (lower) and 3.3 Hz (upper), which is wide enough to preserve the
cardiac pulse and its shape (harmonics) but blocks slow drift below and
sensor noise above.

BUTTERWORTH is a family of filters with a maximally flat pass-band — it does
not add ripples to the signal inside the band we keep. A 4th-order design
gives a steep enough roll-off to cleanly reject the unwanted bands without
distorting the pulse shape.

ZERO-PHASE FILTERING via scipy's `filtfilt` runs the filter forwards and
then backwards over the signal. Ordinary filters delay the signal slightly;
filtfilt cancels that delay, so peaks in the filtered signal line up with
the real peaks in time. That matters later when we want to measure
inter-beat intervals accurately.

After filtering, the signal is centred around zero (the DC is gone) and
shows the pulsatile "AC" component only. Taking the FFT of that signal
reveals a sharp peak at the heart-rate frequency, which we verify falls
between 0.8 and 2.5 Hz for each participant.
"""


def write_summary(participants: List[str],
                  peaks: Dict[str, Dict[str, Tuple[float, float]]],
                  flagged: List[str]) -> None:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("PPG FILTERING SUMMARY")
    lines.append(f"Filter : 4th-order Butterworth band-pass, "
                 f"{LOW_HZ:.2f}–{HIGH_HZ:.2f} Hz, zero-phase (filtfilt)")
    lines.append(f"Sampling rate : {FS_HZ:.1f} Hz")
    lines.append(f"Expected HR band : {PEAK_BAND_HZ[0]:.1f}–{PEAK_BAND_HZ[1]:.1f} Hz "
                 f"({PEAK_BAND_HZ[0]*60:.0f}–{PEAK_BAND_HZ[1]*60:.0f} BPM)")
    lines.append("=" * 78)
    lines.append("")

    header = f"{'participant':<12}{'channel':<8}{'peak (Hz)':>12}{'peak (BPM)':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for pid in participants:
        for ch in CHANNELS:
            pk_hz, pk_bpm = peaks[pid][ch]
            hz_s  = f"{pk_hz:.3f}"  if np.isfinite(pk_hz)  else "   n/a"
            bpm_s = f"{pk_bpm:.1f}" if np.isfinite(pk_bpm) else "   n/a"
            lines.append(f"{pid:<12}{ch:<8}{hz_s:>12}{bpm_s:>14}")
        lines.append("")

    lines.append("FLAGGED PARTICIPANTS (no clear cardiac peak on IR):")
    if flagged:
        for p in flagged:
            lines.append(f"  - {p}")
    else:
        lines.append("  (none — every participant shows a clear peak in 0.8–2.5 Hz)")
    lines.append("")
    lines.append(EXPLANATION)
    lines.append("=" * 78)
    lines.append("END OF SUMMARY")
    lines.append("=" * 78)

    SUMMARY_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    random.seed(SEED)
    participants, peaks, flagged = filter_all()

    if flagged:
        # Flag and stop *before* producing deliverables, so the user can decide.
        print("FLAGGED (no clear cardiac peak on IR):", ", ".join(flagged))
        print("Stopping before plots/summary so you can review.")
        return 2

    chosen = random.choice(participants)
    print(f"Randomly chose {chosen} for comparison plots.")
    make_plots_for(chosen)

    write_summary(participants, peaks, flagged)
    print(f"Wrote {len(participants)} filtered CSVs -> {FILT_DIR}")
    print(f"Wrote plots -> {PLOT_DIR}")
    print(f"Wrote report -> {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
