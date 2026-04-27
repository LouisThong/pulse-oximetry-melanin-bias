"""
compute_summary.py
------------------
Stage 4 of the PPG pipeline: aggregate per-participant signal-quality metrics
into a single table ``results/summary.csv``.

For each participant and each channel (red, IR, green) this script computes:

  DC      : mean of the *cleaned* (unfiltered) signal.  This is the large
            baseline light intensity reaching the photodiode when no pulse
            is present -- the average "brightness" of the tissue.
  AC      : mean peak-to-trough amplitude of the cardiac pulse, taken as
            (systolic-peak value - pulse-onset value) averaged across
            every beat in the *filtered* signal.  This is the size of
            the actual pulse.
  PI      : Perfusion Index = (AC / DC) * 100, expressed as a percent.
            Standard clinical metric of signal strength.
  SNR     : Power inside the cardiac band (0.6-3.3 Hz) divided by power
            everywhere else (DC bin excluded) in the cleaned signal.
            Reported as a linear ratio.
  HR      : Heart rate in beats per minute, taken from the dominant FFT
            peak of the *filtered IR* signal inside the plausible
            cardiac band (0.8-2.5 Hz = 48-150 BPM).  Single value per
            participant.
  a_wave  : Mean APG a-wave amplitude for the RED channel (from
            ``fiducials/{pid}.csv``).  Primary melanin-sensitive feature.

Outputs
-------
  results/summary.csv        : one row per participant, 16 columns.
  results/quality_flags.txt  : any physiologically implausible values.

Physiological sanity checks applied
-----------------------------------
  - HR must fall in [40, 180] BPM.
  - PI must fall in [0.1, 20] %.
  - DC, AC must be positive finite.
  - SNR must be non-negative finite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE       = Path("/home/claude/oximeter")
CLEAN_DIR  = BASE / "cleaned"
FILT_DIR   = BASE / "filtered"
FID_DIR    = BASE / "fiducials"
META_PATH  = BASE / "metadata.csv"
OUT_DIR    = BASE / "results"
OUT_CSV    = OUT_DIR / "summary.csv"
FLAG_PATH  = OUT_DIR / "quality_flags.txt"

FS_HZ         = 25.0
CHANNELS      = ["red", "ir", "green"]
CARDIAC_BAND  = (0.6, 3.3)                 # bandpass cut-offs (Hz)
HR_SEARCH     = (0.8, 2.5)                 # HR search window (Hz) = 48-150 BPM
HR_REF_CH     = "ir"                       # reference channel for HR

# Plausibility thresholds
HR_OK         = (40.0, 180.0)              # BPM
PI_OK         = (0.1, 20.0)                # percent


# ---------------------------------------------------------------------------
# Per-metric computations
# ---------------------------------------------------------------------------

def dc_baseline(cleaned: pd.DataFrame, ch: str) -> float:
    """Mean of the cleaned (raw-ish) channel, excluding samples flagged as
    outliers by stage 1 so a single transient spike does not bias DC."""
    s = cleaned.loc[cleaned["outlier_flag"] == 0, ch].to_numpy(dtype=float)
    if s.size == 0:
        s = cleaned[ch].to_numpy(dtype=float)
    return float(np.mean(s))


def ac_amplitude(fid: pd.DataFrame, ch: str) -> float:
    """Mean (SP-ON) amplitude across all beats in the filtered channel."""
    sub = fid[fid["channel"] == ch]
    if sub.empty:
        return float("nan")
    amp = sub["sp_amp"].to_numpy(dtype=float) - sub["on_amp"].to_numpy(dtype=float)
    amp = amp[np.isfinite(amp)]
    if amp.size == 0:
        return float("nan")
    return float(np.mean(amp))


def snr_in_band(cleaned_channel: np.ndarray, fs: float,
                band: Tuple[float, float]) -> float:
    """SNR = sum(|X(f)|^2 inside band) / sum(|X(f)|^2 outside band),
    computed on the cleaned (raw-ish) signal with its mean removed.

    Using the filtered signal would give an (artificially) huge SNR
    because the filter has already killed the off-band power -- so we
    compute SNR on the cleaned signal, where both the cardiac content
    and the noise are still present.  DC bin (f = 0) is excluded from
    both numerator and denominator.
    """
    x = np.asarray(cleaned_channel, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    mag = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    power = mag ** 2

    # Exclude DC bin so a tiny residual offset does not inflate "noise".
    analysable = freq > 0
    in_band    = (freq >= band[0]) & (freq <= band[1])
    out_band   = analysable & ~in_band

    sig  = float(power[in_band].sum())
    nois = float(power[out_band].sum())
    if nois <= 0.0:
        return float("inf")
    return sig / nois


def heart_rate_bpm(filtered_ir: np.ndarray, fs: float,
                   band: Tuple[float, float]) -> float:
    """Dominant FFT peak inside ``band`` on the filtered IR signal.
    Returns BPM (= peak_hz * 60).  NaN if no peak in band."""
    x = np.asarray(filtered_ir, dtype=float)
    x = x - np.mean(x)
    mag  = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mask = (freq >= band[0]) & (freq <= band[1])
    if not mask.any():
        return float("nan")
    sub_f, sub_m = freq[mask], mag[mask]
    pk_hz = float(sub_f[int(np.argmax(sub_m))])
    return pk_hz * 60.0


def mean_red_a_wave(fid: pd.DataFrame) -> float:
    red = fid[fid["channel"] == "red"]
    a = red["a_amp"].dropna().to_numpy(dtype=float)
    return float(np.mean(a)) if a.size else float("nan")


def spo2_dual_wavelength(ac_red: float, dc_red: float,
                          ac_ir: float,  dc_ir: float) -> float:
    """Standard two-wavelength SpO2 from the ratio-of-ratios.

        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        SpO2 ~ 110 - 25 * R   (the empirical Masimo/Nellcor-style linear
                               calibration that appears in most teaching
                               materials; correct to within ~2% in the
                               80-100% range)

    Returns NaN if any input is non-finite or non-positive.
    """
    if not all(np.isfinite([ac_red, dc_red, ac_ir, dc_ir])):
        return float("nan")
    if dc_red <= 0 or dc_ir <= 0 or ac_ir <= 0:
        return float("nan")
    R = (ac_red / dc_red) / (ac_ir / dc_ir)
    return 110.0 - 25.0 * R



# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def process_participant(pid: str) -> Dict[str, float]:
    cleaned  = pd.read_csv(CLEAN_DIR / f"{pid}.csv")
    filtered = pd.read_csv(FILT_DIR  / f"{pid}.csv")
    fid      = pd.read_csv(FID_DIR   / f"{pid}.csv")

    row: Dict[str, float] = {"participant_id": pid}

    # Heart rate: one value per participant, from the reference channel.
    row["heart_rate_bpm"] = heart_rate_bpm(
        filtered[HR_REF_CH].to_numpy(dtype=float), FS_HZ, HR_SEARCH
    )

    # Per-channel block: DC, AC, PI, SNR.
    for ch in CHANNELS:
        dc = dc_baseline(cleaned, ch)
        ac = ac_amplitude(fid, ch)
        pi = (ac / dc * 100.0) if (dc and np.isfinite(ac)) else float("nan")
        snr = snr_in_band(cleaned[ch].to_numpy(dtype=float),
                          FS_HZ, CARDIAC_BAND)

        row[f"DC_{ch}"]  = dc
        row[f"AC_{ch}"]  = ac
        row[f"PI_{ch}"]  = pi
        row[f"SNR_{ch}"] = snr

    # Red a-wave: single column, red channel only.
    row["a_wave_red"] = mean_red_a_wave(fid)

    # Dual-wavelength SpO2 from ratio-of-ratios (red / IR).
    row["SpO2_dual_wavelength"] = spo2_dual_wavelength(
        ac_red=row["AC_red"], dc_red=row["DC_red"],
        ac_ir=row["AC_ir"],   dc_ir=row["DC_ir"],
    )
    return row


def assemble_summary(meta: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for pid in meta["participant_id"]:
        rows.append(process_participant(pid))
    df = pd.DataFrame(rows)

    # Merge MST from metadata and enforce the requested column order.
    df = df.merge(meta[["participant_id", "MST"]],
                  on="participant_id", how="left")
    col_order = [
        "participant_id", "MST", "heart_rate_bpm", "SpO2_dual_wavelength",
        "DC_red",   "AC_red",   "PI_red",   "SNR_red", "a_wave_red",
        "DC_ir",    "AC_ir",    "PI_ir",    "SNR_ir",
        "DC_green", "AC_green", "PI_green", "SNR_green",
    ]
    return df[col_order]


# ---------------------------------------------------------------------------
# Plausibility
# ---------------------------------------------------------------------------

def check_plausibility(df: pd.DataFrame) -> List[str]:
    """Return a list of human-readable flags for any out-of-range values."""
    flags: List[str] = []
    for _, r in df.iterrows():
        pid = r["participant_id"]

        hr = r["heart_rate_bpm"]
        if not np.isfinite(hr) or hr < HR_OK[0] or hr > HR_OK[1]:
            flags.append(f"{pid}: heart_rate_bpm={hr:.1f} outside "
                         f"[{HR_OK[0]:.0f},{HR_OK[1]:.0f}] BPM")

        for ch in CHANNELS:
            dc  = r[f"DC_{ch}"]
            ac  = r[f"AC_{ch}"]
            pi  = r[f"PI_{ch}"]
            snr = r[f"SNR_{ch}"]
            if not np.isfinite(dc) or dc <= 0:
                flags.append(f"{pid}: DC_{ch}={dc} is non-positive")
            if not np.isfinite(ac) or ac <= 0:
                flags.append(f"{pid}: AC_{ch}={ac} is non-positive")
            if not np.isfinite(pi) or pi < PI_OK[0] or pi > PI_OK[1]:
                flags.append(f"{pid}: PI_{ch}={pi:.3f}% outside "
                             f"[{PI_OK[0]},{PI_OK[1]}]%")
            if not np.isfinite(snr) or snr < 0:
                flags.append(f"{pid}: SNR_{ch}={snr} is negative / non-finite")

    return flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPLANATION = """\
WHAT EACH METRIC MEANS -- IN PLAIN ENGLISH
==========================================

MST (Monk Skin Tone score, 1-10)
  A ten-point scale of skin tone.  1 is the lightest tone, 10 the
  darkest.  Our participants were inferred from filename suffixes
  (_MT2 -> 2, _MT3 -> 3).

heart_rate_bpm
  Heart rate in beats per minute, taken from the strongest peak in the
  FFT of the filtered IR signal within 0.8-2.5 Hz.  48-150 BPM is the
  plausible adult range used elsewhere in this pipeline.

DC_<channel>  (the baseline)
  The large, slow component of the raw signal -- essentially the
  "average brightness" the photodiode sees when no pulse is present.
  For a MAX30101 sensor the raw counts are dimensionless ADC units; the
  important thing is that DC stays stable and well away from saturation.

AC_<channel>  (the pulse amplitude)
  The size of one cardiac pulse (systolic peak minus pulse onset) in
  the band-pass filtered signal, averaged across every beat we
  detected.  Bigger = more pulsatile blood reached the sensor on each
  beat.

PI_<channel>  (Perfusion Index, %)
  PI = AC / DC * 100.  It tells you what fraction of the detected light
  is actually coming from the pulsating arterial blood versus the
  static background.  Clinically, 0.1-1 % is poor perfusion,
  1-5 % is normal fingertip perfusion, and 5-20 % is excellent.
  Outside 0.1-20 % suggests something is wrong (saturation, the
  sensor slipped, or the detector jumped to the wrong peak).

SNR_<channel>  (Signal-to-Noise Ratio, unitless)
  Ratio of the signal power inside the cardiac band (0.6-3.3 Hz) to
  the power everywhere else (excluding the DC bin), computed on the
  cleaned (non-filtered) signal.  Computed on the cleaned signal
  because computing it on the already-filtered signal would be
  meaningless (the filter has already removed the out-of-band power).
  Higher SNR means a cleaner cardiac waveform.  As a rule of thumb
  SNR > 1 means cardiac power dominates, SNR < 0.1 means the beat is
  buried in noise.

a_wave_red
  Mean amplitude of the "a" wave in the APG (second derivative) on the
  RED channel, averaged across every beat.  The a-wave is the first
  and largest positive excursion of the APG during the upstroke -- it
  measures peak acceleration of the arterial wall.  This is the
  feature most sensitive to melanin absorption: darker skin tones
  attenuate the red light so the a-wave flattens.  Tracking a_wave_red
  vs MST is the core comparison this study is set up to make.
"""


def write_flags(flags: List[str]) -> None:
    if flags:
        body = "SUSPICIOUS VALUES DETECTED:\n" + "\n".join(f"  - {f}" for f in flags)
    else:
        body = ("No suspicious values detected.\n"
                "All heart rates in 40-180 BPM, all PIs in 0.1-20 %, all\n"
                "DC/AC positive and finite, all SNRs non-negative finite.\n")
    FLAG_PATH.write_text(body + "\n\n" + EXPLANATION)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not META_PATH.exists():
        raise FileNotFoundError(f"metadata.csv not found at {META_PATH}")
    meta = pd.read_csv(META_PATH)
    if not {"participant_id", "MST"}.issubset(meta.columns):
        raise ValueError(f"metadata.csv must have columns "
                         f"participant_id, MST; got {list(meta.columns)}")

    df = assemble_summary(meta)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    # Console preview -- helpful when running interactively.
    with pd.option_context("display.max_columns", None,
                           "display.width",       200,
                           "display.float_format", "{:.3f}".format):
        print(df)

    flags = check_plausibility(df)
    write_flags(flags)
    if flags:
        print("\nFLAGS:")
        for f in flags:
            print("  -", f)
    else:
        print("\nAll values within physiological ranges.")

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {FLAG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
