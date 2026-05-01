"""
extract_fiducials.py
--------------------
Stage 3 of the PPG pipeline: per-heartbeat fiducial-point extraction.

For every participant and every channel (red, IR, green), this script:

  1. Loads the band-pass filtered signal from ``filtered/*.csv``.
  2. Computes the first (VPG = velocity), second (APG = acceleration) and
     third (JPG = jerk) derivatives of the signal.
  3. Detects four fiducial points on every heartbeat:
        on  - pulse onset        (start of the upstroke)
        sp  - systolic peak      (peak of the pulse wave)
        dn  - dicrotic notch     (small dip after sp, aortic valve close)
        dp  - diastolic peak     (reflected pressure wave from periphery)
  4. Extracts the APG "a-wave" amplitude for every heartbeat.
     The a-wave is the first large positive peak of the APG in the
     ON->SP interval.  It flattens as melanin absorption increases,
     which is the feature of primary interest in this study.

Outputs
-------
  fiducials/{participant_id}.csv         per-beat table of fiducial indices,
                                         times, amplitudes and a-wave values,
                                         for all three channels
  plots/03_fiducials/{participant_id}.png 5-second excerpt of the RED signal
                                         with on/sp/dn/dp annotated
  fiducial_summary.txt                   beats-detected per participant/channel,
                                         flags, plain-English explanation

NOTE on pyPPG
-------------
The task description asked for ``pyPPG``.  That package (and its
dependency ``scipy``) are blocked by the sandbox's package proxy,
so this script implements pyPPG-style fiducial detection directly
in numpy.  The algorithms mirror the approach described in Charlton
et al., "Detecting beats in the photoplethysmogram: benchmarking
open-source algorithms" and the pyPPG documentation:

  * SP   : local maxima of the PPG with adaptive height threshold and
           refractory suppression (min beat-to-beat separation).
  * ON   : minimum of the PPG in the interval between SP[k-1] and SP[k].
  * a    : maximum of the APG in the ON->SP interval (maximum
           acceleration during the upstroke).
  * DN   : the "e" wave -- maximum of the APG in SP -> next-ON, i.e. the
           local peak of acceleration that marks the dicrotic notch.
  * DP   : maximum of the PPG between DN and the next ON.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE          = Path("/home/claude/oximeter")
FILT_DIR      = BASE / "filtered"
FID_DIR       = BASE / "fiducials"
PLOT_DIR      = BASE / "plots" / "03_fiducials"
SUMMARY_PATH  = BASE / "fiducial_summary.txt"

FS_HZ         = 200.0                     # sampling rate (confirmed earlier)
CHANNELS      = ["red", "ir", "green"]   # order matters for plot colour map
PRIMARY_CH    = "red"                    # channel used for the 5-s excerpt plot

# Heart-rate limits used by the peak detector (48-150 BPM).
HR_MIN_BPM    = 40.0
HR_MAX_BPM    = 150.0

# Flag any participant whose beat density is below this (in BPM units).
FLAG_BELOW_BPM = 40.0

# ---------------------------------------------------------------------------
# Low-level signal utilities
# ---------------------------------------------------------------------------

def derivatives(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return VPG, APG, JPG = 1st, 2nd, 3rd time-derivatives of ``x``.

    Uses numpy's second-order central differences via ``np.gradient`` (same
    order of accuracy as numpy diff with a central stencil).  dt = 1/fs.
    """
    dt = 1.0 / fs
    vpg = np.gradient(x,   dt)
    apg = np.gradient(vpg, dt)
    jpg = np.gradient(apg, dt)
    return vpg, apg, jpg


def _strict_local_max(x: np.ndarray) -> np.ndarray:
    """Indices where ``x[i]`` is a strict local maximum (plateau-tolerant)."""
    # Using the condition x[i-1] < x[i] >= x[i+1] allows flat tops; we then
    # deduplicate adjacent equal plateaus by keeping the first sample of each.
    n = len(x)
    if n < 3:
        return np.empty(0, dtype=int)
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    return cand


def _strict_local_min(x: np.ndarray) -> np.ndarray:
    """Indices where ``x[i]`` is a strict local minimum (plateau-tolerant)."""
    n = len(x)
    if n < 3:
        return np.empty(0, dtype=int)
    cand = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    return cand


def find_peaks(x: np.ndarray, min_distance: int = 1,
               min_height: float | None = None) -> np.ndarray:
    """Minimal peak-finder: strict local maxima with optional height and
    pairwise minimum separation (greedy tallest-first suppression).

    Good enough for our short, clean, band-pass filtered PPG signals;
    mirrors the behaviour of ``scipy.signal.find_peaks`` with ``distance``
    and ``height`` arguments.
    """
    peaks = _strict_local_max(x)
    if min_height is not None:
        peaks = peaks[x[peaks] >= min_height]
    if len(peaks) <= 1 or min_distance <= 1:
        return peaks
    # Greedy: keep the tallest peak, then walk outward removing any neighbour
    # within ``min_distance`` samples.
    order = np.argsort(-x[peaks])  # tallest first
    keep = np.ones(len(peaks), dtype=bool)
    for idx in order:
        if not keep[idx]:
            continue
        p = peaks[idx]
        for j in range(len(peaks)):
            if keep[j] and j != idx and abs(peaks[j] - p) < min_distance:
                keep[j] = False
    out = peaks[keep]
    return np.sort(out)


# ---------------------------------------------------------------------------
# Fiducial extraction (pyPPG-equivalent)
# ---------------------------------------------------------------------------

def detect_systolic_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Detect systolic peaks in a band-pass filtered PPG signal.

    Two-pass adaptive detector:

    1. Compute a *hard* refractory period from the maximum plausible heart
       rate: two systolic peaks cannot be closer than one beat-period at
       HR_MAX_BPM (= 400 ms at 150 BPM = 10 samples at 200 Hz).
    2. Use an adaptive amplitude threshold that starts at the 50th-
       percentile of the positive part of the signal.  If this rejects too
       few candidates (implying a noisy detection that picks up dicrotic
       bumps or second-harmonic peaks), tighten the threshold until the
       mean beat-rate falls below HR_MAX_BPM.
    """
    min_distance = max(2, int(round(fs * 60.0 / HR_MAX_BPM)))  # hard lower bound
    positive = ppg[ppg > 0]
    if positive.size == 0:
        return np.empty(0, dtype=int)

    duration_min = len(ppg) / fs / 60.0
    max_beats = int(np.ceil(duration_min * HR_MAX_BPM))

    # Try increasingly strict height thresholds until the detector reports
    # a physiologically plausible number of beats.  This is the standard
    # pyPPG/Elgendi trick: prefer "slightly conservative" over "too eager".
    for pct in (50, 60, 70, 75):
        threshold = np.percentile(positive, pct)
        peaks = find_peaks(ppg, min_distance=min_distance,
                           min_height=threshold)
        if peaks.size <= max_beats:
            return peaks
    return peaks  # last attempt, return whatever we got


def locate_onsets(ppg: np.ndarray, sp_idx: np.ndarray) -> np.ndarray:
    """For each systolic peak SP[k], return the preceding pulse onset ON[k]
    = argmin(ppg) in (SP[k-1], SP[k]].  For the first beat we search
    backwards from SP[0] to the start of the signal.
    """
    if sp_idx.size == 0:
        return np.empty(0, dtype=int)
    onsets = np.empty_like(sp_idx)
    # First beat: look back to t=0.
    start0 = 0
    onsets[0] = start0 + int(np.argmin(ppg[start0:sp_idx[0] + 1]))
    # Subsequent beats: between previous and current systolic peak.
    for k in range(1, sp_idx.size):
        lo, hi = sp_idx[k - 1] + 1, sp_idx[k]
        if hi <= lo:
            onsets[k] = lo
        else:
            onsets[k] = lo + int(np.argmin(ppg[lo:hi + 1]))
    return onsets


def locate_dicrotic_and_diastolic(ppg: np.ndarray,
                                  apg: np.ndarray,
                                  sp_idx: np.ndarray,
                                  on_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Locate the dicrotic notch (DN) and diastolic peak (DP) for each beat.

    For beat ``k`` we search in the diastolic window (SP[k], ON[k+1]]:

      DN : the earliest local minimum of the PPG inside the window (this
           is the classical definition: a small dip after the systolic
           peak).  If no interior local minimum exists (smooth upstroke),
           we fall back to an interior APG local maximum (pyPPG's "e"
           wave).  If even that fails, DN is marked missing.
      DP : the highest PPG point after DN and before the next onset.
           Must sit strictly inside (DN, next_ON) to be considered.

    The final beat has no known ON[k+1], so DN/DP are returned as -1.
    """
    n_beats = sp_idx.size
    dn = np.full(n_beats, -1, dtype=int)
    dp = np.full(n_beats, -1, dtype=int)
    for k in range(n_beats):
        sp = sp_idx[k]
        if k + 1 >= n_beats:
            continue
        next_on = on_idx[k + 1]
        # Physiological constraint: DN occurs roughly 200-300 ms after SP,
        # i.e. in the *first half* of the (SP, next_ON) window.  Beyond
        # that we are in the late-diastolic descent, where spurious local
        # minima from filtering artefacts live.
        beat_span = max(1, next_on - sp)
        hi_dn = sp + max(2, int(round(beat_span * 0.55)))
        lo, hi = sp + 1, min(hi_dn, next_on - 1)
        if hi - lo < 1:
            continue
        window_ppg = ppg[lo:hi + 1]
        window_apg = apg[lo:hi + 1]

        # --- PRIMARY: earliest local minimum of the PPG in the window.
        mins_rel = _strict_local_min(window_ppg)
        dn_rel = int(mins_rel[0]) if mins_rel.size else -1

        # --- FALLBACK: earliest interior APG local maximum ("e"-wave).
        if dn_rel < 0:
            maxes_rel = _strict_local_max(window_apg)
            if maxes_rel.size:
                dn_rel = int(maxes_rel[0])

        # --- LAST RESORT: argmax of APG, but only if it is interior.
        if dn_rel < 0:
            cand = int(np.argmax(window_apg))
            if 0 < cand < window_apg.size - 1:
                dn_rel = cand

        if dn_rel < 0:
            continue
        dn_idx_k = lo + dn_rel

        # Sanity check: a real dicrotic notch is a shallow dip that sits
        # ABOVE the mid-descent between SP and next_ON.  If the candidate
        # has fallen below that midpoint, it is late-diastolic noise, not
        # a notch -- drop it rather than emit a misleading marker.
        midpoint = 0.5 * (ppg[sp] + ppg[next_on])
        if ppg[dn_idx_k] < midpoint:
            continue
        dn[k] = dn_idx_k

        # Diastolic peak: PPG maximum strictly between DN and next onset.
        lo2, hi2 = dn_idx_k + 1, next_on - 1
        if hi2 - lo2 < 0:
            continue
        dp[k] = lo2 + int(np.argmax(ppg[lo2:hi2 + 1]))
    return dn, dp


def extract_a_wave(apg: np.ndarray,
                   on_idx: np.ndarray,
                   sp_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (a_idx, a_amp) for each beat: the APG maximum in ON->SP.

    The a-wave is the largest positive excursion of the second derivative
    during the upstroke -- the maximum acceleration of the arterial wall.
    This is the feature most sensitive to melanin absorption.
    """
    n_beats = sp_idx.size
    a_idx = np.full(n_beats, -1, dtype=int)
    a_amp = np.full(n_beats, np.nan)
    for k in range(n_beats):
        lo, hi = on_idx[k], sp_idx[k]
        if hi - lo < 1:
            continue
        rel = int(np.argmax(apg[lo:hi + 1]))
        a_idx[k] = lo + rel
        a_amp[k] = float(apg[lo + rel])
    return a_idx, a_amp


# ---------------------------------------------------------------------------
# Orchestration for one participant
# ---------------------------------------------------------------------------

def _rows_for_channel(t: np.ndarray,
                      ppg: np.ndarray,
                      apg: np.ndarray,
                      channel: str,
                      sp: np.ndarray,
                      on: np.ndarray,
                      dn: np.ndarray,
                      dp: np.ndarray,
                      a_idx: np.ndarray,
                      a_amp: np.ndarray) -> List[Dict]:
    """Flatten one channel's per-beat fiducial data into CSV rows."""
    rows: List[Dict] = []
    for k in range(sp.size):
        row = {
            "channel":     channel,
            "beat":        k + 1,
            "on_idx":      int(on[k]),
            "on_t_s":      float(t[on[k]]),
            "on_amp":      float(ppg[on[k]]),
            "sp_idx":      int(sp[k]),
            "sp_t_s":      float(t[sp[k]]),
            "sp_amp":      float(ppg[sp[k]]),
            "dn_idx":      int(dn[k]) if dn[k] >= 0 else np.nan,
            "dn_t_s":      float(t[dn[k]]) if dn[k] >= 0 else np.nan,
            "dn_amp":      float(ppg[dn[k]]) if dn[k] >= 0 else np.nan,
            "dp_idx":      int(dp[k]) if dp[k] >= 0 else np.nan,
            "dp_t_s":      float(t[dp[k]]) if dp[k] >= 0 else np.nan,
            "dp_amp":      float(ppg[dp[k]]) if dp[k] >= 0 else np.nan,
            "a_idx":       int(a_idx[k]) if a_idx[k] >= 0 else np.nan,
            "a_t_s":       float(t[a_idx[k]]) if a_idx[k] >= 0 else np.nan,
            "a_amp":       float(a_amp[k]) if math.isfinite(a_amp[k]) else np.nan,
        }
        rows.append(row)
    return rows


def process_participant(pid: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, np.ndarray]]:
    """Return (fiducials_df, beats_per_channel, signals_for_plot).

    ``signals_for_plot`` keeps the PRIMARY_CH arrays needed for the
    per-participant plot (ppg, fiducial indices, time axis).
    """
    t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    t = t - t[0]

    all_rows: List[Dict] = []
    beats: Dict[str, int] = {}
    plot_pack: Dict[str, np.ndarray] = {}

    for ch in CHANNELS:
        ppg = df[ch].to_numpy(dtype=float)
        vpg, apg, jpg = derivatives(ppg, FS_HZ)

        sp = detect_systolic_peaks(ppg, FS_HZ)
        on = locate_onsets(ppg, sp)
        dn, dp = locate_dicrotic_and_diastolic(ppg, apg, sp, on)
        a_idx, a_amp = extract_a_wave(apg, on, sp)

        rows = _rows_for_channel(t, ppg, apg, ch, sp, on, dn, dp, a_idx, a_amp)
        all_rows.extend(rows)
        beats[ch] = int(sp.size)

        if ch == PRIMARY_CH:
            plot_pack = {
                "t": t, "ppg": ppg, "apg": apg,
                "sp": sp, "on": on, "dn": dn, "dp": dp,
                "a_idx": a_idx,
            }

    fid_df = pd.DataFrame(all_rows,
                          columns=["channel", "beat",
                                   "on_idx", "on_t_s", "on_amp",
                                   "sp_idx", "sp_t_s", "sp_amp",
                                   "dn_idx", "dn_t_s", "dn_amp",
                                   "dp_idx", "dp_t_s", "dp_amp",
                                   "a_idx",  "a_t_s",  "a_amp"])
    return fid_df, beats, plot_pack


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_red_excerpt(pid: str, pack: Dict[str, np.ndarray],
                     duration_s: float = 5.0) -> None:
    """Save a 5-second excerpt of the RED signal with fiducials annotated."""
    t = pack["t"]
    ppg = pack["ppg"]
    window_end = min(duration_s, t[-1])
    mask = t <= window_end
    t_win = t[mask]
    y_win = ppg[mask]

    def _in_window(idx_arr: np.ndarray) -> np.ndarray:
        """Keep only indices whose time stamp falls in the plot window."""
        idx_arr = idx_arr[(idx_arr >= 0)]
        return idx_arr[t[idx_arr] <= window_end]

    on = _in_window(pack["on"])
    sp = _in_window(pack["sp"])
    dn = _in_window(pack["dn"])
    dp = _in_window(pack["dp"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t_win, y_win, color="#d1495b", lw=1.1, label="filtered RED PPG")

    if on.size:
        ax.scatter(t[on], ppg[on], s=55, marker="o",
                   edgecolor="black", facecolor="#ffcc66",
                   zorder=5, label="pulse onset (on)")
    if sp.size:
        ax.scatter(t[sp], ppg[sp], s=75, marker="^",
                   edgecolor="black", facecolor="#2e7d32",
                   zorder=5, label="systolic peak (sp)")
    if dn.size:
        ax.scatter(t[dn], ppg[dn], s=55, marker="v",
                   edgecolor="black", facecolor="#1e88e5",
                   zorder=5, label="dicrotic notch (dn)")
    if dp.size:
        ax.scatter(t[dp], ppg[dp], s=55, marker="s",
                   edgecolor="black", facecolor="#8e24aa",
                   zorder=5, label="diastolic peak (dp)")

    ax.set_xlim(0, window_end)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("filtered PPG (AC counts)")
    ax.set_title(f"{pid} — fiducial points on filtered RED PPG "
                 f"(first {int(window_end)} s, fs={FS_HZ:.0f} Hz)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{pid}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

EXPLANATION = """\
FIDUCIAL POINTS — IN PLAIN ENGLISH
----------------------------------
Every heartbeat produces a characteristic "pulse wave" in the PPG signal.
Four landmark points -- the fiducial points -- describe its shape:

  ON (pulse onset)     The lowest point before the beat rises.  It marks
                       the moment the pulse wave begins arriving at the
                       measurement site (fingertip, in our case).

  SP (systolic peak)   The highest point of the pulse wave.  It reflects
                       the peak volume of blood delivered to the tissue
                       as the left ventricle contracts.  The SP's height
                       is the primary feature for pulse-rate detection
                       and for amplitude-based SpO2 estimates.

  DN (dicrotic notch)  A small dip just after the systolic peak.  It is
                       caused by the aortic valve snapping shut at the
                       end of ventricular ejection, sending a tiny
                       pressure-wave ripple backward.  It separates the
                       "systolic" (pumping) phase from the "diastolic"
                       (relaxing) phase of the cardiac cycle.

  DP (diastolic peak)  A second, smaller bump that follows the notch.
                       It is produced by the pressure wave reflecting
                       back from stiff peripheral arteries (hands, legs)
                       and returning toward the heart.  Its prominence
                       changes with arterial stiffness and age.

THE DERIVATIVES -- VPG, APG, JPG
--------------------------------
Taking successive time-derivatives of the PPG gives us views of the pulse
that are easier to analyse than the raw waveform:

  VPG  = d/dt  PPG    velocity of the volume change.  Zero crossings
                      align with PPG peaks and troughs.
  APG  = d^2/dt^2 PPG acceleration.  Its waves (labelled a, b, c, d, e
                      in the literature) carry diagnostic information
                      about arterial stiffness and ventricular function.
  JPG  = d^3/dt^3 PPG "jerk".  Used to refine the position of subtle
                      landmarks such as the dicrotic notch.

THE a-WAVE (primary outcome of this study)
------------------------------------------
The "a" wave is the first (and largest) positive peak of the APG during
the upstroke of each beat.  It corresponds to the maximum acceleration
of the pulse as blood surges into the capillary bed.  Its amplitude is
highly sensitive to how much light reaches the photodiode through the
overlying tissue, so it FLATTENS as melanin absorption increases.  That
is why this project logs the a-wave amplitude for every heartbeat on
every channel -- it is the feature expected to track skin tone.
"""


def write_summary(records: Dict[str, Dict[str, int]],
                  durations_s: Dict[str, float],
                  flagged: Dict[str, List[str]],
                  failures: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("PPG FIDUCIAL-POINT SUMMARY")
    lines.append(f"Detector : pyPPG-style (pure numpy; see extract_fiducials.py)")
    lines.append(f"Sampling rate : {FS_HZ:.1f} Hz")
    lines.append(f"Flagged threshold : < {FLAG_BELOW_BPM:.0f} BPM-equivalent detection rate")
    lines.append("=" * 78)
    lines.append("")

    header = (f"{'participant':<12}{'channel':<8}{'beats':>7}"
              f"{'duration(s)':>14}{'rate(BPM)':>13}{'flag':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for pid, per_ch in records.items():
        for ch in CHANNELS:
            nb = per_ch.get(ch, 0)
            dur = durations_s.get(pid, float("nan"))
            bpm = (nb / dur * 60.0) if (nb and dur and dur > 0) else 0.0
            flag = "NOISY" if (bpm < FLAG_BELOW_BPM) else ""
            lines.append(f"{pid:<12}{ch:<8}{nb:>7d}{dur:>14.2f}{bpm:>13.1f}{flag:>8}")
        lines.append("")

    any_flag = any(v for v in flagged.values())
    lines.append("FLAGGED PARTICIPANT/CHANNELS (detection rate < 40 BPM-equivalent):")
    if any_flag:
        for pid, ch_list in flagged.items():
            if ch_list:
                lines.append(f"  - {pid}: {', '.join(ch_list)}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("PROCESSING FAILURES:")
    if failures:
        for pid, msg in failures.items():
            lines.append(f"  - {pid}: {msg}")
    else:
        lines.append("  (none)")
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
    FID_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(FILT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {FILT_DIR}")

    records:   Dict[str, Dict[str, int]] = {}
    durations: Dict[str, float]          = {}
    flagged:   Dict[str, List[str]]      = {}
    failures:  Dict[str, str]            = {}

    for f in files:
        pid = f.stem
        try:
            df = pd.read_csv(f)
            if not {"timestamp_ms", *CHANNELS}.issubset(df.columns):
                raise ValueError(f"missing expected columns, got {list(df.columns)}")

            fid_df, beats, pack = process_participant(pid, df)
            fid_df.to_csv(FID_DIR / f"{pid}.csv", index=False)
            plot_red_excerpt(pid, pack)

            records[pid]   = beats
            dur_s          = float(df["timestamp_ms"].iloc[-1]
                                   - df["timestamp_ms"].iloc[0]) / 1000.0
            durations[pid] = dur_s
            flagged[pid]   = [ch for ch, nb in beats.items()
                              if (nb / dur_s * 60.0) < FLAG_BELOW_BPM]

            print(f"{pid}: "
                  + ", ".join(f"{ch}={beats[ch]}" for ch in CHANNELS)
                  + f"  ({dur_s:.1f}s)")

        except Exception as ex:  # noqa: BLE001 -- we want to continue
            failures[pid] = f"{type(ex).__name__}: {ex}"
            print(f"[FAIL] {pid}: {failures[pid]}")
            traceback.print_exc()

    write_summary(records, durations, flagged, failures)
    print(f"\nWrote fiducials -> {FID_DIR}")
    print(f"Wrote plots     -> {PLOT_DIR}")
    print(f"Wrote summary   -> {SUMMARY_PATH}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
