# Engineering Framework for Evaluating Epidermal Melanin Attenuation in PPG

BEng Research Project — University of Manchester  
Author: Louis Thong (11425291) · Supervisor: Prof. Thomas Anthopoulos

## Introduction

This is the code for my third-year project on melanin-induced bias in 
pulse oximetry. The hardware side is a SparkFun SEN-15219 module 
(MAX30101 optical front-end + MAX32664 sensor hub) driven by a 
SparkFun RedBoard Qwiic. The point of the firmware is to disable the 
MAX32664's onboard Automatic Gain Control so that the photodiode 
returns raw 18-bit ADC counts, not the smoothed values a consumer 
device would normally output. Once you have the raw counts, the 
Python pipeline filters them, finds fiducial points on each heartbeat, 
and computes the signal-quality metrics used in Section 6 of the 
report (DC, AC, PI, SNR, a-wave amplitude, dual-wavelength SpO₂).

There's also a separate script that prototypes the three-wavelength 
spectroscopic decomposition proposed in Section 7 of the report — 
solving a 3×3 Beer-Lambert system with red, IR and the green LED to 
treat melanin as an explicit unknown rather than an unmodelled error.

## How the pieces fit together

The data flow is one direction, sensor → CSV → filtered CSV → 
fiducial CSV → summary table:
┌──────────────┐  USB    ┌────────────────┐    ┌─────────┐
│ MAX30101 +   │────────►│ capture_ppg.py │───►│ P*.csv  │
│ MAX32664 on  │ serial  │  (logging)     │    │  raw    │
│ RedBoard     │ 115200  └────────────────┘    └────┬────┘
└──────┬───────┘                                    │
│ AGC bypass via                             ▼
│ arduino_sensor.ino                  ┌──────────────┐
▼                                     │ clean_ppg.py │
raw 18-bit counts                           │ NaN drop,    │
red/IR/green                                │ 2-s trim,    │
│ outlier flag │
└──────┬───────┘
▼
┌──────────────┐
│ filter_ppg.py│
│ Butterworth  │
│ BP 0.6-3.3Hz │
└──────┬───────┘
▼
┌────────────────────┐
│ extract_fiducials  │
│ on/sp/dn/dp + APG  │
│ a-wave per beat    │
└─────────┬──────────┘
▼
┌────────────────────┐
│ compute_summary.py │
│ DC, AC, PI, SNR,   │
│ HR, SpO₂_dual      │
└─────────┬──────────┘
▼
┌────────────────────┐
│ 3-wavelength model │
│ (Section 7 proto)  │
└────────────────────┘
Each script is meant to be run on its own — they read the previous 
stage's CSV and write the next one. That way if something goes wrong 
in the middle you can re-run just the affected stage.

## What's in the repo

| File | What it does |
|---|---|
| `arduino_sensor.ino` | Arduino sketch. Sets up I²C, exits the MAX32664 bootloader, disables AGC, locks LED current, streams red/IR/green ADC counts over serial at 115200 baud. |
| `capture_ppg.py` | Reads the serial stream and writes it to a CSV. Auto-names the file P1.csv, P2.csv, etc. so you don't overwrite previous participants. |
| `clean_ppg.py` | Drops any rows with NaNs, trims the first 2 s (bootloader artefacts), and flags outliers using a rolling 5σ rule. Writes to `cleaned/`. |
| `filter_ppg.py` | 4th-order zero-phase Butterworth bandpass, 0.6–3.3 Hz. Also produces FFT plots so you can sanity-check the cardiac peak landed in the right place. |
| `extract_fiducials.py` | Finds the four fiducial points (onset, systolic peak, dicrotic notch, diastolic peak) on every beat and pulls the APG a-wave amplitude. |
| `compute_summary.py` | Aggregates everything into one row per participant: DC, AC, PI, SNR per channel, plus HR and dual-wavelength SpO₂. |
| `three_wavelength_decomposition.py` | The Section 7 prototype. Builds the 3×3 Beer-Lambert matrix and solves for melanin-corrected SpO₂. |

## Installation

### Hardware
- SparkFun RedBoard Qwiic (DEV-15123)
- SparkFun Pulse Oximeter and Heart Rate Sensor (SEN-15219)
- Qwiic cable, USB-B cable, dark enclosure for the measurement
- A laptop running Python 3.9+

### Arduino side
1. Install Arduino IDE 2.x.
2. In Library Manager, install **SparkFun Bio Sensor Hub Library**.
3. Open `arduino_sensor.ino`, select board "SparkFun RedBoard Qwiic"
   and the right COM/USB port, then upload.

If the upload works, opening the Serial Monitor at 115200 baud should 
show a banner like `# capture starting` followed by CSV rows of three 
numbers. If you see static numbers that never change, the AGC didn't 
disable properly — usually a power-cycle of the board fixes it.

### Python side
Tested on Python 3.10 and 3.11. Install the dependencies:
pip install pyserial numpy pandas scipy matplotlib pyPPG
`pyPPG` is the Charlton et al. toolbox referenced in Section 6.2 of 
the report. If you can't install it (it has some heavy dependencies), 
`extract_fiducials.py` also contains a pure-NumPy fallback that 
mirrors the same algorithm — see the docstring at the top of that 
file.

## How to run it

A full run for one participant looks like this:

```bash
# 1. Plug in the RedBoard. Make sure it's running arduino_sensor.ino.
# 2. Capture 60-120 s of data:
python capture_ppg.py --port /dev/cu.usbserial-1420
#    -> writes ~/Desktop/P1.csv (auto-numbered)

# 3. Move all participant CSVs into a folder, then run the pipeline:
python clean_ppg.py
python filter_ppg.py
python extract_fiducials.py
python compute_summary.py

# 4. Optional — try the proposed 3-wavelength model:
python three_wavelength_decomposition.py
```

`compute_summary.py` writes `results/summary.csv` (one row per 
participant) and `results/quality_flags.txt` (any values outside 
physiological ranges, e.g. HR < 40 BPM). Those two files are what 
Section 6.3's tables and Figure 6.3.1 are drawn from.

A `metadata.csv` is expected with columns `participant_id, MST` so 
that the summary script can attach the Monk Skin Tone score to each 
participant. Mine is not committed because it contains pseudonymised 
participant IDs collected under UREC ethics approval.

## Technical details

A few things worth knowing if you want to read or extend the code:

- **Sampling rate is 25 Hz.** This is what the MAX32664 actually 
  delivers in raw mode after the AGC bypass — the Arduino loop's 
  `delay(10)` sets a 100 Hz attempt, but the sensor hub decimates. 
  The filter and HR-search bands are sized for 25 Hz; if you change 
  the sketch, change `FS_HZ` in every Python script too.

- **Filter is zero-phase (`scipy.signal.filtfilt`).** Standard 
  forward filtering would shift the systolic peaks by a few samples 
  and break inter-beat interval estimation. `filtfilt` runs the 
  filter forwards and backwards so peak timing is preserved.

- **AC is defined as (systolic peak − pulse onset).** Not 
  peak-to-peak. This matters because peak-to-peak would include the 
  diastolic shoulder, which is what melanin attenuates more, giving 
  a misleadingly large AC drop. Onset-to-peak is what the 
  Beer-Lambert AC term is supposed to be.

- **SNR is computed on the cleaned (unfiltered) signal,** not the 
  filtered one. Computing SNR after filtering would be circular — 
  the filter has already killed the off-band power. The DC bin is 
  excluded so a small residual offset doesn't inflate the noise 
  estimate.

- **Dual-wavelength SpO₂ uses the textbook linear approximation:** 
  `SpO₂ = 110 − 25·R`, where R = (AC_red/DC_red) / (AC_ir/DC_ir). 
  This is the calibration printed in basic teaching materials and in 
  the AFE4403 application note; commercial devices use proprietary 
  curves that I don't have access to. Section 6.3 of the report 
  discusses why this calibration becomes unstable above MST 6.

- **Three-wavelength matrix uses Prahl's hemoglobin extinction 
  coefficients and Jacques' eumelanin law** (μ_a = 1.70×10¹² · 
  λ⁻³·⁴⁸). Sources are cited in the docstring of 
  `three_wavelength_decomposition.py`.

## Known issues and future work

- **The dual-wavelength SpO₂ becomes unreliable for MST ≥ 7.** Once 
  AC_red drops below ~50 counts the ratio is dominated by shot noise 
  rather than haemoglobin absorption. No amount of post-processing 
  fixes this; you need a brighter LED or a narrower-bandwidth source 
  (a VCSEL would be the obvious upgrade).

- **The three-wavelength model is a prototype, not validated 
  hardware.** Section 7 of the report describes the maths; this 
  script implements the inversion, but I haven't run live 
  participants through the green-channel time-division multiplexing 
  scheme that would be needed for real-time use. The MAX30101 has a 
  green LED but the SparkFun library only exposes it for heart-rate 
  mode by default.

- **No unit tests.** Each stage was verified by eye against the FFT 
  plots and the fiducial-overlay plots saved in `plots/`. For a 
  longer-running project I would add tests for the filter response 
  and the peak detector against synthetic signals.

- **Sampling rate is locked at 25 Hz** by the sensor-hub firmware. 
  Higher sampling rates would tighten fiducial timing, which matters 
  for arterial stiffness work but isn't critical for this study's 
  amplitude-based analysis.

## Acknowledgements & third-party code

- `SparkFun_Bio_Sensor_Hub_Library` — used unmodified, MIT licence.
- `pyPPG` (Goda et al., 2024) — used unmodified for fiducial 
  extraction. Cited in the report.
- `numpy`, `scipy`, `pandas`, `matplotlib` — standard scientific 
  Python stack.
- Eumelanin absorption law: Jacques, S. L. (2013), *Phys. Med. 
  Biol.* 58, R37–R61.
- Hemoglobin extinction coefficients: Scott Prahl's compilation, 
  https://omlc.org/spectra/hemoglobin/summary.html.

All other code in this repository was written by me for the project.

## Licence

Code released for academic review under the MIT licence. The dataset 
itself is not in this repository — it was collected under University 
of Manchester UREC ethics approval and contains pseudonymised 
participant data which is held separately.
