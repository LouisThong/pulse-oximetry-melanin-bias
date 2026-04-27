# Engineering Framework for Evaluating Epidermal Melanin Attenuation in PPG

Using the MAX30101 optical sensor and MAX32664 biometric hub.

BEng Research Project — Department of Electrical and Electronic Engineering, University of Manchester (2025/26).

## Introduction

This repository contains the full firmware and signal-processing pipeline developed for a research project investigating melanin-induced attenuation bias in pulse oximetry across different skin tones. Standard two-wavelength pulse oximeters systematically overestimate blood oxygen saturation (SpO2) in darker skin because epidermal melanin absorbs red light (660 nm) far more strongly than near-infrared light (940 nm), distorting the Ratio of Ratios on which the calibration depends.

The Arduino sketch configures a SparkFun SEN-15219 module (MAX30101 + MAX32664) to bypass the on-board Automatic Gain Control (AGC) and stream raw 18-bit ADC photon counts to the host. The Python pipeline then logs the data, cleans it, band-pass filters the signal, extracts cardiac fiducial points (pulse onset, systolic peak, dicrotic notch, diastolic peak, plus the APG a-wave), and produces a per-participant signal-quality summary. A final stage implements the proposed three-wavelength spectroscopic decomposition that adds the 527 nm green channel as a third equation in order to algebraically eliminate melanin from the SpO2 calculation.

## Contextual Overview

The system has three layers:

1. **Sensor layer** — MAX30101 emits red (660 nm), infrared (940 nm) and green (527 nm) light through a single photodiode aperture. The MAX32664 hub manages the optical front end over an internal I2C bus.
2. **Microcontroller layer** — A SparkFun RedBoard Qwiic (Arduino-compatible) acts as I2C master, runs arduino_sensor.ino, performs the AGC-bypass startup handshake (RSTN/MFIO toggle, mode write, AGC disable, static LED current lock), and forwards raw red/IR/green ADC counts over USB serial at 115 200 baud.
3. **Host layer (Python)** — A five-stage pipeline consumes the serial stream and converts raw counts into signal-quality metrics:

~~~
arduino_sensor.ino  --USB serial-->  capture_ppg.py    (raw P#.csv)
                                            |
                                            v
                                      clean_ppg.py     (cleaned/)
                                            |
                                            v
                                      filter_ppg.py    (filtered/)
                                            |
                                            v
                                  extract_fiducials.py (fiducials/)
                                            |
                                            v
                                   compute_summary.py  (results/summary.csv)
                                            |
                                            v
                          three_wavelength_decomposition.py
                                            |
                                            v
                                  SpO2_corrected per participant
~~~

## Installation

### Hardware required
- SparkFun RedBoard Qwiic (DEV-15123) or any Arduino Uno-compatible board with I2C
- SparkFun Pulse Oximeter and Heart Rate Sensor — MAX30101 + MAX32664 (SEN-15219)
- Qwiic cable, USB-B cable
- A dark enclosure to exclude ambient light during recording

### Arduino-side setup
1. Install Arduino IDE (2.x or later)
2. Install the **SparkFun Bio Sensor Hub Library** via Library Manager
3. Open arduino_sensor.ino, select the correct board and port, and upload

### Python-side setup
Python 3.9 or later. From the repository root:

~~~
pip install pyserial numpy pandas scipy matplotlib pyPPG
~~~

Note: extract_fiducials.py includes a numpy fallback for environments where pyPPG cannot be installed; the algorithms mirror the pyPPG reference implementation (Aboy++ beat detection plus derivative-based fiducial logic).

## How to run

The pipeline is run stage-by-stage. Each stage reads the previous stage's output.

### 1. Record a participant
With the Arduino sketch uploaded and the sensor placed on the index finger inside the dark enclosure:

~~~
python3 capture_ppg.py --port /dev/cu.usbserial-1420
~~~

This auto-names the file P1.csv, P2.csv, ... on the Desktop. Use --out to override the path. The capture window is approximately 120 seconds (the sketch self-terminates).

### 2. Clean the raw recordings
Drops NaN rows, trims the first 2 s of bootloader artifacts, and flags samples beyond 5 sigma of a centred 51-sample rolling window:

~~~
python3 clean_ppg.py
~~~

### 3. Band-pass filter
Applies a zero-phase 4th-order Butterworth band-pass at 0.6–3.3 Hz (36–198 BPM) to each channel and confirms a cardiac peak in the IR FFT spectrum:

~~~
python3 filter_ppg.py
~~~

### 4. Extract fiducial points
Computes the VPG, APG and JPG derivatives and locates on, sp, dn, dp, and the APG a wave for every beat on every channel:

~~~
python3 extract_fiducials.py
~~~

### 5. Aggregate per-participant metrics
Produces results/summary.csv with DC, AC, PI, SNR for each channel, plus heart rate, mean red a-wave amplitude, and the standard dual-wavelength SpO2 from the Ratio of Ratios:

~~~
python3 compute_summary.py
~~~

### 6. Apply the three-wavelength model
Solves the 3x3 spectroscopic decomposition (HbO2, Hb, melanin) using extinction coefficients from Prahl (hemoglobin) and Jacques 2013 (eumelanin), and writes the melanin-corrected SpO2 alongside the dual-wavelength estimate:

~~~
python3 three_wavelength_decomposition.py
~~~

A metadata.csv file with columns participant_id, MST must exist alongside the cleaned files for stages 5–6.

## Technical Details

| Parameter | Value |
|---|---|
| Sampling rate | 25 Hz |
| LED wavelengths | Red 660 nm, IR 940 nm, Green 527 nm |
| ADC resolution | 18 bit |
| AGC | Disabled (LED pulse amplitudes locked via static hex writes) |
| Band-pass filter | 4th-order Butterworth, 0.6–3.3 Hz, zero-phase (scipy.signal.filtfilt) |
| Outlier rule | abs(x - local_mean) > 5 * local_std on a 51-sample centred window (~2.04 s) |
| Beat detection | Aboy++-style adaptive-threshold peak finder with refractory suppression |
| Fiducials | on, sp, dn, dp on the PPG; a wave on the APG |
| Dual-wavelength SpO2 | SpO2 (%) = 110 - 25 * R, where R = (AC_red/DC_red) / (AC_IR/DC_IR) |
| Three-wavelength model | C = inv(E) * A, SpO2_corrected = C_HbO2 / (C_HbO2 + C_Hb) * 100 |
| Hemoglobin coefficients | Prahl compilation, omlc.org/spectra/hemoglobin |
| Melanin coefficients | Jacques 2013, mu_a,mel(lambda) = 1.70e12 * lambda^(-3.48) |

## Repository Contents

| File | Purpose |
|---|---|
| arduino_sensor.ino | RedBoard firmware: AGC bypass, fixed LED currents, raw ADC streaming over serial |
| capture_ppg.py | Reads serial output, auto-names P#.csv files |
| clean_ppg.py | Stage 1 — NaN drop, 2 s bootloader trim, 5 sigma outlier flagging |
| filter_ppg.py | Stage 2 — Butterworth band-pass plus FFT verification |
| extract_fiducials.py | Stage 3 — VPG/APG/JPG derivatives, per-beat fiducial extraction |
| compute_summary.py | Stage 4 — DC, AC, PI, SNR, HR, dual-wavelength SpO2 aggregation |
| three_wavelength_decomposition.py | Stage 5 — 3x3 inverse problem, melanin-corrected SpO2 |

## Known Issues and Future Improvements

- **Sensor noise floor at MST 9–10**: For participants at the darkest end of the Monk Skin Tone scale, AC_red drops close to the photodiode shot-noise floor, so even the corrected model cannot recover oxygen saturation reliably. This is a hardware limit, not a software limit.
- **Spectral broadening of LEDs**: The MAX30101 uses standard LEDs whose output is not strictly monochromatic. Narrow-bandwidth LEDs or VCSELs would tighten the extinction-coefficient assumptions in the 3x3 model.
- **Static extinction-coefficient matrix**: The current implementation treats the matrix as fixed. Future work should make the melanin row a function of estimated epidermal thickness (ITA) per participant.
- **Real-time operation**: The pipeline currently runs offline on logged CSVs. Porting the matrix inversion onto the MAX32664's Cortex-M4 for live SpO2 readout is a natural next step.
- **Monte Carlo cross-validation**: A forward optical model in MCXLAB would let the corrected SpO2 output be checked against simulated ground-truth tissues across the full MST range.

## Academic Integrity and Third-Party Code

- The SparkFun Bio Sensor Hub Library (MIT licence) is used as a dependency for low-level I2C transactions with the MAX32664. It is not redistributed in this repository.
- The pyPPG toolbox is referenced and used through its public API; the numpy fallback in extract_fiducials.py reproduces the published algorithm and is documented as such within the file.
- Hemoglobin extinction coefficients are sourced from the Prahl tables at omlc.org; the melanin absorption law is taken from Jacques (2013), Phys. Med. Biol. 58, R37–R61.
- All other code in this repository is original work by the author for the BEng Research Project.

## Author and Supervision

Louis Thong Fook Kei — BEng Electrical and Electronic Engineering, University of Manchester. Supervised by Prof. Thomas Anthopoulos.
