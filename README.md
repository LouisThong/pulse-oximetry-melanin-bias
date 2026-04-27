# Engineering Framework for Evaluating Epidermal Melanin Attenuation in PPG

## Introduction
This repository contains the firmware and signal-processing code 
developed for a BEng research project investigating melanin-induced 
attenuation bias in pulse oximetry across skin tones. The Arduino 
sketch configures a SparkFun SEN-15219 module (MAX30101 + MAX32664) 
to bypass the onboard Automatic Gain Control and stream raw 18-bit 
ADC photon counts. The Python scripts log this data over serial and 
process it through the pyPPG toolbox to extract fiducial points and 
compute SpO₂ via the dual-wavelength Ratio of Ratios method.

## Contextual Overview
The system has three layers:
1. **Sensor layer** — MAX30101 emits red (660 nm), IR (940 nm), and 
   green (527 nm) light; the photodiode captures reflected counts.
2. **Microcontroller layer** — SparkFun RedBoard Qwiic (Arduino) 
   communicates with the MAX32664 over I²C, disables AGC, and 
   forwards raw counts over USB serial.
3. **Host layer** — Python script logs serial data to CSV, then 
   bandpass-filters (0.6–3.3 Hz) and runs pyPPG analysis.

[Insert a simple diagram here later if you want — even a hand-drawn 
photo works]

## Installation

### Hardware
- SparkFun RedBoard Qwiic (DEV-15123)
- SparkFun Pulse Oximeter Sensor (SEN-15219)
- Qwiic cable, USB-B cable

### Arduino setup
1. Install Arduino IDE (version 2.x or later)
2. Install the SparkFun Bio Sensor Hub library via Library Manager
3. Open `arduino/arduino_sensor.ino` and upload to the RedBoard

### Python setup
Requires Python 3.9 or later. Install dependencies:

    pip install pyserial pyPPG numpy scipy pandas matplotlib

## How to run

1. Connect the RedBoard via USB and note the COM port
2. Upload the Arduino sketch — open the Serial Monitor at 115200 
   baud to confirm "System Ready. AGC Disabled. Streaming Raw Data."
3. Close the Serial Monitor, then edit `python/data_logger.py` and 
   set the correct COM port (e.g. `COM3` on Windows, `/dev/ttyUSB0` 
   on Linux)
4. Run:

       python python/data_logger.py

   This will log 60 seconds of raw ADC counts to `raw_ppg_data.csv`
5. Run the processing script:

       python python/ppg_processing.py raw_ppg_data.csv

## Technical Details
- **Sampling rate:** 25 Hz (set in Arduino loop with 10 ms delay = 100 Hz, 
  decimated by sensor hub)
- **Bandpass filter:** 4th-order zero-phase Butterworth, fL = 0.6 Hz, 
  fH = 3.3 Hz (captures 36–198 BPM)
- **Fiducial extraction:** pyPPG with Aboy++ beat detector, smoothing 
  windows {ppg: 50ms, vpg/apg/jpg: 10ms}
- **SpO₂ calibration:** standard linear curve, SpO₂ (%) = 110 − 25·R, 
  where R = (AC_red/DC_red) / (AC_ir/DC_ir)
- **AGC bypass:** achieved via I²C handshake disabling autonomous 
  LED current modulation; fixed pulse amplitudes locked via static 
  hex writes to MAX30101 LED PA registers

## Known Issues and Future Improvements
- The dual-wavelength algorithm becomes unstable at MST ≥ 7 because 
  AC_red drops near the photodiode shot-noise floor
- A third-wavelength (527 nm green) extension is proposed in the 
  report but not yet implemented in code
- Future work: integrate Monte Carlo skin-layer simulation (MCXLAB) 
  for forward-model validation, and a PyTorch calibration network 
  using continuous MST as input

## Acknowledgements
- pyPPG toolbox: Goda et al. (2024)
- SparkFun Bio Sensor Hub library (open-source, MIT licence)
- Project supervised by Prof. Thomas Anthopoulos, University of Manchester
