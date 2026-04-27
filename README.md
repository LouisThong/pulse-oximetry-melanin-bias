# Engineering Framework for Evaluating Epidermal Melanin Attenuation in PPG

Using the MAX30101 optical sensor and MAX32664 biometric hub.

BEng Research Project — Department of Electrical and Electronic Engineering, University of Manchester (2025/26).

## Introduction

This repository contains the full firmware and signal-processing pipeline developed for a research project investigating melanin-induced attenuation bias in pulse oximetry across different skin tones. Standard two-wavelength pulse oximeters systematically overestimate blood oxygen saturation (SpO2) in darker skin because epidermal melanin absorbs red light (660 nm) far more strongly than near-infrared light (940 nm), distorting the Ratio of Ratios on which the calibration depends.

The Arduino sketch configures a SparkFun SEN-15219 module (MAX30101 + MAX32664) to bypass the on-board Automatic Gain Control (AGC) and stream raw 18-bit ADC photon counts to the host. The Python pipeline then logs the data, cleans it, band-pass filters the signal, extracts cardiac fiducial points (pulse onset, systolic peak, dicrotic notch, diastolic peak, plus the APG a-wave), and produces a per-participant signal-quality summary. A final stage implements the proposed three-wavelength spectroscopic decomposition that adds the 527 nm green channel as a third equation in order to algebraically eliminate melanin from the SpO2 calculation.

## Contextual Overview

The system has three layers:

1. **Sensor layer** — MAX30101 emits red (660 nm), infrared (940 nm) and green (527 nm) light through a single photodiode aperture. The MAX32664 hub manages the optical front end over an internal I2C bus.
2. **Microcontroller layer** — A SparkFun RedBoard Qwiic (Arduino-compatible) acts as I2C master, runs `arduino_sensor.ino`, performs the AGC-bypass startup handshake (RSTN/MFIO toggle, mode write, AGC disable, static LED current lock), and forwards raw red/IR/green ADC counts over USB serial at 115 200 baud.
3. **Host layer (Python)** — A five-stage pipeline consumes the serial stream and converts raw counts into signal-quality metrics:
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
## Installation

### Hardware required
- SparkFun RedBoard Qwiic (DEV-15123) or any Arduino Uno-compatible board with I2C
- SparkFun Pulse Oximeter and Heart Rate Sensor — MAX30101 + MAX32664 (SEN-15219)
- Qwiic cable, USB-B cable
- A dark enclosure to exclude ambient light during recording

### Arduino-side setup
1. Install Arduino IDE (2.x or later)
2. Install the **SparkFun Bio Sensor Hub Library** via Library Manager
3. Open `arduino_sensor.ino`, select the correct board and port, and upload

### Python-side setup
Python 3.9 or later. From the repository root:
