# MADTP Wearable Project — Improved Working Copy

This folder preserves the supplied project modules and adds a safer launcher plus the live ESP32-to-Python runner. It does not claim medical-grade validation.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --mode audit
```

## Run with the ESP32

1. Upload the matching MADTP ESP32 firmware and confirm it emits one JSON object per line at 115200 baud.
2. Stop any other serial-reader process (only one process can own the USB serial port).
3. Set your port if needed, then run:

```bash
export MADTP_SERIAL_PORT=/dev/cu.usbserial-0001
export MADTP_BAUD_RATE=115200
python main.py --mode live
```

The live runner uses the first **15 minutes of valid observations** as its personal baseline by default, then freezes that baseline. For a short bench test only, you can set `MADTP_BASELINE_MINUTES=0.25` before launch; do not describe that test as a 15-minute baseline. After baseline collection, the runner performs the existing feature extraction, multimodal fusion, twin-divergence analysis, Random Forest inference, and current-risk calculation.

## Important boundaries

- The DS18B20 is a sensor/surface temperature proxy, not a validated core-body-temperature sensor.
- MAX30102 readings are prototype measurements and are not medical-grade. Missing HR/SpO2 values should not be replaced with zero.
- The activity score is a heuristic based on accelerometer and gyroscope variation, not a calibrated activity classifier.
- The live pipeline currently reports **current risk**. Future-horizon prediction and what-if intervention outputs are not yet integrated into the live output path.
- The buzzer/vibration firmware is not automatically controlled by the Python risk score unless a bidirectional communication path is implemented and tested.
- Model files were trained in a different software environment. If scikit-learn reports pickle-version warnings or inference errors, use a compatible environment or retrain/evaluate reproducibly; do not ignore compatibility warnings in formal results.
- Do not use this prototype for medical diagnosis or as the only worker-safety control.

## Start the dashboard with live API data

Run the serial prediction process in Terminal 1:

```bash
python main.py --mode live
```

Run the dashboard/API in Terminal 2:

```bash
python api_server.py
```

Open `http://127.0.0.1:8000` in your browser. In Settings, turn **Demo values** off to poll `/api/latest` and `/api/history`. The API reads the results written by the Python live pipeline; it does not calculate a second risk score. If no live result exists yet, the dashboard will say it is waiting. Keep the dashboard server bound to localhost unless you deliberately configure and secure remote access.

The frontend sign-in screen is presentation-only and accepts any submitted values; it does not authenticate or protect the dashboard or API. Do not expose this prototype to untrusted networks.

## Modes

- `python main.py --mode audit` — checks required files; safe default.
- `python main.py --mode live` — starts the ESP32 serial prediction pipeline.
- `python main.py --mode batch` — runs the original multi-script batch workflow only after explicit confirmation. It is retained for reference and is not the live system.

## Current development sequence

1. Validate live sensor data and frozen baseline.
2. Validate persistent Digital Twin state and divergence.
3. Integrate and validate future-risk horizons.
4. Implement what-if intervention simulations and clearly label them as simulations.
5. Connect Python risk results to firmware alerts with communication-failure handling.
6. Connect the frontend to a real backend API.
7. Evaluate on held-out, realistic data before reporting performance.
