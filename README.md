# Real-Time Smartphone Inertial Motion Tracking

An Android phone streams accelerometer, gyroscope and magnetometer data over Wi-Fi to a Python server that estimates
**orientation, steps, movement state and a 2-D trajectory by inertial dead reckoning** (no GPS), shown live in a Streamlit dashboard.

> **Honest expectations:** double-integrating a phone IMU drifts. Expect decimetre-level error within seconds of motion and metres
> after a minute. Zero-velocity updates, bias learning, filtering and a calibration step reduce drift but cannot remove it.
> The dashboard says so, shows a heuristic uncertainty, and every number comes from real received packets.

## Status of this delivery
| part | status |
|---|---|
| Python server, algorithms, recording, evaluation | built and **tested** (51 pytest tests pass; also run as a real uvicorn process driven over a real WebSocket) |
| Streamlit dashboard | built; render-tested with Streamlit's AppTest against a live server and started headless. **Not visually reviewed in a browser.** |
| Android app | source written, **NOT compiled or run** (no Android SDK available here). First build happens on your machine. |
| ML movement classifier | pipeline built and tested on synthetic mechanics only. **No model is shipped** - train it on your own recordings. |
| Accuracy on real walking/hand motion | **not measured** - needs your phone and a ground-truth procedure (docs/testing.md) |

## Quick start (Windows)
1. Install Python 3.11+ (tick "Add to PATH"). Double-click **`run_server.bat`** (creates `.venv`, installs packages, starts server).
   It prints e.g. `Phone WebSocket URL : ws://192.168.1.5:8000/ws`.
2. Double-click **`run_dashboard.bat`** -> opens http://localhost:8501.
3. If the phone can't connect, allow the port once (Admin PowerShell):
   `netsh advfirewall firewall add rule name="DeadReckoning" dir=in action=allow protocol=TCP localport=8000`
4. Build/install the Android app (`android_app/README.md`), enter the laptop IP, **CONNECT**, put the phone flat and still, **START TRACKING**.
5. Wait for "LIVE TRACKING" (≈1.5 s calibration), move the phone, press **STOP TRACKING** for the session summary.

Linux/macOS: `./run_server.sh` and `./run_dashboard.sh`.

## Try it without a phone
```
python -m server.main
python -m tools.phone_simulator --scenario square --stop     # dashboard shows a "SIMULATED DATA" banner
```

## Recording sensor data
Dashboard sidebar -> name + label -> **Record** -> run on the phone -> **Save**. Files: `data/raw/session_NNN_<name>.csv/.json`,
estimated path in `data/processed/..._trajectory.csv`. Then `python -m ml.train_model` (needs real data, ≥2 sessions per class) and
`python -m server.evaluation ...` for accuracy vs ground truth. Details: `docs/testing.md`.

## Demo script (≈5 min)
1. Show phone app: sensors available, live values. 2. Connect -> dashboard "PHONE CONNECTED". 3. Start tracking, hold still (calibrating -> LIVE TRACKING).
4. Slide the phone ~0.5 m forward: trajectory moves, state changes. 5. Stop: summary (distance, steps, avg/max speed, final position).
6. Rest the phone: no drift (ZUPT). 7. Reset Position / Reset Session. 8. Show the quality panel and say plainly where it drifts.

## Layout
`android_app/` · `server/` · `dashboard/` · `ml/` · `tools/` · `tests/` · `data/` · `docs/` (architecture, protocol, sensor_processing, testing).

## Known limitations
Drift; magnetometer disturbance indoors; constant-velocity motion looks "stationary"; fixed step length; phone-in-pocket/bag not
supported; vertical channel experimental; Kalman mode experimental; thresholds untuned on real data; one phone at a time.
