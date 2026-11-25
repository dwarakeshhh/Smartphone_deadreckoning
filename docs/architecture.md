# Architecture

```mermaid
flowchart LR
  subgraph Phone[Android app]
    S[Accelerometer / Gyroscope / Magnetometer] --> H[SensorHub 50 Hz] --> W[TrackerSocket OkHttp WebSocket]
  end
  W -- "JSON over Wi-Fi ws://ip:8000/ws" --> WS
  subgraph Server[FastAPI server]
    WS[websocket_server] --> E[TrackingEngine]
    E --> P[SensorPreprocessor: validate, dt, filters]
    P --> C[Calibrator] --> O[Orientation: Mahony / Kalman]
    O --> G[Gravity + bias removal] --> Z[Stationary detector / ZUPT] --> D[DeadReckoner]
    P --> ST[StepDetector] --> M[MovementClassifier]
    E --> R[SessionRecorder -> data/raw CSV]
    E -.optional.-> ML[ml.predict RandomForest]
    API[REST: /state /trajectory /reset ...] --- E
  end
  API -- "HTTP polling 0.5 s" --> UI[Streamlit dashboard + Plotly]
```

| path | role |
|---|---|
| `android_app/` | Kotlin app (UI, sensors, WebSocket, auto-reconnect) |
| `server/main.py` | FastAPI app + REST; `websocket_server.py` WebSocket endpoint |
| `server/tracking_engine.py` | pipeline orchestration, session state, snapshots |
| `server/{sensor_processor,orientation,dead_reckoning,step_detector,movement_classifier}.py` | algorithms |
| `server/recorder.py`, `server/evaluation.py` | recording, accuracy metrics against ground truth |
| `ml/` | feature extraction (shared with the live server), training, prediction |
| `dashboard/app.py` | Streamlit UI |
| `tools/` | simulator + synthetic generator (**dev/test only**, packets are flagged `source=simulator`) |
| `tests/` | 51 pytest tests |

Design choices: one in-process engine guarded by a lock (50 Hz of tiny numpy ops is far below one core); the dashboard is a pure
REST client so it can run on another machine; the ML model is optional and never fabricated.
