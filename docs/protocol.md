# Protocol & data formats

## Transport
Phone -> laptop over **WebSocket** `ws://<laptop-ip>:8000/ws` (text frames, JSON). One phone at a time (a new
connection replaces the old one).

## Sensor packet (phone -> server), ~50 per second
```json
{"timestamp": 123456789.0, "acc_x": 0.12, "acc_y": 0.03, "acc_z": 9.81,
 "gyro_x": 0.001, "gyro_y": -0.002, "gyro_z": 0.0,
 "mag_x": 21.5, "mag_y": -5.0, "mag_z": -40.2, "source": "android"}
```
| field | unit | notes |
|---|---|---|
| `timestamp` | ms | monotonic phone clock (`SensorEvent.timestamp/1e6`); only differences matter |
| `acc_*` | m/s² | **includes gravity** (`TYPE_ACCELEROMETER`) |
| `gyro_*` | rad/s | `TYPE_GYROSCOPE` |
| `mag_*` | µT | `TYPE_MAGNETIC_FIELD`; all zeros = no magnetometer |
| `source` | text | optional; anything other than `android` makes the dashboard show a "SIMULATED / REPLAYED" banner |

Axes (phone portrait): X right, Y up the screen, Z out of the screen.
A batch `{"samples":[{...},{...}]}` is also accepted.

Rejected (counted, reported back as `{"type":"error"}`): invalid JSON, missing/non-numeric/NaN fields, |acc|>160 m/s²,
|gyro|>35 rad/s, |mag|>2000 µT, non-increasing timestamps (dropped silently, counted as out-of-order).
Gaps > 0.5 s reset filters and zero the velocity; smaller gaps are counted as missed samples.

## Control messages (phone or any client -> server)
`{"type":"control","command":"start"|"stop"|"reset"}` -> `{"type":"control_ack",...}`
* `start`: begin a new session (calibration first). No-op if already tracking (so reconnects don't reset).
* `stop`: stop tracking, freeze the summary. `reset`: forget the session, re-calibrate.

## Server -> phone
`{"type":"hello",...}` on connect; `{"type":"status","tracking":..,"phase":..,"packets":..,"state":..}` every second.

## REST API (default http://localhost:8000, interactive docs at `/docs`)
| method & path | purpose |
|---|---|
| `GET /health`, `/status` | liveness; connection/tracking status |
| `GET /state` | everything the dashboard shows, in one call |
| `GET /position`, `/sensors`, `/stats` | position+heading; latest sensors+orientation; statistics + quality |
| `GET /trajectory?max_points=` | estimated path (x,y,z,t,speed arrays) |
| `POST /reset?scope=position\|session\|data` | reset position only / whole session+recalibration / clear plot+buffer |
| `POST /tracking/start`, `/tracking/stop` | control from the dashboard |
| `POST /recording/start` `{"name","label"}`, `/recording/stop`, `GET /recording/sessions` | recording mode |

## Raw recording CSV (`data/raw/session_NNN_<name>.csv`)
Header exactly: `timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z` (units as above).
Sidecar `.json`: `session, name, label, file, started_at, stopped_at, rows, duration_s, mean_rate_hz, source`.
Trajectory CSV (`data/processed/..._trajectory.csv`): `t_s,x_m,y_m,z_m,speed_mps,state`.
Labels: `stationary | walking | running | other`.
