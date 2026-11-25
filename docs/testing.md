# Testing, tuning and accuracy evaluation

## Automated tests (no phone needed)
```
python -m pytest -q          # 51 tests
```
Unit/pipeline tests drive the engine with **synthetic IMU data from a known prescribed motion** (`tools/synthetic_imu.py`):
orientation convergence, sign conventions, ZUPT/no-drift, straight moves in 4 directions, closed square, yaw-rotated frame,
step counting, calibration gating, gaps, REST + WebSocket end-to-end, recording files, evaluation maths, ML mechanics.
These prove the code is wired correctly; **they say nothing about real-world accuracy.**

## Manual end-to-end check without a phone
```
python -m server.main                       # terminal 1
python -m tools.phone_simulator --scenario square --stop     # terminal 2 (flagged SIMULATED on the dashboard)
```

## Real test procedure (with the phone)
1. Phone flat on a table at a marked start point -> START TRACKING -> wait for "LIVE TRACKING".
2. **Test 1 straight line**: slide the phone 1.0 m along a ruler/tape, stop, STOP TRACKING. Repeat 5×. Record est. vs 1.0 m.
3. **Test 2 square**: slide around a 0.5 m (or walk a 3 m) taped square back to start. Record end-position error.
4. **Test 3 stationary**: leave 60 s; drift should stay ≈0 (ZUPT).
5. **Test 4 walking**: walk a measured 10 m line holding the phone steadily; compare step count, step distance, integrated distance.
6. Dashboard sidebar -> *Record* before Start so each run is saved; then evaluate:
```
python -m server.evaluation --est data/processed/session_001_<name>_trajectory.csv --truth-waypoints "0,0;3,0;3,3;0,3;0,0"
```
Ground truth must be independent of the system (tape-measured waypoints, floor markings, overhead camera/AprilTags, mocap).
Report distance error %, final-position error, drift % of path, RMSE, rigid-aligned RMSE (shape error without heading offset).

## Tuning (all in `server/config.py`)
* Drift while still / ZUPT never triggers: raise `stat_acc_std`, `stat_gyro_mean` slightly.
* Motion cut short / ZUPT triggers mid-motion: lower them. Steps missed: lower `step_min_peak`.
* Wrong heading: check `mag_trust` on the dashboard; set `use_magnetometer=False` for a relative heading.

## ML model (optional, needs REAL data)
Record >=2 sessions (>= ~30 s each) per class (stationary, walking, running) in recording mode with the matching label, then
`python -m ml.train_model`. It validates with leave-**session**-out cross-validation, prints accuracy/macro-F1/confusion
matrix, and refuses to write a model if data is insufficient. Restart the server to load `ml/model/motion_classifier.joblib`.
Scores apply only to your phone/people/placements.
