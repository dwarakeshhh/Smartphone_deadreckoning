# Sensor processing & algorithms

Pipeline (all in `server/`): **validate -> timestamps -> low-pass -> calibrate -> orientation -> gravity removal
-> stationary detection (ZUPT) -> integrate -> steps / state**.

## Coordinate frames
Phone: X right, Y up-screen, Z out-of-screen. World (ENU): X East, Y North, Z Up. `R` maps phone->world.
A phone flat, face up, top edge North has R = I. Roll/pitch/yaw are intrinsic Z-Y-X; *yaw* is counter-clockwise from East,
*heading* is the compass bearing of the phone's top edge.

## Filters (causal, one sample at a time; `sensor_processor.py`)
| signal | filter | why |
|---|---|---|
| accelerometer | Butterworth low-pass 10 Hz, order 2 | removes noise that would be integrated twice; motion is < 5 Hz |
| gyroscope | low-pass 15 Hz + bias removal | removes vibration; bias learned at start and while still |
| magnetometer | low-pass 3 Hz | heading reference must be smooth |
| world lin. accel. | optional high-pass (`lin_acc_highpass_hz`, default OFF) | a high-pass on acceleration is a velocity leak; measured to cost ~25 % distance at 0.05 Hz, so off |

## Calibration (`Calibrator`)
Needs ~1.5 s of stillness (sliding window; motion simply delays it). Estimates gyro bias, local |g| (absorbs accelerometer
scale error), initial attitude, reference magnetic strength. While stationary afterwards: gyro bias, |g| and a horizontal
acceleration bias are slowly re-learned.

## Orientation (`orientation.py`)
* **Mahony complementary filter** (default): integrates the gyro; accelerometer corrects roll/pitch, magnetometer corrects
  *yaw only* (so magnetic disturbance can never tilt the attitude). Accelerometer trust falls when |a| != g **or** the
  gravity-free acceleration is large; magnetometer trust is 0 when |m| deviates >25 % from its calibration value.
* **Kalman tilt filter** (`fusion_mode="kalman"`): per-axis [angle, gyro-bias] Kalman for roll/pitch + complementary yaw.
  **Experimental** - on the synthetic push test it under-estimates distance (~60 % vs ~90 % for Mahony).
* No magnetometer: world Y is defined by the phone's initial +Y direction (relative heading).

## Dead reckoning (`dead_reckoning.py`)
`a_world = R·a_body - [0,0,g]` (minus learned bias) -> `v += a·dt` -> `p += v·dt`. Mitigation: horizontal dead-band (0.05 m/s²),
**ZUPT** (velocity forced to 0 while stationary), slow velocity leak (τ=60 s horizontal, 2 s vertical), speed clamp, and a 0.3 s
replay on leaving stationary so the detection lag does not swallow the start of a movement.
Distance = horizontal path length of the integrated trajectory. The vertical (Z) channel is experimental.

## Stationary detection (`movement_classifier.py`)
Windowed (0.4 s) std of |a|, mean |ω|, mean gravity-free acceleration and |a|-g must all be small for 0.25 s; leaving is immediate
when instantaneous acceleration/rotation exceeds a threshold. Limitation: perfectly constant-velocity motion looks "still".

## Steps (`step_detector.py`)
Peaks of |a|-g above an adaptive threshold (max(1 m/s², mean+0.8σ over 2 s)), ≥0.25 s apart, 3 regular peaks to confirm
(then counted retroactively). Step distance = steps × `step_length_m` (fixed 0.70 m, user-configurable) - shown separately from
the integrated distance.

## Movement state
Rule-based: STATIONARY (zero-velocity detector), WALKING (cadence ≥ 0.8 steps/s), RUNNING (cadence ≥ 2.6 and strong
oscillation), UNKNOWN (moving without gait pattern, e.g. sliding the phone by hand), debounced over 8 samples.
Optional ML classifier (Random Forest / Logistic Regression, `ml/`) is shown separately as `ml_state`.

## Known limitations
* Double integration => position error grows ~quadratically between ZUPTs. Expect decimetres after seconds of motion
  and metres after a minute of walking. This is inherent to phone-grade IMUs.
* Heading indoors depends on the magnetometer, which steel/electronics distort; the trust gate mitigates but cannot cure.
* Hand-held walking has large orientation changes; step-based distance is usually more reliable than integrated distance.
* Thresholds are starting points, not tuned on real recordings (see docs/testing.md).
