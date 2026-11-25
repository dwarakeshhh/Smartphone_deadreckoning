# data/

* `raw/`       recorded sessions: `session_NNN_<name>.csv` + `.json` metadata (label, rate, duration)
* `processed/` `features.csv` (training windows) and `session_NNN_<name>_trajectory.csv` (estimated paths)

Raw CSV columns (exactly, one row per sample, header required):
`timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z`
See docs/protocol.md for units.
