"""Central configuration for the dead-reckoning server.

Every tunable number lives here so that it can be documented and changed in
one place.  Values can be overridden with environment variables where noted
(``DR_HOST``, ``DR_PORT``, ``DR_DATA_DIR``).

All thresholds were chosen as *reasonable starting points* for a hand-held
Android phone sampled at 50 Hz.  They have NOT been tuned against a ground
truth data-set; see docs/testing.md for how to tune them with your own phone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    return Path(os.getenv("DR_DATA_DIR", str(PROJECT_ROOT / "data")))


@dataclass
class Settings:
    # ------------------------------------------------------------ network
    host: str = os.getenv("DR_HOST", "0.0.0.0")
    port: int = int(os.getenv("DR_PORT", "8000"))

    # -------------------------------------------------------------- paths
    data_dir: Path = field(default_factory=_data_dir)
    model_dir: Path = PROJECT_ROOT / "ml" / "model"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "motion_classifier.joblib"

    # ----------------------------------------------------------- sampling
    nominal_rate_hz: float = 50.0       # expected phone sampling rate
    max_gap_s: float = 0.5              # larger dt => data gap, filters/velocity reset

    # --------------------------------------------------------- validation
    max_acc: float = 160.0              # m/s^2   (|a| above this is rejected)
    max_gyro: float = 35.0              # rad/s   (~2000 deg/s)
    max_mag: float = 2000.0             # uT
    mag_valid_min: float = 10.0         # uT, below => magnetometer unusable for fusion
    mag_valid_max: float = 150.0        # uT, above => strong disturbance

    # ------------------------------------------------------------ filters
    acc_lowpass_hz: float = 10.0
    gyro_lowpass_hz: float = 15.0
    mag_lowpass_hz: float = 3.0
    filter_order: int = 2
    lin_acc_highpass_hz: float = 0.0    # 0 = off. NOTE: a high-pass on acceleration is a velocity leak with
                                        # tau = 1/(2*pi*fc); 0.05 Hz (tau=3.2 s) removed ~25 % of a 3 s move in tests

    # ------------------------------------------------------------- gravity
    gravity: float = 9.80665

    # -------------------------------------------------------- calibration
    calibration_seconds: float = 1.5
    calibration_acc_std_max: float = 0.20   # m/s^2, std of |a| over window
    calibration_gyro_mean_max: float = 0.15  # rad/s

    # ------------------------------------------------------------- fusion
    fusion_mode: str = "mahony"         # "mahony" (complementary) or "kalman"
    use_magnetometer: bool = True
    mahony_kp_acc: float = 0.5
    mahony_kp_mag: float = 0.5
    mahony_ki: float = 0.0
    acc_trust_sigma: float = 1.0        # m/s^2 deviation of |a| from g at which trust drops to 37 %
    acc_trust_lin_sigma: float = 0.5    # m/s^2 gravity-free acceleration at which trust drops to 37 %
    acc_trust_floor: float = 0.0        # minimum accelerometer trust
    mag_trust_tolerance: float = 0.25   # allowed relative change of |m| vs calibration value
    kalman_q_angle: float = 0.001
    kalman_q_bias: float = 0.00001
    kalman_r_meas: float = 1.0

    # -------------------------------------------------- stationary / ZUPT
    stat_window_s: float = 0.4
    stat_acc_std: float = 0.15          # m/s^2
    stat_gyro_mean: float = 0.08        # rad/s
    stat_acc_dev: float = 0.50          # m/s^2, | mean|a| - g |
    stat_lin_acc_mean: float = 0.25     # m/s^2, mean gravity-free world acceleration over the window
    stat_enter_s: float = 0.25          # must look still this long before ZUPT
    stat_exit_lin_acc: float = 0.40     # m/s^2 instantaneous => leave stationary at once
    stat_exit_gyro: float = 0.25        # rad/s instantaneous

    # ------------------------------------------------------ dead reckoning
    acc_deadband: float = 0.05          # m/s^2 horizontal, suppresses noise creep
    velocity_leak_tau_s: float = 60.0   # 0 disables; bounds un-observed horizontal velocity error
    vertical_leak_tau_s: float = 2.0    # vertical channel is experimental: strongly damped
    zupt_replay_s: float = 0.3          # on leaving STATIONARY, re-integrate this much history
    max_speed: float = 8.0              # m/s, clamp for abnormal estimates
    bias_rate_s: float = 1.0            # time-constant of stationary bias learning
    gyro_bias_rate_s: float = 5.0
    gravity_rate_s: float = 10.0

    # --------------------------------------------------------------- steps
    step_min_peak: float = 1.0          # m/s^2 above gravity (absolute floor)
    step_peak_std_factor: float = 0.8   # adaptive: mean + k*std of last 2 s
    step_min_interval_s: float = 0.25
    step_max_interval_s: float = 1.5
    step_confirm_count: int = 3
    step_length_m: float = 0.70         # fixed stride length (user setting)

    # ---------------------------------------------------------- classifier
    run_cadence_hz: float = 2.6
    walk_cadence_min_hz: float = 0.8
    state_debounce_samples: int = 8
    use_ml_state: bool = False          # if True and a model exists, ML label is shown as state

    # ------------------------------------------------------------------ ML
    ml_window_s: float = 2.0
    ml_hop_s: float = 1.0

    # --------------------------------------------------------------- logs
    traj_log_interval_s: float = 0.1
    traj_max_points: int = 60000
    recent_samples: int = 500
    auto_start: bool = False            # start tracking on first sample without a 'start' command


settings = Settings()
