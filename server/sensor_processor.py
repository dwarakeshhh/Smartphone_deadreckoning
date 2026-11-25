"""Sensor validation, timestamp handling, filtering and start-up calibration.

Why each filter is used
-----------------------
* Accelerometer low-pass (2nd-order Butterworth, 10 Hz): MEMS accelerometers
  have broadband noise.  Human motion and walking live below ~5 Hz, so
  removing >10 Hz noise reduces the noise that gets integrated twice.
* Gyroscope low-pass (Butterworth, 15 Hz): removes vibration noise while
  keeping fast rotations; gyro bias is removed separately (calibration).
* Magnetometer low-pass (Butterworth, 3 Hz): the heading reference should be
  smooth; fast magnetometer changes are mostly noise/disturbance.
* Optional world-frame linear-acceleration high-pass (1st-order Butterworth,
  ``lin_acc_highpass_hz``, applied in the engine, OFF by default): removes
  residual constant bias, but acts as a velocity leak with tau = 1/(2*pi*fc),
  so it is only useful at very low cut-offs.  Bias is instead learned during
  stationary periods.

The filters are *causal* (online, one sample at a time) because the system is
real-time.  They are implemented as cascaded biquads (second-order sections).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt_zi

REQUIRED_KEYS = ("timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z")


class ValidationError(ValueError):
    pass


@dataclass
class Sample:
    t_ms: float
    acc: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray
    source: str = "android"


def validate_packet(d: object, cfg) -> Sample:
    """Validate one JSON packet and convert it into a :class:`Sample`."""
    if not isinstance(d, dict):
        raise ValidationError("packet must be a JSON object")
    vals = {}
    for k in REQUIRED_KEYS:
        if k not in d:
            raise ValidationError(f"missing field '{k}'")
        v = d[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(f"field '{k}' must be a number")
        v = float(v)
        if not math.isfinite(v):
            raise ValidationError(f"field '{k}' is not finite")
        vals[k] = v
    acc = np.array([vals["acc_x"], vals["acc_y"], vals["acc_z"]])
    gyro = np.array([vals["gyro_x"], vals["gyro_y"], vals["gyro_z"]])
    mag = np.array([vals["mag_x"], vals["mag_y"], vals["mag_z"]])
    if np.linalg.norm(acc) > cfg.max_acc:
        raise ValidationError("abnormal accelerometer magnitude")
    if np.linalg.norm(gyro) > cfg.max_gyro:
        raise ValidationError("abnormal gyroscope magnitude")
    if np.linalg.norm(mag) > cfg.max_mag:
        raise ValidationError("abnormal magnetometer magnitude")
    src = d.get("source", "android")
    return Sample(vals["timestamp"], acc, gyro, mag, str(src)[:24])


def design_filter(kind: str, fc: float, fs: float, order: int):
    """Butterworth SOS design; returns None when fc <= 0 (filter disabled)."""
    if fc is None or fc <= 0:
        return None
    nyq = fs / 2.0
    fc = min(fc, 0.45 * fs) if kind == "low" else min(fc, 0.45 * nyq)
    return butter(order, fc / nyq, btype=kind, output="sos")


class OnlineSOS:
    """Sample-by-sample cascaded-biquad filter for a vector signal."""

    def __init__(self, sos):
        self.sos = None if sos is None else np.asarray(sos, float)
        self.zi = None if sos is None else sosfilt_zi(self.sos)
        self.z = None

    def reset(self, x0: np.ndarray) -> None:
        if self.sos is not None:
            self.z = self.zi[:, :, None] * np.asarray(x0, float)[None, None, :]

    def step(self, x: np.ndarray) -> np.ndarray:
        if self.sos is None:
            return np.asarray(x, float)
        if self.z is None:
            self.reset(x)
        y = np.asarray(x, float)
        for i, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            out = b0 * y + self.z[i, 0]
            self.z[i, 0] = b1 * y - a1 * out + self.z[i, 1]
            self.z[i, 1] = b2 * y - a2 * out
            y = out
        return y


@dataclass
class Processed:
    t_s: float
    dt: float
    acc_raw: np.ndarray
    acc_f: np.ndarray
    gyro_f: np.ndarray
    mag_f: np.ndarray
    mag_present: bool
    mag_valid: bool
    gap: bool = False


@dataclass
class PreprocStats:
    accepted: int = 0
    out_of_order: int = 0
    gaps: int = 0
    missed_samples: int = 0
    rate_hz: float = 0.0


class SensorPreprocessor:
    """Timestamps -> dt, gap/missing-sample handling, low-pass filtering."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.last_t: float | None = None
        self.dt_ema = 1.0 / self.cfg.nominal_rate_hz
        self.fs_design = self.cfg.nominal_rate_hz
        self.stats = PreprocStats(rate_hz=self.cfg.nominal_rate_hz)
        self._design(self.fs_design)
        self._n = 0

    def _design(self, fs: float) -> None:
        c = self.cfg
        self.f_acc = OnlineSOS(design_filter("low", c.acc_lowpass_hz, fs, c.filter_order))
        self.f_gyro = OnlineSOS(design_filter("low", c.gyro_lowpass_hz, fs, c.filter_order))
        self.f_mag = OnlineSOS(design_filter("low", c.mag_lowpass_hz, fs, c.filter_order))
        self.fs_design = fs

    def process(self, s: Sample) -> Processed | None:
        c = self.cfg
        t = s.t_ms / 1000.0
        gap = False
        if self.last_t is None:
            dt, gap = 1.0 / c.nominal_rate_hz, True
        else:
            dt = t - self.last_t
            if dt <= 0:
                self.stats.out_of_order += 1
                return None
            if dt > c.max_gap_s:
                gap = True
                self.stats.gaps += 1
                dt = 1.0 / c.nominal_rate_hz
            else:
                if dt > 1.5 * self.dt_ema:
                    self.stats.missed_samples += max(0, int(round(dt / self.dt_ema)) - 1)
                self.dt_ema = 0.98 * self.dt_ema + 0.02 * dt
        self.last_t = t
        self._n += 1
        if not gap and self._n % 100 == 0:
            fs_est = 1.0 / self.dt_ema
            self.stats.rate_hz = fs_est
            if abs(fs_est - self.fs_design) / self.fs_design > 0.25:
                self._design(fs_est)       # sampling rate differs from design: re-tune
                gap = True                 # re-initialise filter state
        if gap:
            for f in (self.f_acc, self.f_gyro, self.f_mag):
                f.reset(np.zeros(3))
                f.z = None
        mag_present = float(np.linalg.norm(s.mag)) > 1e-6
        mag_norm = float(np.linalg.norm(s.mag))
        mag_valid = mag_present and c.mag_valid_min <= mag_norm <= c.mag_valid_max
        self.stats.accepted += 1
        return Processed(
            t_s=t, dt=dt, acc_raw=s.acc,
            acc_f=self.f_acc.step(s.acc), gyro_f=self.f_gyro.step(s.gyro), mag_f=self.f_mag.step(s.mag),
            mag_present=mag_present, mag_valid=mag_valid, gap=gap,
        )


@dataclass
class CalibrationResult:
    gyro_bias: np.ndarray
    g_est: float
    acc_mean: np.ndarray
    mag_mean: np.ndarray | None
    mag_ref_norm: float | None


class Calibrator:
    """Start-up calibration: needs the phone to lie still for ``calibration_seconds``.

    Estimates the gyroscope bias, the local gravity magnitude (absorbing the
    accelerometer scale error), the initial attitude inputs and the reference
    magnetic-field strength.  Uses a sliding window, so any motion simply
    postpones completion until the phone has been still long enough.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        n = max(5, int(self.cfg.calibration_seconds * self.cfg.nominal_rate_hz))
        self.buf: deque = deque(maxlen=n)
        self.progress = 0.0
        self.waiting_for_still = False

    def add(self, p: Processed) -> CalibrationResult | None:
        self.buf.append((p.acc_f.copy(), p.gyro_f.copy(), p.mag_f.copy(), p.mag_valid))
        self.progress = len(self.buf) / self.buf.maxlen
        if len(self.buf) < self.buf.maxlen:
            return None
        acc = np.array([b[0] for b in self.buf])
        gyro = np.array([b[1] for b in self.buf])
        acc_norm = np.linalg.norm(acc, axis=1)
        still = acc_norm.std() < self.cfg.calibration_acc_std_max and \
            np.linalg.norm(gyro, axis=1).mean() < self.cfg.calibration_gyro_mean_max
        self.waiting_for_still = not still
        if not still:
            return None
        mags = np.array([b[2] for b in self.buf])
        mag_ok = all(b[3] for b in self.buf)
        g_est = float(np.clip(acc_norm.mean(), 8.0, 11.5))
        return CalibrationResult(
            gyro_bias=gyro.mean(axis=0), g_est=g_est, acc_mean=acc.mean(axis=0),
            mag_mean=mags.mean(axis=0) if mag_ok else None,
            mag_ref_norm=float(np.linalg.norm(mags.mean(axis=0))) if mag_ok else None,
        )
