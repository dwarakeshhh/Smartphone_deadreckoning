"""Feature extraction shared by offline training and the live server.

Per-sample signals are produced by the SAME pipeline the server uses
(filters, orientation, step detector), so training and live prediction see
identical feature definitions.  Signals (see server.tracking_engine.SIGNAL_COLUMNS):
    acc_mag, gyro_mag, accf_mag, roll, pitch, steps_total, cadence
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from server.config import Settings, settings as default_settings
from server.tracking_engine import SIGNAL_COLUMNS, TrackingEngine

RAW_COLUMNS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]

FEATURE_NAMES = [
    "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max", "acc_mag_ptp",
    "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max",
    "accf_mean", "accf_std", "accf_roll_std_mean", "accf_roll_std_max", "accf_roll_mean_range",
    "roll_mean", "roll_std", "pitch_mean", "pitch_std",
    "steps_in_window", "cadence_mean", "peak_count", "dominant_freq_hz", "dominant_power_ratio",
]


def window_size(cfg: Settings = default_settings) -> tuple[int, int]:
    return int(cfg.ml_window_s * cfg.nominal_rate_hz), max(1, int(cfg.ml_hop_s * cfg.nominal_rate_hz))


def window_features(win: np.ndarray, fs: float, rolling: int = 25) -> dict:
    """Features of one window; ``win`` has shape (N, len(SIGNAL_COLUMNS))."""
    w = {name: win[:, i] for i, name in enumerate(SIGNAL_COLUMNS)}
    acc, gyro, accf = w["acc_mag"], w["gyro_mag"], w["accf_mag"]
    s = pd.Series(accf)
    rstd = s.rolling(rolling, min_periods=rolling).std().dropna()
    rmean = s.rolling(rolling, min_periods=rolling).mean().dropna()
    sig = accf - accf.mean()
    pk, _ = find_peaks(sig, height=max(0.5, 0.5 * sig.std()), distance=max(1, int(0.25 * fs)))
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    band = (freqs >= 0.5) & (freqs <= 5.0)
    if band.any() and spec[band].sum() > 1e-12:
        k = np.argmax(np.where(band, spec, 0.0))
        dom_f, dom_r = float(freqs[k]), float(spec[k] / spec[band].sum())
    else:
        dom_f, dom_r = 0.0, 0.0
    return {
        "acc_mag_mean": acc.mean(), "acc_mag_std": acc.std(), "acc_mag_min": acc.min(), "acc_mag_max": acc.max(),
        "acc_mag_ptp": np.ptp(acc),
        "gyro_mag_mean": gyro.mean(), "gyro_mag_std": gyro.std(), "gyro_mag_max": gyro.max(),
        "accf_mean": accf.mean(), "accf_std": accf.std(),
        "accf_roll_std_mean": rstd.mean() if len(rstd) else 0.0, "accf_roll_std_max": rstd.max() if len(rstd) else 0.0,
        "accf_roll_mean_range": np.ptp(rmean) if len(rmean) else 0.0,
        "roll_mean": w["roll"].mean(), "roll_std": w["roll"].std(),
        "pitch_mean": w["pitch"].mean(), "pitch_std": w["pitch"].std(),
        "steps_in_window": w["steps_total"][-1] - w["steps_total"][0], "cadence_mean": w["cadence"].mean(),
        "peak_count": float(len(pk)), "dominant_freq_hz": dom_f, "dominant_power_ratio": dom_r,
    }


def feature_vector(win: np.ndarray, fs: float) -> np.ndarray:
    f = window_features(win, fs)
    return np.array([f[n] for n in FEATURE_NAMES], float)


def derive_signals(df: pd.DataFrame, cfg: Settings = default_settings) -> np.ndarray:
    """Run the server pipeline offline over a raw CSV; returns per-sample signal matrix."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    eng = TrackingEngine(cfg, offline=True)
    eng.start()
    rows = []
    for rec in df[RAW_COLUMNS].to_dict("records"):
        ok, _ = eng.ingest(rec)
        if ok and eng.last_signals:
            rows.append([eng.last_signals[c] for c in SIGNAL_COLUMNS])
    return np.array(rows, float)


def make_windows(signals: np.ndarray, cfg: Settings = default_settings) -> np.ndarray:
    size, hop = window_size(cfg)
    if len(signals) < size:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.array([feature_vector(signals[i:i + size], cfg.nominal_rate_hz)
                     for i in range(0, len(signals) - size + 1, hop)])
