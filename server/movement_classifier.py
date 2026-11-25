"""Stationary detection (for ZUPT) and rule-based movement-state classification.

States: STATIONARY, WALKING, RUNNING, UNKNOWN.
UNKNOWN means "moving, but not with a gait-like periodic pattern" (e.g. the
phone is slid by hand) or "not enough information yet".  An optional ML
classifier (see ml/) is reported separately as ``ml_state``.
"""
from __future__ import annotations

from collections import deque

import numpy as np

STATIONARY, WALKING, RUNNING, UNKNOWN = "STATIONARY", "WALKING", "RUNNING", "UNKNOWN"


class StationaryDetector:
    """Zero-velocity detector using acceleration magnitude and gyro magnitude.

    Entering STATIONARY needs a window that is quiet for ``stat_enter_s``
    (std of |a| small, mean |gyro| small, |a| close to g).  Leaving is
    immediate when an instantaneous linear acceleration or rotation exceeds a
    threshold, so the start of a movement is not suppressed.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        n = max(5, int(self.cfg.stat_window_s * self.cfg.nominal_rate_hz))
        self.acc: deque = deque(maxlen=n)
        self.gyro: deque = deque(maxlen=n)
        self.lin: deque = deque(maxlen=n)
        self.quiet = 0
        self.stationary = False
        self.acc_std = 0.0
        self.gyro_mean = 0.0
        self.lin_mean = 0.0

    def update(self, acc_mag: float, gyro_mag: float, lin_acc_mag: float, g_est: float) -> bool:
        c = self.cfg
        self.acc.append(acc_mag)
        self.gyro.append(gyro_mag)
        self.lin.append(lin_acc_mag)
        if len(self.acc) < self.acc.maxlen:
            return False
        a = np.fromiter(self.acc, float)
        self.acc_std = float(a.std())
        self.gyro_mean = float(np.mean(self.gyro))
        # |a| alone barely changes during horizontal motion (sqrt(g^2+a^2) ~ g), so the
        # gravity-free world acceleration magnitude is checked as well.
        self.lin_mean = float(np.mean(self.lin))
        candidate = (self.acc_std < c.stat_acc_std and self.gyro_mean < c.stat_gyro_mean
                     and abs(float(a.mean()) - g_est) < c.stat_acc_dev
                     and self.lin_mean < c.stat_lin_acc_mean)
        if self.stationary:
            if lin_acc_mag > c.stat_exit_lin_acc or gyro_mag > c.stat_exit_gyro or not candidate:
                self.stationary = False
                self.quiet = 0
        else:
            self.quiet = self.quiet + 1 if candidate else 0
            if self.quiet >= max(1, int(c.stat_enter_s * c.nominal_rate_hz)):
                self.stationary = True
        return self.stationary


class MovementClassifier:
    """Debounced rule-based classifier built on stationary flag + step cadence."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = UNKNOWN
        self._cand = UNKNOWN
        self._count = 0
        self.sig_win: deque = deque(maxlen=max(10, int(2 * self.cfg.nominal_rate_hz)))

    def raw_state(self, stationary: bool, cadence_hz: float, sig_std: float) -> str:
        c = self.cfg
        if stationary:
            return STATIONARY
        if cadence_hz >= c.run_cadence_hz and sig_std > 2.5:
            return RUNNING
        if cadence_hz >= c.walk_cadence_min_hz:
            return WALKING
        return UNKNOWN

    def update(self, stationary: bool, cadence_hz: float, acc_dev: float) -> str:
        self.sig_win.append(acc_dev)
        sig_std = float(np.std(self.sig_win)) if len(self.sig_win) > 5 else 0.0
        cand = self.raw_state(stationary, cadence_hz, sig_std)
        if cand == self.state:
            self._count = 0
            self._cand = cand
            return self.state
        if cand == self._cand:
            self._count += 1
        else:
            self._cand, self._count = cand, 1
        # leaving STATIONARY is detected immediately by the detector itself
        need = 1 if self.state == STATIONARY else self.cfg.state_debounce_samples
        if self._count >= need:
            self.state, self._count = cand, 0
        return self.state
