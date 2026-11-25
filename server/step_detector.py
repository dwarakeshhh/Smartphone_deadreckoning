"""Step detection from accelerometer magnitude (online peak detection).

Method
------
1. Signal  s(t) = |a_filtered| - g_est  (gravity-free magnitude, orientation independent).
2. Adaptive threshold  thr = max(step_min_peak, mean + k*std)  over the last 2 s,
   so small ripples are rejected but the detector adapts to gait intensity.
3. A sample is a peak if it is a local maximum above ``thr``.
4. Plausibility: peaks closer than ``step_min_interval_s`` are ignored; a gap
   longer than ``step_max_interval_s`` ends the walking sequence.
5. Confirmation: ``step_confirm_count`` consecutive peaks with similar spacing
   are required before steps are counted (the first ones are added
   retroactively), which prevents single bumps or hand jerks from counting.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class StepDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        n = max(10, int(2.0 * self.cfg.nominal_rate_hz))
        self.win: deque = deque(maxlen=n)
        self.prev2 = self.prev1 = None          # (t, value)
        self.last_peak_t: float | None = None
        self.seq: list[float] = []
        self.confirmed = False
        self.steps = 0
        self.intervals: deque = deque(maxlen=5)
        self.last_step_t: float | None = None
        self.last_amp = 0.0

    # -------------------------------------------------------------- API
    def update(self, t: float, acc_mag_f: float, g_est: float) -> int:
        """Feed one sample; returns the number of NEW steps counted."""
        c = self.cfg
        v = acc_mag_f - g_est
        self.win.append(v)
        new_steps = 0
        if self.prev1 is not None and self.prev2 is not None:
            t1, v1 = self.prev1
            if v1 > self.prev2[1] and v1 >= v:
                arr = np.fromiter(self.win, float)
                thr = max(c.step_min_peak, float(arr.mean() + c.step_peak_std_factor * arr.std()))
                if v1 > thr:
                    new_steps = self._peak(t1, v1)
        self.prev2, self.prev1 = self.prev1, (t, v)
        if self.last_step_t is not None and t - self.last_step_t > c.step_max_interval_s:
            self.confirmed = False
            self.seq = []
            self.intervals.clear()
        return new_steps

    def cadence_hz(self, now: float | None = None) -> float:
        """Steps per second from the median of the recent step intervals."""
        if not self.confirmed or not self.intervals:
            return 0.0
        if now is not None and self.last_step_t is not None and now - self.last_step_t > self.cfg.step_max_interval_s:
            return 0.0
        med = float(np.median(self.intervals))
        return 1.0 / med if med > 0 else 0.0

    # ---------------------------------------------------------- internals
    def _peak(self, t: float, amp: float) -> int:
        c = self.cfg
        if self.last_peak_t is not None:
            gap = t - self.last_peak_t
            if gap < c.step_min_interval_s:
                return 0
            if gap > c.step_max_interval_s:
                self.seq, self.confirmed = [], False
                self.intervals.clear()
        if self.last_peak_t is not None and (t - self.last_peak_t) <= c.step_max_interval_s:
            self.intervals.append(t - self.last_peak_t)
        self.last_peak_t = t
        self.last_amp = amp
        self.seq.append(t)
        if self.confirmed:
            self.steps += 1
            self.last_step_t = t
            return 1
        if len(self.seq) >= c.step_confirm_count:
            ivs = np.diff(self.seq[-c.step_confirm_count:])
            if ivs.max() / max(ivs.min(), 1e-6) < 1.8:
                self.confirmed = True
                n = len(self.seq)
                self.steps += n
                self.last_step_t = t
                return n
            self.seq = self.seq[-(c.step_confirm_count - 1):]
        return 0
