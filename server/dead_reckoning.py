"""Inertial dead reckoning with practical drift mitigation.

Core equations (per sample, world frame, a = gravity-free linear acceleration):

    velocity_new = velocity_old + a * dt
    position_new = position_old + velocity_new * dt

Drift mitigation (IMU-only integration is NEVER perfectly accurate):
  1. Gravity removal and bias learning happen BEFORE this module (engine).
  2. Horizontal dead-band: |a_xy| below ``acc_deadband`` is treated as zero.
  3. Zero-velocity update (ZUPT): while the phone is detected STATIONARY the
     velocity is forced to zero.
  4. Leaky integrator: velocity decays with time constant ``velocity_leak_tau_s``
     so un-observed acceleration errors cannot grow without bound.
  5. Speed clamp for physically implausible estimates.
Because of double integration, position error grows roughly quadratically
with time between ZUPTs.  Use the reset buttons and keep runs short.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class DeadReckoner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset_all()

    def reset_all(self) -> None:
        self._pending: deque = deque(maxlen=max(1, int(self.cfg.zupt_replay_s * self.cfg.nominal_rate_hz)))
        self._was_stationary = False
        self.reset_position()
        self.distance = 0.0           # horizontal path length of the estimated trajectory (m)
        self.max_speed = 0.0
        self.zupt_count = 0
        self.time_since_zupt = 0.0

    def reset_position(self) -> None:
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self._pending = deque(maxlen=self._pending.maxlen) if hasattr(self, "_pending") else deque()

    @property
    def speed(self) -> float:
        return float(math.hypot(self.vel[0], self.vel[1]))

    def _integrate(self, a: np.ndarray, dt: float) -> None:
        c = self.cfg
        self.vel = self.vel + a * dt                      # velocity_new = velocity_old + a*dt
        if c.velocity_leak_tau_s > 0:
            self.vel[:2] *= math.exp(-dt / c.velocity_leak_tau_s)
        if c.vertical_leak_tau_s > 0:
            self.vel[2] *= math.exp(-dt / c.vertical_leak_tau_s)
        sp = float(np.linalg.norm(self.vel))
        if sp > c.max_speed:
            self.vel *= c.max_speed / sp
        step = self.vel * dt                              # position_new = position_old + v*dt
        self.pos = self.pos + step
        self.distance += math.hypot(step[0], step[1])
        self.max_speed = max(self.max_speed, self.speed)

    def update(self, lin_acc_world: np.ndarray, dt: float, stationary: bool) -> None:
        a = np.array(lin_acc_world, float)
        if math.hypot(a[0], a[1]) < self.cfg.acc_deadband:
            a[0] = a[1] = 0.0
        if stationary:
            if np.any(self.vel):
                self.zupt_count += 1
            self.vel[:] = 0.0
            self.time_since_zupt = 0.0
            self._pending.append((a, dt))                 # remember the last moments of stillness
            self._was_stationary = True
            return
        if self._was_stationary:
            # The stationary detector needs a few samples to notice that motion has started.
            # The velocity that built up during that lag was zeroed by the ZUPT, so replay it.
            self._was_stationary = False
            for pa, pdt in self._pending:
                self._integrate(pa, pdt)
            self._pending.clear()
        self.time_since_zupt += dt
        self._integrate(a, dt)
