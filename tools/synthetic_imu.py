"""Synthetic IMU signal generator - FOR TESTS AND DEVELOPMENT ONLY.

Generates what a phone's accelerometer / gyroscope / magnetometer would read
for a *known, prescribed* motion (phone flat, face up, rotating only about
world Z).  Used by unit tests and by tools/phone_simulator.py to exercise the
pipeline when no phone is available.  Output is always tagged
``source="simulator"`` by the simulator so the dashboard shows a banner; it is
never used for the real demo and no accuracy claims are derived from it.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.80665
MAG_WORLD = np.array([0.0, 22.0, -42.0])      # ENU magnetic field (uT), mid-latitude


def min_jerk_accel(distance: float, T: float, n: int) -> np.ndarray:
    """Acceleration profile of a minimum-jerk move (starts/ends at rest)."""
    s = np.linspace(0, 1, n, endpoint=False)
    return distance / T ** 2 * (60 * s - 180 * s ** 2 + 120 * s ** 3)


def build(segments, fs: float = 50.0):
    """segments: list of tuples
        ("still", T)
        ("move", heading_deg_from_east, distance_m, T)
        ("rotate", total_deg, T)                    # yaw about world Z
        ("walk", T, step_hz, amp_ms2)               # vertical bounce only (no net motion)
    returns a_world (N,3), yaw (N,)
    """
    a_list, yaw_list, yaw = [], [], 0.0
    for seg in segments:
        kind = seg[0]
        if kind == "still":
            n = int(seg[1] * fs)
            a_list.append(np.zeros((n, 3)))
            yaw_list.append(np.full(n, yaw))
        elif kind == "move":
            _, hdg, dist, T = seg
            n = int(T * fs)
            prof = min_jerk_accel(dist, T, n)
            h = math.radians(hdg)
            a_list.append(np.column_stack([prof * math.cos(h), prof * math.sin(h), np.zeros(n)]))
            yaw_list.append(np.full(n, yaw))
        elif kind == "rotate":
            _, deg, T = seg
            n = int(T * fs)
            s = np.linspace(0, 1, n, endpoint=False)
            yaw_list.append(yaw + math.radians(deg) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))
            a_list.append(np.zeros((n, 3)))
            yaw += math.radians(deg)
        elif kind == "walk":
            _, T, f, amp = seg
            n = int(T * fs)
            t = np.arange(n) / fs
            a = np.zeros((n, 3))
            a[:, 2] = amp * np.sin(2 * math.pi * f * t)
            a_list.append(a)
            yaw_list.append(np.full(n, yaw))
        else:
            raise ValueError(kind)
    return np.vstack(a_list), np.concatenate(yaw_list)


def to_packets(a_world, yaw, fs=50.0, noise_acc=0.03, noise_gyro=0.002, noise_mag=0.3,
               acc_bias=(0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0), t0_ms=1_000_000.0, seed=0,
               source="simulator", drop_every=0):
    rng = np.random.default_rng(seed)
    n = len(a_world)
    yaw_dot = np.gradient(yaw, 1.0 / fs)
    out = []
    for i in range(n):
        if drop_every and i % drop_every == drop_every - 1:
            continue
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        acc = R.T @ (a_world[i] + np.array([0, 0, G])) + np.asarray(acc_bias) + rng.normal(0, noise_acc, 3)
        gyro = np.array([0, 0, yaw_dot[i]]) + np.asarray(gyro_bias) + rng.normal(0, noise_gyro, 3)
        mag = R.T @ MAG_WORLD + rng.normal(0, noise_mag, 3)
        out.append({"timestamp": t0_ms + i * 1000.0 / fs, "acc_x": acc[0], "acc_y": acc[1], "acc_z": acc[2],
                    "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2],
                    "mag_x": mag[0], "mag_y": mag[1], "mag_z": mag[2], "source": source})
    return out


SCENARIOS = {
    "square": [("still", 3), ("move", 0, 1.0, 2.0), ("still", 1), ("move", 90, 1.0, 2.0), ("still", 1),
               ("move", 180, 1.0, 2.0), ("still", 1), ("move", 270, 1.0, 2.0), ("still", 2)],
    "forward": [("still", 3), ("move", 90, 2.0, 3.0), ("still", 3)],
    "stationary": [("still", 15)],
    "walk": [("still", 3), ("walk", 12, 1.8, 3.0), ("still", 3)],
}
