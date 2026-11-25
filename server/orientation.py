"""Orientation estimation (sensor fusion).

COORDINATE SYSTEMS
------------------
Phone (body) frame - Android device coordinates, phone held in portrait:
    X : points to the RIGHT edge of the screen
    Y : points to the TOP edge of the screen
    Z : points OUT of the screen (towards the user's face)

World frame - ENU, right-handed:
    X : East
    Y : North
    Z : Up (opposite to gravity)

A phone lying flat, face up, with its top edge pointing North has
R = identity (body axes == world axes).  An at-rest accelerometer then reads
(0, 0, +9.81): the sensor measures the reaction to gravity, i.e. "up".

``R`` always maps BODY -> WORLD:  v_world = R @ v_body.

Euler angles (intrinsic Z-Y-X, R = Rz(yaw) Ry(pitch) Rx(roll)):
    roll  : rotation about phone X
    pitch : rotation about phone Y
    yaw   : rotation about world Z, counter-clockwise from East
Compass heading of the phone's top edge (clockwise from North) is also given.

Two fusion algorithms share one interface:
    MahonyFilter      complementary filter on SO(3) (default)
    KalmanTiltFilter  Kalman filter for roll/pitch (with gyro bias state)
                      + complementary magnetometer yaw (optional / advanced)
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


# ----------------------------------------------------------------- helpers
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def rot_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in radians for a body->world matrix."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def heading_deg(R: np.ndarray) -> float:
    """Compass heading (deg clockwise from North) of the phone's top edge (+Y)."""
    y_world = R[:, 1]
    return (math.degrees(math.atan2(y_world[0], y_world[1])) + 360.0) % 360.0


def initial_rotation(acc: np.ndarray, mag: np.ndarray | None) -> np.ndarray:
    """Static initial attitude from one accelerometer (+ magnetometer) reading.

    With a magnetometer the world X axis is magnetic East.  Without one, the
    world Y axis is defined as the projection of the phone's +Y axis at start
    ("relative heading" mode) so the trajectory is still self-consistent.
    """
    n = np.linalg.norm(acc)
    up = acc / n if n > EPS else np.array([0.0, 0.0, 1.0])
    east = None
    if mag is not None and np.linalg.norm(mag) > EPS:
        e = np.cross(mag, up)
        if np.linalg.norm(e) > 1e-3 * np.linalg.norm(mag):
            east = e / np.linalg.norm(e)
    if east is None:
        ref = np.array([0.0, 1.0, 0.0])
        north = ref - np.dot(ref, up) * up
        if np.linalg.norm(north) < 1e-3:               # phone standing on its top edge
            ref = np.array([1.0, 0.0, 0.0])
            north = ref - np.dot(ref, up) * up
        north /= np.linalg.norm(north)
        east = np.cross(north, up)
    north = np.cross(up, east)
    return np.vstack([east, north, up])


# ----------------------------------------------------------------- Mahony
class MahonyFilter:
    """Complementary (Mahony) attitude filter using gyro + accel + mag.

    The gyroscope is integrated for short-term accuracy.  The accelerometer
    (gravity direction) corrects roll/pitch drift, the magnetometer corrects
    yaw drift.  Corrections are weighted by "trust" values in [0, 1] so that
    dynamic acceleration or magnetic disturbances are ignored.
    """

    name = "mahony"

    def __init__(self, kp_acc: float = 2.0, kp_mag: float = 1.0, ki: float = 0.02):
        self.kp_acc, self.kp_mag, self.ki = kp_acc, kp_mag, ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral = np.zeros(3)

    def initialize(self, acc, mag=None) -> None:
        self.q = rot_to_quat(initial_rotation(np.asarray(acc, float), None if mag is None else np.asarray(mag, float)))
        self.integral[:] = 0.0

    def update(self, gyro, acc, mag, dt: float, acc_trust: float = 1.0, mag_trust: float = 1.0) -> None:
        gyro = np.asarray(gyro, float)
        R = quat_to_rot(self.q)
        v_up = R[2, :]                       # world "up" expressed in body frame
        err = np.zeros(3)
        an = np.linalg.norm(acc)
        if acc_trust > 0 and an > 1e-3:
            err += self.kp_acc * acc_trust * np.cross(np.asarray(acc, float) / an, v_up)
        if mag is not None and mag_trust > 0:
            mn = np.linalg.norm(mag)
            if mn > 1e-3:
                m = np.asarray(mag, float) / mn
                h = R @ m
                b = np.array([0.0, math.hypot(h[0], h[1]), h[2]])   # reference field: north + vertical
                w_body = R.T @ b
                e_m = np.cross(m, w_body)
                # only allow the magnetometer to change YAW (rotation about "up"),
                # so magnetic disturbances can never tilt the attitude
                err += self.kp_mag * mag_trust * np.dot(e_m, v_up) * v_up
        if self.ki > 0:
            self.integral += self.ki * err * dt
            np.clip(self.integral, -0.5, 0.5, out=self.integral)
        omega = gyro + err + self.integral
        self.q = quat_normalize(self.q + 0.5 * quat_mult(self.q, np.array([0.0, *omega])) * dt)

    def rotation(self) -> np.ndarray:
        return quat_to_rot(self.q)

    def euler(self) -> tuple[float, float, float]:
        return rot_to_euler(self.rotation())


# ----------------------------------------------------------------- Kalman
class _AngleKalman:
    """1-D Kalman filter: state [angle, gyro_bias]."""

    def __init__(self, q_angle: float, q_bias: float, r_meas: float):
        self.q_angle, self.q_bias, self.r = q_angle, q_bias, r_meas
        self.angle = 0.0
        self.bias = 0.0
        self.P = np.eye(2) * 0.1

    def update(self, rate: float, meas: float | None, dt: float, r_scale: float = 1.0) -> float:
        # predict
        self.angle += (rate - self.bias) * dt
        self.P[0, 0] += dt * (dt * self.P[1, 1] - self.P[0, 1] - self.P[1, 0] + self.q_angle)
        self.P[0, 1] -= dt * self.P[1, 1]
        self.P[1, 0] -= dt * self.P[1, 1]
        self.P[1, 1] += self.q_bias * dt
        if meas is None:                       # measurement gated out (phone is accelerating)
            return self.angle
        # correct (wrap-safe innovation)
        innov = (meas - self.angle + math.pi) % (2 * math.pi) - math.pi
        S = self.P[0, 0] + self.r * r_scale
        K = self.P[:, 0] / S
        self.angle += K[0] * innov
        self.bias += K[1] * innov
        p00, p01 = self.P[0, 0], self.P[0, 1]
        self.P[0, 0] -= K[0] * p00
        self.P[0, 1] -= K[0] * p01
        self.P[1, 0] -= K[1] * p00
        self.P[1, 1] -= K[1] * p01
        return self.angle


class KalmanTiltFilter:
    """Optional advanced fusion: Kalman roll/pitch + magnetometer-corrected yaw.

    Roll and pitch each have their own Kalman filter (angle + gyro-bias state);
    the accelerometer is the measurement and its noise is inflated when the
    phone is accelerating (low ``acc_trust``).  Yaw is integrated from the
    gyro and pulled towards the tilt-compensated magnetometer heading with a
    complementary gain.  Euler-angle singularity at pitch = +/-90 deg is
    avoided by clamping.
    """

    name = "kalman"

    def __init__(self, q_angle=0.001, q_bias=0.00001, r_meas=1.0, yaw_gain: float = 0.8):
        self.kr = _AngleKalman(q_angle, q_bias, r_meas)
        self.kp = _AngleKalman(q_angle, q_bias, r_meas)
        self.yaw = 0.0
        self.yaw_gain = yaw_gain

    def initialize(self, acc, mag=None) -> None:
        R = initial_rotation(np.asarray(acc, float), None if mag is None else np.asarray(mag, float))
        roll, pitch, yaw = rot_to_euler(R)
        self.kr.angle, self.kp.angle, self.yaw = roll, pitch, yaw
        self.kr.bias = self.kp.bias = 0.0

    def update(self, gyro, acc, mag, dt: float, acc_trust: float = 1.0, mag_trust: float = 1.0) -> None:
        gx, gy, gz = (float(v) for v in gyro)
        roll, pitch = self.kr.angle, self.kp.angle
        pitch_c = max(-1.45, min(1.45, pitch))
        tp = math.tan(pitch_c)
        roll_rate = gx + math.sin(roll) * tp * gy + math.cos(roll) * tp * gz
        pitch_rate = math.cos(roll) * gy - math.sin(roll) * gz
        yaw_rate = (math.sin(roll) * gy + math.cos(roll) * gz) / math.cos(pitch_c)

        a = np.asarray(acc, float)
        an = np.linalg.norm(a)
        # Gate the accelerometer measurement: while the phone accelerates, the accelerometer
        # does not point "up", so the Kalman filter only predicts (gyro) and skips the update.
        use_acc = acc_trust >= 0.9 and an > 1e-3
        if use_acc:
            roll_meas = math.atan2(a[1], a[2])
            pitch_meas = -math.asin(max(-1.0, min(1.0, a[0] / an)))
            r_scale = 1.0 / (acc_trust * acc_trust)
        else:
            roll_meas = pitch_meas = None
            r_scale = 1.0
        self.kr.update(roll_rate, roll_meas, dt, r_scale)
        self.kp.update(pitch_rate, pitch_meas, dt, r_scale)

        self.yaw += yaw_rate * dt
        if mag is not None and mag_trust > 0 and np.linalg.norm(mag) > 1e-3:
            R_m = initial_rotation(a, np.asarray(mag, float))
            yaw_meas = rot_to_euler(R_m)[2]
            innov = (yaw_meas - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += min(1.0, self.yaw_gain * mag_trust * dt) * innov
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

    def rotation(self) -> np.ndarray:
        return euler_to_rot(self.kr.angle, self.kp.angle, self.yaw)

    def euler(self) -> tuple[float, float, float]:
        return self.kr.angle, self.kp.angle, self.yaw


def make_orientation_filter(cfg):
    if cfg.fusion_mode == "kalman":
        return KalmanTiltFilter(cfg.kalman_q_angle, cfg.kalman_q_bias, cfg.kalman_r_meas)
    return MahonyFilter(cfg.mahony_kp_acc, cfg.mahony_kp_mag, cfg.mahony_ki)
