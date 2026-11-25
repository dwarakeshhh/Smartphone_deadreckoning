import math

import numpy as np
import pytest

from server.orientation import (KalmanTiltFilter, MahonyFilter, euler_to_rot, heading_deg, initial_rotation,
                                quat_to_rot, rot_to_euler, rot_to_quat)

G = 9.80665
MAG = np.array([0.0, 22.0, -42.0])


def body_readings(R):
    return R.T @ np.array([0, 0, G]), R.T @ MAG


def test_euler_roundtrip():
    R = euler_to_rot(0.3, -0.2, 1.1)
    r, p, y = rot_to_euler(R)
    assert (r, p, y) == pytest.approx((0.3, -0.2, 1.1))
    assert quat_to_rot(rot_to_quat(R)) == pytest.approx(R, abs=1e-9)


def test_flat_phone_is_identity():
    acc, mag = body_readings(np.eye(3))
    assert initial_rotation(acc, mag) == pytest.approx(np.eye(3), abs=1e-9)


def test_initial_rotation_recovers_attitude():
    R_true = euler_to_rot(math.radians(20), math.radians(-35), math.radians(70))
    acc, mag = body_readings(R_true)
    assert initial_rotation(acc, mag) == pytest.approx(R_true, abs=1e-6)


def test_no_magnetometer_gives_consistent_tilt():
    R_true = euler_to_rot(0.2, 0.3, 0.0)
    acc, _ = body_readings(R_true)
    R = initial_rotation(acc, None)
    assert R[2] == pytest.approx(R_true[2], abs=1e-9)           # "up" is exact


@pytest.mark.parametrize("make", [lambda: MahonyFilter(0.5, 0.5, 0.0), lambda: KalmanTiltFilter()])
def test_filters_converge_from_wrong_start(make):
    f = make()
    f.initialize(np.array([0, 0, G]), MAG)                       # start flat, true pose is tilted
    R_true = euler_to_rot(math.radians(25), math.radians(-15), math.radians(40))
    acc, mag = body_readings(R_true)
    for _ in range(1500):                                        # 30 s at 50 Hz, gyro = 0
        f.update(np.zeros(3), acc, mag, 0.02)
    roll, pitch, yaw = f.euler()
    assert math.degrees(roll) == pytest.approx(25, abs=1.0)
    assert math.degrees(pitch) == pytest.approx(-15, abs=1.0)
    assert math.degrees(yaw) == pytest.approx(40, abs=3.0)


@pytest.mark.parametrize("make", [lambda: MahonyFilter(0.5, 0.5, 0.0), lambda: KalmanTiltFilter()])
def test_gyro_integration_tracks_yaw_rotation(make):
    f = make()
    f.initialize(np.array([0, 0, G]), MAG)
    yaw = 0.0
    for _ in range(100):                                         # 2 s at 90 deg/s about Z
        yaw += math.radians(90) * 0.02
        R = euler_to_rot(0, 0, yaw)
        acc, mag = body_readings(R)
        f.update(np.array([0, 0, math.radians(90)]), acc, mag, 0.02)
    assert math.degrees(f.euler()[2]) == pytest.approx(180, abs=4.0) or abs(math.degrees(f.euler()[2])) > 170


def test_mag_cannot_tilt_attitude():
    f = MahonyFilter(0.5, 5.0, 0.0)
    f.initialize(np.array([0, 0, G]), MAG)
    bad_mag = np.array([40.0, 5.0, 30.0])                         # strong disturbance
    for _ in range(500):
        f.update(np.zeros(3), np.array([0, 0, G]), bad_mag, 0.02)
    roll, pitch, _ = f.euler()
    assert abs(math.degrees(roll)) < 0.5 and abs(math.degrees(pitch)) < 0.5


def test_heading_convention():
    assert heading_deg(np.eye(3)) == pytest.approx(0.0)                       # top edge North
    assert heading_deg(euler_to_rot(0, 0, math.radians(-90))) == pytest.approx(90.0)   # clockwise -> East
