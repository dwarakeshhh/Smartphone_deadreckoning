"""Pipeline tests on SYNTHETIC signals generated from a known, prescribed motion.

These check that the algorithms are wired correctly (sign conventions, ZUPT,
calibration, ...).  They say nothing about real-world accuracy.
"""
import math

import numpy as np
import pytest

from server.config import Settings
from server.tracking_engine import CALIBRATING, TRACKING, TrackingEngine
from tools import synthetic_imu as sim
from tests.conftest import run_scenario


def test_stationary_phone_does_not_drift(cfg):
    eng = run_scenario(cfg, [("still", 30)], acc_bias=(0.04, -0.03, 0.02), gyro_bias=(0.01, -0.01, 0.005))
    st = eng.stats()
    assert math.hypot(st["x_m"], st["y_m"]) < 0.05
    assert st["total_distance_m"] < 0.05
    assert eng.state == "STATIONARY"


@pytest.mark.parametrize("heading,expect", [(0, (2, 0)), (90, (0, 2)), (180, (-2, 0)), (270, (0, -2))])
def test_straight_move_direction_and_distance(cfg, heading, expect):
    eng = run_scenario(cfg, [("still", 3), ("move", heading, 2.0, 3.0), ("still", 3)])
    st = eng.stats()
    assert st["x_m"] == pytest.approx(expect[0], abs=0.4)
    assert st["y_m"] == pytest.approx(expect[1], abs=0.4)
    # sign must be right: the dominant axis moved the right way
    assert (st["x_m"] * expect[0] + st["y_m"] * expect[1]) > 1.4


def test_square_path_closes(cfg):
    eng = run_scenario(cfg, sim.SCENARIOS["square"])
    st = eng.stats()
    assert math.hypot(st["x_m"], st["y_m"]) < 0.5
    assert st["total_distance_m"] == pytest.approx(4.0, abs=0.8)


def test_move_while_rotated_uses_orientation(cfg):
    """Phone yawed 90 deg, then pushed along WORLD +X: needs the rotation to be applied."""
    eng = run_scenario(cfg, [("still", 3), ("rotate", 90, 2.0), ("still", 3), ("move", 0, 1.5, 2.5), ("still", 3)])
    st = eng.stats()
    assert st["x_m"] == pytest.approx(1.5, abs=0.4)
    assert abs(st["y_m"]) < 0.4


def test_step_detection_on_synthetic_gait(cfg):
    eng = run_scenario(cfg, [("still", 3), ("walk", 12, 1.8, 3.0), ("still", 3)])
    expected = 12 * 1.8
    assert abs(eng.stats()["steps"] - expected) <= 3


def test_no_steps_when_still_or_noise(cfg):
    eng = run_scenario(cfg, [("still", 20)], noise_acc=0.05)
    assert eng.stats()["steps"] == 0


def test_calibration_phase_waits_for_stillness(cfg):
    eng = TrackingEngine(cfg)
    eng.start()
    assert eng.phase == CALIBRATING
    a, yaw = sim.build([("still", 3)], 50)
    for p in sim.to_packets(a, yaw, 50)[:30]:
        eng.ingest(p)
    assert eng.phase == CALIBRATING                      # <1.5 s of data
    for p in sim.to_packets(a, yaw, 50, t0_ms=2_000_000.0):
        eng.ingest(p)
    assert eng.phase == TRACKING
    assert eng.quality()["calibration"]["g_est"] == pytest.approx(9.80665, abs=0.05)


def test_data_gap_zeroes_velocity_and_is_counted(cfg):
    eng = TrackingEngine(cfg)
    eng.start()
    a, yaw = sim.build([("still", 3), ("move", 0, 3.0, 2.0)], 50)
    pk = sim.to_packets(a, yaw, 50)
    for p in pk[:200]:
        eng.ingest(p)
    for p in pk[200:]:
        q = dict(p)
        q["timestamp"] += 5000                            # 5 s Wi-Fi outage in the middle of the move
        eng.ingest(q)
    assert eng.quality()["gaps"] >= 1


def test_reset_scopes(cfg):
    eng = run_scenario(cfg, [("still", 3), ("move", 0, 1.0, 2.0), ("still", 2)])
    assert eng.stats()["total_distance_m"] > 0.3
    eng.reset_position()
    s = eng.stats()
    assert s["x_m"] == 0 and s["y_m"] == 0 and s["total_distance_m"] > 0.3     # distance kept
    eng.clear_data()
    assert eng.trajectory()["count"] == 0
    eng.reset_session()
    assert eng.stats()["total_distance_m"] == 0 and eng.phase == CALIBRATING


def test_rejected_packets_are_counted(cfg):
    eng = TrackingEngine(cfg)
    ok, err = eng.ingest({"timestamp": 1})
    assert not ok and "missing" in err
    assert eng.packets_rejected == 1


def test_kalman_mode_is_experimental_but_directionally_correct(cfg):
    """Kalman fusion is EXPERIMENTAL: on this synthetic push it under-estimates the distance
    (about 60 % vs ~90 % for the default Mahony filter), so only direction is asserted."""
    k = Settings(fusion_mode="kalman")
    eng = run_scenario(k, [("still", 3), ("move", 90, 2.0, 3.0), ("still", 3)])
    st = eng.stats()
    assert st["y_m"] > 0.8 and abs(st["x_m"]) < 0.4
