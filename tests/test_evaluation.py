import numpy as np
import pytest

from server.evaluation import evaluate, path_length, rigid_align, waypoints_to_truth


def test_path_length_and_waypoints():
    truth = waypoints_to_truth("0,0;3,0;3,3;0,3;0,0")
    assert path_length(truth) == pytest.approx(12.0)


def test_perfect_estimate_has_zero_error():
    truth = waypoints_to_truth("0,0;2,0;2,2")
    m = evaluate(truth, truth)
    assert m["final_position_error_m"] == 0 and m["position_error_rms_m"] == 0 and m["distance_error_m"] == 0


def test_known_scale_error_and_drift():
    truth = waypoints_to_truth("0,0;4,0")
    est = truth * 1.1                                    # 10 % too long
    m = evaluate(est, truth)
    assert m["distance_error_pct"] == pytest.approx(10.0, abs=0.2)
    assert m["final_position_error_m"] == pytest.approx(0.4, abs=0.01)
    assert m["drift_pct_of_path"] == pytest.approx(10.0, abs=0.3)


def test_rigid_alignment_removes_rotation():
    truth = waypoints_to_truth("0,0;3,0;3,3")
    th = np.radians(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    est = truth @ R.T + [5, -2]
    m = evaluate(est, truth)
    assert m["rigid_aligned_rmse_m"] < 1e-6 and m["position_error_rms_m"] > 1.0


def test_speed_metrics_with_time():
    t = np.linspace(0, 10, 101)
    est = np.column_stack([t * 0.5, np.zeros_like(t)])
    m = evaluate(est, est, est_t=t, truth_t=t)
    assert m["average_speed_est_mps"] == pytest.approx(0.5) and m["max_speed_est_mps"] == pytest.approx(0.5)
