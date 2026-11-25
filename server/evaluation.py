"""Accuracy evaluation: compare an ESTIMATED trajectory against GROUND TRUTH.

GROUND TRUTH must come from an independent method (tape-measured waypoints,
a marked floor path, overhead camera / ArUco tracking, motion capture, ...),
never from this system.  Estimated data comes from
data/processed/session_NNN_<name>_trajectory.csv (columns t_s,x_m,y_m,...).

Metrics (all in metres unless stated):
  est_path_length / truth_path_length / distance_error / distance_error_pct
  final_position_error     |end(est) - end(truth)|
  drift_pct                final_position_error / truth_path_length * 100
  position_error_{mean,rms,max}   point-wise error after pairing points
  rigid_aligned_rmse       RMSE after best 2-D rotation+translation (shape error,
                           removes heading/origin mismatch)
  average_speed_est / max_speed_est   from the estimated trajectory timestamps

Pairing: if both trajectories have timestamps, truth is interpolated at the
estimate's times; otherwise points are paired by normalised arc length.

CLI examples
  python -m server.evaluation --est data/processed/session_001_square_trajectory.csv \
         --truth-waypoints "0,0;3,0;3,3;0,3;0,0"
  python -m server.evaluation --est est.csv --truth truth.csv   (truth columns: x_m,y_m[,t_s])
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def _resample_arclength(xy: np.ndarray, n: int) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(xy[:1], n, axis=0)
    s = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])])


def rigid_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2-D rotation+translation of src onto dst (Kabsch)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return (R @ (src - mu_s).T).T + mu_d


def waypoints_to_truth(waypoints: str, n_per_segment: int = 50) -> np.ndarray:
    pts = np.array([[float(v) for v in p.split(",")] for p in waypoints.split(";") if p.strip()])
    out = [np.linspace(pts[i], pts[i + 1], n_per_segment, endpoint=False) for i in range(len(pts) - 1)]
    return np.vstack(out + [pts[-1:]])


def evaluate(est_xy, truth_xy, est_t=None, truth_t=None) -> dict:
    est, truth = np.asarray(est_xy, float), np.asarray(truth_xy, float)
    if len(est) < 2 or len(truth) < 2:
        raise ValueError("need at least 2 points in both trajectories")
    el, tl = path_length(est), path_length(truth)
    if est_t is not None and truth_t is not None:
        tt = np.asarray(truth_t, float) - truth_t[0]
        et = np.asarray(est_t, float) - est_t[0]
        truth_p = np.column_stack([np.interp(et, tt, truth[:, 0]), np.interp(et, tt, truth[:, 1])])
        est_p, pairing = est, "time"
    else:
        n = max(len(est), len(truth), 200)
        est_p, truth_p, pairing = _resample_arclength(est, n), _resample_arclength(truth, n), "arc-length"
    err = np.linalg.norm(est_p - truth_p, axis=1)
    aligned = rigid_align(est_p, truth_p)
    res = {
        "pairing": pairing,
        "est_path_length_m": round(el, 3), "truth_path_length_m": round(tl, 3),
        "distance_error_m": round(el - tl, 3),
        "distance_error_pct": round((el - tl) / tl * 100, 2) if tl > 0 else None,
        "final_position_error_m": round(float(np.linalg.norm(est[-1] - truth[-1])), 3),
        "drift_pct_of_path": round(float(np.linalg.norm(est[-1] - truth[-1])) / tl * 100, 2) if tl > 0 else None,
        "position_error_mean_m": round(float(err.mean()), 3),
        "position_error_rms_m": round(float(np.sqrt((err ** 2).mean())), 3),
        "position_error_max_m": round(float(err.max()), 3),
        "rigid_aligned_rmse_m": round(float(np.sqrt(((aligned - truth_p) ** 2).sum(1).mean())), 3),
    }
    if est_t is not None and len(est_t) > 1:
        et = np.asarray(est_t, float)
        seg = np.linalg.norm(np.diff(est, axis=0), axis=1) / np.maximum(np.diff(et), 1e-9)
        dur = et[-1] - et[0]
        res["average_speed_est_mps"] = round(el / dur, 3) if dur > 0 else None
        res["max_speed_est_mps"] = round(float(seg.max()), 3)
    return res


def load_estimated(path: str):
    df = pd.read_csv(path)
    return df[["x_m", "y_m"]].to_numpy(), df["t_s"].to_numpy() if "t_s" in df else None


def load_truth(path: str):
    df = pd.read_csv(path)
    return df[["x_m", "y_m"]].to_numpy(), df["t_s"].to_numpy() if "t_s" in df else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", required=True, help="estimated trajectory CSV (t_s,x_m,y_m,...)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--truth", help="ground-truth CSV (x_m,y_m[,t_s])")
    g.add_argument("--truth-waypoints", help='tape-measured waypoints "x,y;x,y;..." in metres')
    a = ap.parse_args()
    est, est_t = load_estimated(a.est)
    if a.truth:
        truth, truth_t = load_truth(a.truth)
    else:
        truth, truth_t = waypoints_to_truth(a.truth_waypoints), None
    print(json.dumps(evaluate(est, truth, est_t, truth_t), indent=2))
    print("\nNOTE: 'truth' is the externally supplied reference; 'est' is IMU dead reckoning (not GPS).")


if __name__ == "__main__":
    main()
