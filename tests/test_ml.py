"""ML pipeline MECHANICS on synthetic recordings (written to a temp dir).

No model produced here is ever saved in ml/model and no accuracy figure from
synthetic data is claimed anywhere; real accuracy needs real recordings.
"""
import numpy as np
import pandas as pd
import pytest

from ml import train_model
from ml.features import FEATURE_NAMES, derive_signals, make_windows, window_features
from ml.predict import MotionPredictor
from server.tracking_engine import SIGNAL_COLUMNS
from tools import synthetic_imu as sim


def write_csv(path, segments, seed):
    a, yaw = sim.build(segments, 50)
    pk = sim.to_packets(a, yaw, 50, seed=seed)
    pd.DataFrame(pk).drop(columns="source").to_csv(path, index=False)


def test_feature_vector_shape_and_names():
    win = np.random.default_rng(0).normal(size=(100, len(SIGNAL_COLUMNS)))
    f = window_features(win, 50.0)
    assert list(f) == FEATURE_NAMES and all(np.isfinite(list(f.values())))


def test_make_windows_count():
    df = pd.DataFrame(sim.to_packets(*sim.build([("still", 10)], 50)))
    X = make_windows(derive_signals(df))
    assert X.shape == (9, len(FEATURE_NAMES))            # (500-100)/50+1


def test_training_refuses_without_enough_data(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_csv(raw / "stationary_a.csv", [("still", 30)], 1)
    rc = train_model.main(["--data-dir", str(raw), "--processed-dir", str(tmp_path / "p"), "--out", str(tmp_path / "m")])
    assert rc == 2 and not (tmp_path / "m").exists()
    assert "INSUFFICIENT" in capsys.readouterr().out


def test_training_pipeline_runs_and_saves_loadable_model(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (1, 2, 3):
        write_csv(raw / f"stationary_{i}.csv", [("still", 40)], i)
        write_csv(raw / f"walking_{i}.csv", [("still", 2), ("walk", 36, 1.7 + 0.1 * i, 2.5 + 0.3 * i), ("still", 2)], 10 + i)
    out = tmp_path / "model"
    rc = train_model.main(["--data-dir", str(raw), "--processed-dir", str(tmp_path / "p"), "--out", str(out)])
    assert rc == 0
    pred = MotionPredictor.load(out / "motion_classifier.joblib")
    assert (tmp_path / "p" / "features.csv").exists() and (out / "metrics.json").exists()
    walk = derive_signals(pd.read_csv(raw / "walking_1.csv"))[100:200]
    label, conf = pred.predict_window(walk)
    assert label in ("walking", "stationary") and 0 <= conf <= 1


def test_label_inference_from_sidecar_and_name(tmp_path):
    p = tmp_path / "session_001_foo.csv"
    p.write_text("x")
    assert train_model.infer_label(p) is None
    p.with_suffix(".json").write_text('{"label": "running"}')
    assert train_model.infer_label(p) == "running"
    assert train_model.infer_label(tmp_path / "walking_forward.csv") == "walking"
