"""Train the movement classifier (stationary / walking / running) from REAL recordings.

  python -m ml.train_model [--data-dir data/raw] [--out ml/model]

Data: every CSV in data/raw (written by the recording mode).  The class label
comes from the sidecar <file>.json ("label") or, failing that, from the file
name containing 'stationary', 'walking' or 'running'.

Honesty rules implemented here:
  * No synthetic or fabricated data is ever used.
  * Evaluation is leave-SESSION-out cross-validation (windows overlap, so a
    random window split would leak and inflate accuracy).
  * If there is not enough real data (>= 2 sessions and >= --min-windows
    windows per class, >= 2 classes) NO model is written; the script explains
    what to record.
Models compared: Random Forest vs Logistic Regression (macro-F1 decides).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_NAMES, derive_signals, make_windows
from server.config import settings

CLASSES = ("stationary", "walking", "running")


def infer_label(csv_path: Path) -> str | None:
    meta = csv_path.with_suffix(".json")
    if meta.exists():
        lab = (json.loads(meta.read_text()).get("label") or "").lower()
        if lab in CLASSES:
            return lab
    name = csv_path.stem.lower()
    for c in CLASSES:
        if c in name:
            return c
    return None


def build_dataset(data_dir: Path, processed_dir: Path):
    X, y, g, used, skipped = [], [], [], [], []
    for p in sorted(data_dir.glob("*.csv")):
        label = infer_label(p)
        if label is None:
            skipped.append(f"{p.name} (no label: name it e.g. walking_*.csv or set the label when recording)")
            continue
        df = pd.read_csv(p)
        if len(df) > 1:
            rate = (len(df) - 1) / ((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1000.0)
            if abs(rate - settings.nominal_rate_hz) / settings.nominal_rate_hz > 0.25:
                skipped.append(f"{p.name} (sampling rate {rate:.1f} Hz != {settings.nominal_rate_hz} Hz)")
                continue
        feats = make_windows(derive_signals(df))
        if len(feats) == 0:
            skipped.append(f"{p.name} (shorter than one {settings.ml_window_s:.0f}s window)")
            continue
        X.append(feats)
        y += [label] * len(feats)
        g += [p.stem] * len(feats)
        used.append({"file": p.name, "label": label, "windows": len(feats)})
    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.array([]), np.array([]), used, skipped
    X = np.vstack(X)
    out = pd.DataFrame(X, columns=FEATURE_NAMES)
    out["label"], out["session"] = y, g
    processed_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(processed_dir / "features.csv", index=False)
    return X, np.array(y), np.array(g), used, skipped


def check_enough(y, g, min_windows: int) -> list[str]:
    problems = []
    classes = sorted(set(y))
    if len(classes) < 2:
        problems.append(f"need at least 2 classes, found {classes or 'none'}")
    for c in classes:
        n_w, n_s = int((y == c).sum()), len(set(g[y == c]))
        if n_s < 2:
            problems.append(f"class '{c}': {n_s} session(s), need >= 2 (for leave-session-out validation)")
        if n_w < min_windows:
            problems.append(f"class '{c}': {n_w} windows, need >= {min_windows}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=settings.raw_dir)
    ap.add_argument("--processed-dir", type=Path, default=settings.processed_dir)
    ap.add_argument("--out", type=Path, default=settings.model_dir)
    ap.add_argument("--min-windows", type=int, default=20, help="minimum windows per class")
    a = ap.parse_args(argv)

    X, y, g, used, skipped = build_dataset(a.data_dir, a.processed_dir)
    print(f"Data directory: {a.data_dir}")
    for u in used:
        print(f"  used    {u['file']:45s} label={u['label']:10s} windows={u['windows']}")
    for s in skipped:
        print(f"  skipped {s}")
    problems = check_enough(y, g, a.min_windows) if len(y) else ["no usable labelled recordings found"]
    if problems:
        print("\nINSUFFICIENT REAL DATA - no model trained (nothing is fabricated):")
        for p in problems:
            print("  -", p)
        print("\nRecord more sessions with the dashboard's recording mode (see README, 'Recording sensor data'):"
              "\n  >= 2 sessions per class, each >= ~30 s, for stationary / walking / running.")
        return 2

    n_sessions_min = min(len(set(g[y == c])) for c in set(y))
    cv = StratifiedGroupKFold(n_splits=max(2, min(5, n_sessions_min)), shuffle=True, random_state=0)
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    }
    results = {}
    labels = sorted(set(y))
    for name, model in candidates.items():
        pred = cross_val_predict(model, X, y, groups=g, cv=cv)
        results[name] = {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
                         "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}
        print(f"\n{name}: leave-session-out CV accuracy={results[name]['accuracy']:.3f} macro-F1={results[name]['macro_f1']:.3f}")
        print("  labels:", labels)
        for row in results[name]["confusion_matrix"]:
            print("  ", row)
    best = max(results, key=lambda k: results[k]["macro_f1"])
    final = candidates[best].fit(X, y)
    meta = {"best_model": best, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_windows": int(len(y)),
            "n_sessions": int(len(set(g))), "files": used, "validation": "leave-session-out stratified group k-fold",
            "cv_results": results, "window_s": settings.ml_window_s, "hop_s": settings.ml_hop_s}
    a.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_names": FEATURE_NAMES, "classes": labels, "meta": meta},
                a.out / "motion_classifier.joblib")
    (a.out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model '{best}' -> {a.out / 'motion_classifier.joblib'}")
    print("Reported scores are cross-validated estimates on YOUR recordings only; they do not generalise to other people/phones/placements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
