"""Load the trained movement classifier and predict.

  python -m ml.predict data/raw/session_001_walking.csv      # classify a recording
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES, derive_signals, feature_vector, make_windows
from server.config import settings


class MotionPredictor:
    def __init__(self, bundle: dict):
        if list(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("model was trained with a different feature set - retrain it")
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.meta = bundle.get("meta", {})

    @classmethod
    def load(cls, path: Path | str) -> "MotionPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path} (run: python -m ml.train_model)")
        return cls(joblib.load(path))

    def predict_window(self, win: np.ndarray) -> tuple[str, float]:
        x = feature_vector(win, settings.nominal_rate_hz).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        k = int(np.argmax(proba))
        return str(self.model.classes_[k]), float(proba[k])


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pred = MotionPredictor.load(settings.model_path)
    X = make_windows(derive_signals(pd.read_csv(sys.argv[1])))
    if len(X) == 0:
        print("recording too short for a single window")
        return
    labels = pred.model.predict(X)
    print(f"{len(labels)} windows ->", dict(Counter(labels)))


if __name__ == "__main__":
    main()
