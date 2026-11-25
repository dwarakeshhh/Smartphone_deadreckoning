"""Recording mode: stores received sensor samples as CSV for ML training / evaluation.

Raw file   : data/raw/session_NNN_<name>.csv   (exact columns, see COLUMNS)
Metadata   : data/raw/session_NNN_<name>.json  (label, rate, duration, ...)
Trajectory : data/processed/session_NNN_<name>_trajectory.csv (estimated path)
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

COLUMNS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
TRAJ_COLUMNS = ["t_s", "x_m", "y_m", "z_m", "speed_mps", "state"]
VALID_LABELS = ("stationary", "walking", "running", "other")


def sanitize(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip()).strip("_").lower()
    return name[:40] or "recording"


class SessionRecorder:
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir, self.processed_dir = Path(raw_dir), Path(processed_dir)
        self.active = False
        self._fh = None
        self._writer = None
        self.rows = 0
        self.meta: dict = {}
        self._traj: list = []
        self._first_t = self._last_t = None

    def _next_number(self) -> int:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        nums = [int(m.group(1)) for p in self.raw_dir.glob("session_*.csv")
                if (m := re.match(r"session_(\d+)_", p.name))]
        return max(nums, default=0) + 1

    def start(self, name: str, label: str | None = None, source: str = "android") -> dict:
        if self.active:
            raise RuntimeError("a recording is already running")
        label = (label or "").lower() or None
        if label is not None and label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}")
        num = self._next_number()
        stem = f"session_{num:03d}_{sanitize(name)}"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.raw_dir / f"{stem}.csv"
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)
        self.stem, self.rows, self._traj = stem, 0, []
        self._first_t = self._last_t = None
        self.meta = {"session": num, "name": sanitize(name), "label": label, "file": self.path.name,
                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source}
        self.active = True
        return self.status()

    def write(self, sample) -> None:
        if not self.active:
            return
        self._writer.writerow([f"{sample.t_ms:.3f}", *map(float, sample.acc), *map(float, sample.gyro), *map(float, sample.mag)])
        self.rows += 1
        if self._first_t is None:
            self._first_t = sample.t_ms
        self._last_t = sample.t_ms
        if self.rows % 50 == 0:
            self._fh.flush()

    def add_trajectory_point(self, row) -> None:
        if self.active:
            self._traj.append(row)

    def stop(self) -> dict:
        if not self.active:
            raise RuntimeError("no recording is running")
        self._fh.close()
        self.active = False
        dur = ((self._last_t - self._first_t) / 1000.0) if self._first_t is not None else 0.0
        self.meta.update({"rows": self.rows, "duration_s": round(dur, 3),
                          "mean_rate_hz": round(self.rows / dur, 2) if dur > 0 else None,
                          "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        (self.raw_dir / f"{self.stem}.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        if self._traj:
            with open(self.processed_dir / f"{self.stem}_trajectory.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(TRAJ_COLUMNS)
                w.writerows(self._traj)
        return dict(self.meta)

    def status(self) -> dict:
        return {"active": self.active, "rows": self.rows, **({"meta": self.meta} if self.meta else {})}

    def list_sessions(self) -> list[dict]:
        out = []
        for p in sorted(self.raw_dir.glob("session_*.csv")):
            meta_p = p.with_suffix(".json")
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"file": p.name}
            out.append(meta)
        return out
