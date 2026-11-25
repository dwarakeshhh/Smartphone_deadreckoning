"""TrackingEngine: glues the whole processing pipeline together.

RAW SAMPLE -> validation -> timestamps/filters -> calibration -> orientation
-> world-frame acceleration -> gravity & bias removal -> stationary detection
-> dead reckoning (velocity, position) -> step detection -> movement state.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from .config import Settings, settings as default_settings
from .dead_reckoning import DeadReckoner
from .movement_classifier import STATIONARY, UNKNOWN, MovementClassifier, StationaryDetector
from .orientation import heading_deg, make_orientation_filter, rot_to_euler
from .recorder import SessionRecorder
from .sensor_processor import (Calibrator, OnlineSOS, SensorPreprocessor, ValidationError, design_filter,
                               validate_packet)
from .step_detector import StepDetector

SIGNAL_COLUMNS = ["acc_mag", "gyro_mag", "accf_mag", "roll", "pitch", "steps_total", "cadence"]

IDLE, CALIBRATING, TRACKING, STOPPED = "IDLE", "CALIBRATING", "TRACKING", "STOPPED"


class TrackingEngine:
    def __init__(self, cfg: Settings | None = None, offline: bool = False, predictor=None,
                 recorder: SessionRecorder | None = None):
        self.cfg = cfg or default_settings
        self.offline = offline                # offline: no calibration phase (used for feature extraction)
        self.predictor = predictor
        self.recorder = recorder
        self._lock = threading.RLock()
        # connection / packet counters (survive session resets)
        self.connected = False
        self.client_ip: str | None = None
        self.packets_received = 0
        self.packets_rejected = 0
        self.last_error: str | None = None
        self.last_packet_wall = 0.0
        self._arrivals: deque = deque(maxlen=200)
        self.source = None
        self.tracking = False
        self.phase = IDLE
        self.latest_raw: dict | None = None
        self._init_session()

    # ------------------------------------------------------------ setup
    def _init_session(self) -> None:
        c = self.cfg
        self.pre = SensorPreprocessor(c)
        self.cal = Calibrator(c)
        self.orient = make_orientation_filter(c)
        self.detector = StationaryDetector(c)
        self.dr = DeadReckoner(c)
        self.steps = StepDetector(c)
        self.classifier = MovementClassifier(c)
        self.hpf = OnlineSOS(design_filter("high", c.lin_acc_highpass_hz, c.nominal_rate_hz, 1))
        self.R = np.eye(3)
        self.gyro_bias = np.zeros(3)
        self.g_est = c.gravity
        self.bias_w = np.zeros(2)
        self._bias_q: deque = deque(maxlen=max(1, int(c.zupt_replay_s * c.nominal_rate_hz)))
        self.mag_ref: float | None = None
        self.mag_available = False
        self.t0: float | None = None
        self.t_last: float | None = None
        self.state = UNKNOWN
        self.ml_state: str | None = None
        self.ml_conf: float | None = None
        self.acc_trust = 1.0
        self.mag_trust = 0.0
        self.stationary = False
        self.stationary_samples = 0
        self.total_samples = 0
        self.last_signals: dict = {}
        self.sig_buf: deque = deque(maxlen=max(10, int(c.ml_window_s * c.nominal_rate_hz)))
        self._hop = 0
        self.recent: deque = deque(maxlen=c.recent_samples)
        self._clear_traj()
        self._next_traj_t = 0.0
        self.ended_wall: float | None = None
        self.started_wall: float | None = None
        self.calibration_info: dict = {}

    def _clear_traj(self) -> None:
        self.traj_t, self.traj_x, self.traj_y, self.traj_z, self.traj_v = [], [], [], [], []

    # ----------------------------------------------------- connection
    def on_connect(self, ip: str | None) -> None:
        with self._lock:
            self.connected, self.client_ip = True, ip

    def on_disconnect(self) -> None:
        with self._lock:
            self.connected = False

    # --------------------------------------------------------- control
    def start(self) -> None:
        with self._lock:
            if self.tracking:
                return
            self.reset_session()
            self.tracking = True
            self.started_wall = time.time()
            self.phase = TRACKING if self.offline else CALIBRATING

    def stop(self) -> None:
        with self._lock:
            if not self.tracking:
                return
            self.tracking = False
            self.phase = STOPPED
            self.ended_wall = time.time()

    def reset_session(self) -> None:
        """Forget everything about the current session (stats, trajectory, calibration)."""
        with self._lock:
            was = self.tracking
            self._init_session()
            self.tracking = was
            self.phase = (TRACKING if self.offline else CALIBRATING) if was else IDLE
            if was:
                self.started_wall = time.time()

    def reset_position(self) -> None:
        """Set position/velocity to (0,0,0) and restart the plotted trajectory. Other stats are kept."""
        with self._lock:
            self.dr.reset_position()
            self._clear_traj()
            self._next_traj_t = 0.0

    def clear_data(self) -> None:
        """Clear the in-memory sample buffer and the trajectory log (filters and stats are kept)."""
        with self._lock:
            self.recent.clear()
            self._clear_traj()
            self._next_traj_t = 0.0

    # ------------------------------------------------------- recording
    def start_recording(self, name: str, label: str | None) -> dict:
        if self.recorder is None:
            raise RuntimeError("recording is not enabled")
        with self._lock:
            return self.recorder.start(name, label, source=self.source or "android")

    def stop_recording(self) -> dict:
        if self.recorder is None:
            raise RuntimeError("recording is not enabled")
        with self._lock:
            return self.recorder.stop()

    # ---------------------------------------------------------- ingest
    def ingest(self, packet) -> tuple[bool, str | None]:
        """Validate and process one raw packet. Returns (accepted, error)."""
        with self._lock:
            try:
                s = validate_packet(packet, self.cfg)
            except ValidationError as e:
                self.packets_rejected += 1
                self.last_error = str(e)
                return False, str(e)
            now = time.time()
            self.packets_received += 1
            self.last_packet_wall = now
            self._arrivals.append(now)
            self.source = s.source
            self.latest_raw = {"t_ms": s.t_ms, "acc": s.acc.tolist(), "gyro": s.gyro.tolist(), "mag": s.mag.tolist()}
            if self.recorder is not None and self.recorder.active:
                self.recorder.write(s)
            if not self.tracking and self.cfg.auto_start:
                self.start()
            if self.tracking:
                self._process(s)
            return True, None

    # --------------------------------------------------------- pipeline
    def _process(self, s) -> None:
        c = self.cfg
        p = self.pre.process(s)
        if p is None:
            return
        if p.gap and self.t_last is not None:
            self.dr.vel[:] = 0.0            # data gap (e.g. Wi-Fi drop): velocity is unknowable
            self.hpf.z = None
        self.total_samples += 1
        acc_mag_raw = float(np.linalg.norm(p.acc_raw))
        acc_mag = float(np.linalg.norm(p.acc_f))
        use_mag = c.use_magnetometer and p.mag_valid

        if self.phase == CALIBRATING:
            res = self.cal.add(p)
            self._record_signal(acc_mag_raw, 0.0, acc_mag, 0.0, 0.0, 0.0)
            if res is not None:
                self.gyro_bias, self.g_est = res.gyro_bias.copy(), res.g_est
                self.mag_ref, self.mag_available = res.mag_ref_norm, res.mag_mean is not None
                self.orient.initialize(res.acc_mean, res.mag_mean if (c.use_magnetometer and res.mag_mean is not None) else None)
                self.hpf.z = None
                self.phase = TRACKING
                self.t0 = p.t_s
                self.calibration_info = {"gyro_bias": res.gyro_bias.round(5).tolist(), "g_est": round(res.g_est, 4),
                                         "mag_ref_uT": None if res.mag_ref_norm is None else round(res.mag_ref_norm, 2),
                                         "heading_reference": "magnetic north" if (c.use_magnetometer and res.mag_mean is not None) else "initial phone +Y (relative)"}
            self.t_last = p.t_s
            return

        # ---- orientation ------------------------------------------------
        if self.t0 is None:                          # offline mode: initialise on first sample
            self.orient.initialize(p.acc_f, p.mag_f if use_mag else None)
            self.t0 = p.t_s
            self.g_est = float(np.clip(acc_mag, 8.0, 11.5))
        gyro_c = p.gyro_f - self.gyro_bias
        gyro_mag = float(np.linalg.norm(gyro_c))
        # Accelerometer trust: low when |a| differs from g OR when the (previous-attitude)
        # gravity-free acceleration is large - |a| alone barely changes during horizontal
        # acceleration, which would otherwise be mistaken for a tilt of the phone.
        lin_prev = float(np.linalg.norm(self.R @ p.acc_f - np.array([0.0, 0.0, self.g_est])))
        trust = math.exp(-((acc_mag - self.g_est) / c.acc_trust_sigma) ** 2)
        sig_lin = c.acc_trust_lin_sigma * (3.0 if self.stationary else 1.0)
        trust *= math.exp(-(lin_prev / sig_lin) ** 2)
        self.acc_trust = max(c.acc_trust_floor, trust)
        if use_mag and self.mag_ref:
            rel = abs(float(np.linalg.norm(p.mag_f)) - self.mag_ref) / self.mag_ref
            self.mag_trust = 1.0 if rel < c.mag_trust_tolerance else 0.0
        else:
            self.mag_trust = 1.0 if use_mag else 0.0
        self.orient.update(gyro_c, p.acc_f, p.mag_f if (use_mag and self.mag_trust > 0) else None,
                           p.dt, self.acc_trust, self.mag_trust)
        self.R = self.orient.rotation()
        roll, pitch, _yaw = rot_to_euler(self.R)

        # ---- gravity removal + bias ------------------------------------
        a_w = self.R @ p.acc_f
        lin = a_w - np.array([0.0, 0.0, self.g_est])
        lin[:2] -= self.bias_w
        lin_mag = float(np.linalg.norm(lin))

        # ---- stationary detection / ZUPT --------------------------------
        stationary = self.detector.update(acc_mag, gyro_mag, lin_mag, self.g_est)
        self.stationary = stationary
        if stationary:
            self.stationary_samples += 1
            # learn the horizontal bias from samples that are older than the stationary-detection
            # lag, so the first moments of a push are never mistaken for bias
            ka = min(1.0, p.dt / c.bias_rate_s)
            if len(self._bias_q) == self._bias_q.maxlen:
                self.bias_w += ka * (self._bias_q[0] - self.bias_w)
            self._bias_q.append(a_w[:2].copy())
            self.gyro_bias += min(1.0, p.dt / c.gyro_bias_rate_s) * (p.gyro_f - self.gyro_bias)
            self.g_est += min(1.0, p.dt / c.gravity_rate_s) * (np.clip(acc_mag, 8.0, 11.5) - self.g_est)
        else:
            self._bias_q.clear()
        lin_hp = self.hpf.step(lin)

        # ---- dead reckoning ---------------------------------------------
        self.dr.update(lin_hp, p.dt, stationary)

        # ---- steps / state ----------------------------------------------
        self.steps.update(p.t_s, acc_mag, self.g_est)
        cadence = self.steps.cadence_hz(p.t_s)
        self.state = self.classifier.update(stationary, cadence, acc_mag - self.g_est)
        self._record_signal(acc_mag_raw, gyro_mag, acc_mag, roll, pitch, cadence)
        self.t_last = p.t_s

        # ---- logs ---------------------------------------------------------
        if p.t_s >= self._next_traj_t:
            self._next_traj_t = p.t_s + c.traj_log_interval_s
            rel_t = p.t_s - (self.t0 or p.t_s)
            if len(self.traj_t) < c.traj_max_points:
                self.traj_t.append(rel_t)
                self.traj_x.append(float(self.dr.pos[0]))
                self.traj_y.append(float(self.dr.pos[1]))
                self.traj_z.append(float(self.dr.pos[2]))
                self.traj_v.append(self.dr.speed)
            if self.recorder is not None and self.recorder.active:
                self.recorder.add_trajectory_point([round(rel_t, 3), round(float(self.dr.pos[0]), 4),
                                                    round(float(self.dr.pos[1]), 4), round(float(self.dr.pos[2]), 4),
                                                    round(self.dr.speed, 4), self.state])
        self.recent.append((p.t_s, *p.acc_raw.tolist(), *p.gyro_f.tolist()))

    def _record_signal(self, acc_mag, gyro_mag, accf_mag, roll, pitch, cadence) -> None:
        row = (acc_mag, gyro_mag, accf_mag, roll, pitch, float(self.steps.steps), cadence)
        self.last_signals = dict(zip(SIGNAL_COLUMNS, row))
        self.sig_buf.append(row)
        self._hop += 1
        hop = max(1, int(self.cfg.ml_hop_s * self.cfg.nominal_rate_hz))
        if self.predictor is not None and self._hop >= hop and len(self.sig_buf) == self.sig_buf.maxlen:
            self._hop = 0
            try:
                label, conf = self.predictor.predict_window(np.array(self.sig_buf))
                self.ml_state, self.ml_conf = label, conf
            except Exception as e:                       # never let ML break tracking
                self.last_error = f"ml: {e}"
                self.predictor = None

    # ------------------------------------------------------- snapshots
    def input_rate_hz(self) -> float:
        a = list(self._arrivals)
        if len(a) < 5:
            return 0.0
        recent = [t for t in a if a[-1] - t <= 2.0]
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 and len(recent) > 1 else 0.0

    def ui_status(self) -> str:
        live = (time.time() - self.last_packet_wall) < 1.5
        if not self.connected:
            return "PHONE DISCONNECTED" if self.packets_received else "WAITING FOR PHONE"
        if not self.tracking:
            return "TRACKING STOPPED" if self.phase == STOPPED else "PHONE CONNECTED"
        if self.phase == CALIBRATING:
            return "CALIBRATING - KEEP PHONE STILL"
        return "LIVE TRACKING" if live else "TRACKING PAUSED - NO DATA"

    def stats(self) -> dict:
        with self._lock:
            dur = (self.t_last - self.t0) if (self.t0 is not None and self.t_last is not None) else 0.0
            dist = self.dr.distance
            return {
                "total_distance_m": round(dist, 3),
                "step_distance_m": round(self.steps.steps * self.cfg.step_length_m, 3),
                "steps": int(self.steps.steps),
                "current_speed_mps": round(self.dr.speed, 3),
                "max_speed_mps": round(self.dr.max_speed, 3),
                "average_speed_mps": round(dist / dur, 3) if dur > 1e-6 else 0.0,
                "duration_s": round(max(dur, 0.0), 2),
                "x_m": round(float(self.dr.pos[0]), 3),
                "y_m": round(float(self.dr.pos[1]), 3),
                "z_m": round(float(self.dr.pos[2]), 3),
                "final_position_m": [round(float(self.dr.pos[0]), 3), round(float(self.dr.pos[1]), 3)],
                "step_length_m": self.cfg.step_length_m,
            }

    def sensors(self) -> dict:
        with self._lock:
            roll, pitch, yaw = rot_to_euler(self.R)
            raw = self.latest_raw or {}
            acc = np.array(raw.get("acc", [0, 0, 0]))
            gyro = np.array(raw.get("gyro", [0, 0, 0]))
            mag = np.array(raw.get("mag", [0, 0, 0]))
            return {
                "t_ms": raw.get("t_ms"),
                "acc": [round(float(v), 4) for v in acc], "gyro": [round(float(v), 4) for v in gyro],
                "mag": [round(float(v), 3) for v in mag],
                "acc_magnitude": round(float(np.linalg.norm(acc)), 4),
                "gyro_magnitude": round(float(np.linalg.norm(gyro)), 4),
                "orientation_deg": {"roll": round(math.degrees(roll), 2), "pitch": round(math.degrees(pitch), 2),
                                    "yaw": round(math.degrees(yaw), 2), "heading": round(heading_deg(self.R), 2)},
            }

    def quality(self) -> dict:
        with self._lock:
            st = self.pre.stats
            tsz = self.dr.time_since_zupt
            return {
                "fusion": self.orient.name, "stationary": bool(self.stationary),
                "stationary_fraction": round(self.stationary_samples / max(1, self.total_samples), 3),
                "acc_trust": round(self.acc_trust, 3), "mag_trust": round(self.mag_trust, 3),
                "mag_available": self.mag_available, "mag_in_use": bool(self.cfg.use_magnetometer and self.mag_trust > 0),
                "calibration_progress": round(self.cal.progress, 2), "calibration_waiting_for_still": self.cal.waiting_for_still,
                "calibration": self.calibration_info, "g_estimate": round(self.g_est, 4),
                "gyro_bias": self.gyro_bias.round(5).tolist(), "zupt_count": self.dr.zupt_count,
                "seconds_since_zupt": round(tsz, 1),
                "position_uncertainty_heuristic_m": round(0.5 * 0.05 * tsz ** 2, 2),
                "uncertainty_note": "rule-of-thumb 0.5*0.05 m/s^2*t^2 since last zero-velocity update, NOT a measured error",
                "samples_accepted": st.accepted, "out_of_order": st.out_of_order, "gaps": st.gaps,
                "missed_samples": st.missed_samples, "filter_rate_hz": round(st.rate_hz, 1),
                "packets_rejected": self.packets_rejected, "last_error": self.last_error,
                "warning": "IMU-only dead reckoning drifts; position error grows with time (quadratically between zero-velocity updates).",
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {
                "ui_status": self.ui_status(), "connected": self.connected, "client_ip": self.client_ip,
                "tracking": self.tracking, "phase": self.phase,
                "data_live": (time.time() - self.last_packet_wall) < 1.5,
                "packets_received": self.packets_received, "packets_rejected": self.packets_rejected,
                "sampling_rate_hz": round(self.input_rate_hz(), 1), "source": self.source,
                "simulated": bool(self.source and self.source != "android"),
                "movement_state": (self.ml_state if (self.cfg.use_ml_state and self.ml_state) else self.state),
                "rule_state": self.state, "ml_state": self.ml_state,
                "ml_confidence": None if self.ml_conf is None else round(self.ml_conf, 3),
                "ml_model_loaded": self.predictor is not None,
                "sensors": self.sensors(), "stats": self.stats(), "quality": self.quality(),
                "recording": self.recorder.status() if self.recorder else {"active": False},
                "server_time": time.time(),
            }

    def trajectory(self, max_points: int = 2000) -> dict:
        with self._lock:
            n = len(self.traj_x)
            stride = max(1, math.ceil(n / max_points))
            idx = list(range(0, n, stride))
            if n and idx[-1] != n - 1:
                idx.append(n - 1)
            return {"count": n, "t": [self.traj_t[i] for i in idx], "x": [self.traj_x[i] for i in idx],
                    "y": [self.traj_y[i] for i in idx], "z": [self.traj_z[i] for i in idx],
                    "speed": [self.traj_v[i] for i in idx],
                    "current": [float(self.dr.pos[0]), float(self.dr.pos[1])], "start": [0.0, 0.0]}
