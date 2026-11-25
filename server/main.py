"""FastAPI application: WebSocket ingest + REST API.

Run from the project root:   python -m server.main
"""
from __future__ import annotations

import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .recorder import SessionRecorder
from .tracking_engine import TrackingEngine
from .websocket_server import ConnectionManager, phone_endpoint

log = logging.getLogger("dr.main")


class ResetBody(BaseModel):
    scope: str = "session"          # position | session | data


class RecBody(BaseModel):
    name: str = "recording"
    label: str | None = None


def get_local_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))            # no packet is actually sent
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips


def load_predictor():
    """Load the optional ML model; returns None if absent/unusable (never fabricated)."""
    try:
        from ml.predict import MotionPredictor
        return MotionPredictor.load(settings.model_path)
    except Exception as e:
        log.info("ML model not loaded: %s", e)
        return None


def create_app(cfg=settings, with_ml: bool = True) -> FastAPI:
    recorder = SessionRecorder(cfg.raw_dir, cfg.processed_dir)
    engine = TrackingEngine(cfg, predictor=load_predictor() if with_ml else None, recorder=recorder)
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        ips = get_local_ips()
        print("\n=== Smartphone Dead-Reckoning Server ===")
        for ip in ips or ["<laptop-ip>"]:
            print(f"  Phone WebSocket URL : ws://{ip}:{cfg.port}/ws")
        print(f"  REST / API docs     : http://localhost:{cfg.port}/docs")
        print(f"  ML model            : {'loaded' if engine.predictor else 'not found (optional - see README)'}")
        print("========================================\n")
        yield

    app = FastAPI(title="Smartphone Inertial Dead-Reckoning Server", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.engine, app.state.recorder, app.state.manager = engine, recorder, manager

    @app.get("/")
    def index():
        return {"service": "smartphone-dead-reckoning", "websocket": "/ws", "docs": "/docs",
                "local_ips": get_local_ips(), "port": cfg.port}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/status")
    def status():
        s = engine.state_dict()
        return {k: s[k] for k in ("ui_status", "connected", "client_ip", "tracking", "phase", "data_live",
                                  "packets_received", "packets_rejected", "sampling_rate_hz", "source",
                                  "simulated", "movement_state", "ml_model_loaded", "recording")}

    @app.get("/state")
    def state():
        """Everything the dashboard needs in one call."""
        return engine.state_dict()

    @app.get("/position")
    def position():
        st = engine.stats()
        return {"x_m": st["x_m"], "y_m": st["y_m"], "z_m": st["z_m"],
                "speed_mps": st["current_speed_mps"], "heading_deg": engine.sensors()["orientation_deg"]["heading"],
                "frame": "ENU world frame, origin = position at start/reset (estimated by IMU dead reckoning, NOT GPS)"}

    @app.get("/sensors")
    def sensors():
        return engine.sensors()

    @app.get("/stats")
    def stats():
        return {**engine.stats(), "movement_state": engine.state_dict()["movement_state"], "quality": engine.quality()}

    @app.get("/trajectory")
    def trajectory(max_points: int = Query(2000, ge=10, le=20000)):
        return engine.trajectory(max_points)

    @app.post("/reset")
    def reset(body: ResetBody | None = None, scope: str | None = None):
        sc = (scope or (body.scope if body else "session")).lower()
        if sc == "position":
            engine.reset_position()
        elif sc == "session":
            engine.reset_session()
        elif sc == "data":
            engine.clear_data()
        else:
            raise HTTPException(400, "scope must be position, session or data")
        return {"ok": True, "scope": sc}

    @app.post("/tracking/start")
    def tracking_start():
        engine.start()
        return {"ok": True, "phase": engine.phase}

    @app.post("/tracking/stop")
    def tracking_stop():
        engine.stop()
        return {"ok": True, "phase": engine.phase, "summary": engine.stats()}

    @app.post("/recording/start")
    def recording_start(body: RecBody):
        try:
            return engine.start_recording(body.name, body.label)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/recording/stop")
    def recording_stop():
        try:
            return engine.stop_recording()
        except RuntimeError as e:
            raise HTTPException(400, str(e))

    @app.get("/recording/sessions")
    def recording_sessions():
        return recorder.list_sessions()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await phone_endpoint(ws, engine, manager)

    return app


app = create_app()


def run() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("server.main:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
