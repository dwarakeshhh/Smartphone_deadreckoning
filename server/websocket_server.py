"""WebSocket endpoint that receives sensor packets from the Android phone.

Accepted text frames (JSON):
  * a sensor packet            {"timestamp":..., "acc_x":..., ... "mag_z":...}
  * a batch of packets         {"samples":[ {...}, {...} ]}
  * a control message          {"type":"control","command":"start"|"stop"|"reset"}
The server sends {"type":"hello"} on connect and {"type":"status",...} once a second.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from .tracking_engine import TrackingEngine

log = logging.getLogger("dr.ws")


class ConnectionManager:
    def __init__(self):
        self.active: WebSocket | None = None

    async def replace(self, ws: WebSocket) -> None:
        """Only one phone is tracked at a time; a new connection replaces the old one."""
        old, self.active = self.active, ws
        if old is not None:
            try:
                await old.close(code=1000, reason="replaced by a new connection")
            except Exception:
                pass

    def release(self, ws: WebSocket) -> bool:
        if self.active is ws:
            self.active = None
            return True
        return False


async def _status_loop(ws: WebSocket, engine: TrackingEngine) -> None:
    try:
        while True:
            await asyncio.sleep(1.0)
            st = engine.state_dict()
            await ws.send_text(json.dumps({
                "type": "status", "tracking": st["tracking"], "phase": st["phase"],
                "packets": st["packets_received"], "state": st["movement_state"],
                "ui_status": st["ui_status"]}))
    except Exception:
        return


def apply_control(engine: TrackingEngine, command: str) -> bool:
    if command == "start":
        engine.start()
    elif command == "stop":
        engine.stop()
    elif command == "reset":
        engine.reset_session()
    else:
        return False
    return True


async def phone_endpoint(ws: WebSocket, engine: TrackingEngine, manager: ConnectionManager) -> None:
    await ws.accept()
    await manager.replace(ws)
    ip = ws.client.host if ws.client else None
    engine.on_connect(ip)
    log.info("phone connected from %s", ip)
    await ws.send_text(json.dumps({"type": "hello", "server": "dead-reckoning", "protocol": 1,
                                   "expected_rate_hz": engine.cfg.nominal_rate_hz}))
    status_task = asyncio.create_task(_status_loop(ws, engine))
    last_err_sent = 0.0
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                engine.packets_rejected += 1
                engine.last_error = "invalid JSON"
                continue
            if isinstance(msg, dict) and msg.get("type") == "control":
                ok = apply_control(engine, str(msg.get("command", "")))
                await ws.send_text(json.dumps({"type": "control_ack", "command": msg.get("command"), "ok": ok}))
                continue
            packets = msg["samples"] if isinstance(msg, dict) and isinstance(msg.get("samples"), list) else [msg]
            for pkt in packets:
                ok, err = engine.ingest(pkt)
                if not ok and time.time() - last_err_sent > 1.0:
                    last_err_sent = time.time()
                    await ws.send_text(json.dumps({"type": "error", "message": err}))
    except WebSocketDisconnect:
        pass
    except RuntimeError:        # receive after close
        pass
    finally:
        status_task.cancel()
        if manager.release(ws):
            engine.on_disconnect()
        log.info("phone disconnected (%s)", ip)
