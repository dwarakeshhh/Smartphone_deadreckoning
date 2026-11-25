"""Development tool: stream synthetic or recorded IMU data to the server.

NOT used in the real demo - the demo uses the Android app.  Packets carry
``"source": "simulator"`` (or "replay") so the dashboard shows a clear
"SIMULATED / REPLAYED DATA" banner and nobody mistakes them for phone data.

  python -m tools.phone_simulator --scenario square
  python -m tools.phone_simulator --replay data/raw/session_001_walking.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time

import websockets

from tools import synthetic_imu as sim


async def main_async(a) -> None:
    if a.replay:
        with open(a.replay, newline="") as fh:
            packets = [{k: float(v) for k, v in row.items()} | {"source": "replay"} for row in csv.DictReader(fh)]
    else:
        acc, yaw = sim.build(sim.SCENARIOS[a.scenario], a.rate)
        packets = sim.to_packets(acc, yaw, a.rate, source="simulator", seed=a.seed)
    uri = f"ws://{a.host}:{a.port}/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "control", "command": "start"}))
        print(f"streaming {len(packets)} packets to {uri} (speed x{a.speed}) ...")
        t0 = time.perf_counter()
        base = packets[0]["timestamp"]
        for p in packets:
            due = (p["timestamp"] - base) / 1000.0 / a.speed
            delay = due - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(json.dumps(p))
        await asyncio.sleep(0.5)
        if a.stop:
            await ws.send(json.dumps({"type": "control", "command": "stop"}))
            await asyncio.sleep(0.3)
    print("done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scenario", choices=sorted(sim.SCENARIOS), default="square")
    ap.add_argument("--replay", help="replay a recorded CSV instead")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop", action="store_true", help="send 'stop' at the end")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
