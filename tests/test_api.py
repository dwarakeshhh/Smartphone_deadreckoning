import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from tools import synthetic_imu as sim


@pytest.fixture
def client(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    app = create_app(cfg, with_ml=False)
    with TestClient(app) as c:
        yield c


def stream(ws, segments, n=None):
    a, yaw = sim.build(segments, 50)
    pk = sim.to_packets(a, yaw, 50)
    for p in (pk if n is None else pk[:n]):
        ws.send_json(p)


def test_rest_endpoints_idle(client):
    assert client.get("/health").json() == {"ok": True}
    st = client.get("/status").json()
    assert st["connected"] is False and st["ui_status"] == "WAITING FOR PHONE"
    for path in ("/position", "/sensors", "/stats", "/trajectory", "/state"):
        assert client.get(path).status_code == 200


def test_websocket_end_to_end(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        assert client.get("/status").json()["ui_status"] == "PHONE CONNECTED"
        ws.send_json({"type": "control", "command": "start"})
        assert ws.receive_json()["type"] == "control_ack"
        stream(ws, [("still", 3), ("move", 90, 2.0, 3.0), ("still", 3)])
        ws.send_json({"type": "control", "command": "stop"})
        assert ws.receive_json()["command"] == "stop"
        st = client.get("/state").json()
    assert st["packets_received"] == 450 and st["packets_rejected"] == 0
    assert st["stats"]["y_m"] == pytest.approx(2.0, abs=0.5)
    assert st["simulated"] is True                       # simulator packets are flagged
    traj = client.get("/trajectory").json()
    assert traj["count"] > 20 and traj["current"][1] == pytest.approx(st["stats"]["y_m"], abs=1e-2)
    assert client.get("/status").json()["connected"] is False


def test_bad_packets_get_error_message_and_are_counted(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("not json")
        ws.send_json({"timestamp": 1})
        msg = ws.receive_json()
        assert msg["type"] == "error"
    st = client.get("/status").json()
    assert st["packets_rejected"] == 2


def test_batch_packets(client):
    a, yaw = sim.build([("still", 1)], 50)
    pk = sim.to_packets(a, yaw, 50)[:20]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "control", "command": "start"})
        ws.receive_json()
        ws.send_json({"samples": pk})
        ws.send_json({"type": "control", "command": "stop"})
        ws.receive_json()
    assert client.get("/status").json()["packets_received"] == 20


def test_reset_endpoints(client):
    assert client.post("/reset", params={"scope": "position"}).json()["ok"]
    assert client.post("/reset", params={"scope": "session"}).json()["ok"]
    assert client.post("/reset", params={"scope": "data"}).json()["ok"]
    assert client.post("/reset", params={"scope": "bogus"}).status_code == 400


def test_recording_writes_csv_with_spec_columns(client, tmp_path):
    r = client.post("/recording/start", json={"name": "Walking Forward", "label": "walking"})
    assert r.status_code == 200
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "control", "command": "start"})
        ws.receive_json()
        stream(ws, [("still", 4)])
        ws.send_json({"type": "control", "command": "stop"})
        ws.receive_json()
    meta = client.post("/recording/stop").json()
    csv_path = tmp_path / "raw" / meta["file"]
    header = csv_path.read_text().splitlines()[0]
    assert header == "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z"
    assert meta["rows"] == 200 and meta["label"] == "walking" and meta["file"].startswith("session_001_walking_forward")
    assert list((tmp_path / "processed").glob("*_trajectory.csv"))
    assert client.post("/recording/stop").status_code == 400
