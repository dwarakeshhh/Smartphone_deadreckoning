import pytest

from server.config import Settings
from server.tracking_engine import TrackingEngine
from tools import synthetic_imu as sim


@pytest.fixture
def cfg():
    return Settings()


def run_scenario(cfg, segments, **kw):
    """Feed a SYNTHETIC scenario (known motion) through a fresh engine - test helper only."""
    eng = TrackingEngine(cfg)
    eng.start()
    a, yaw = sim.build(segments, 50.0)
    for p in sim.to_packets(a, yaw, 50.0, **kw):
        eng.ingest(p)
    return eng
