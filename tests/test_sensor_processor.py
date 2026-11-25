import math

import numpy as np
import pytest

from server.config import Settings
from server.sensor_processor import SensorPreprocessor, ValidationError, validate_packet

GOOD = dict(timestamp=1000.0, acc_x=0.1, acc_y=0.2, acc_z=9.8, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            mag_x=0.0, mag_y=22.0, mag_z=-42.0)


def test_valid_packet():
    s = validate_packet(GOOD, Settings())
    assert s.t_ms == 1000.0 and s.acc[2] == 9.8 and s.source == "android"


@pytest.mark.parametrize("mut", [
    lambda d: d.pop("acc_x"), lambda d: d.update(acc_y=float("nan")), lambda d: d.update(acc_z="9.8"),
    lambda d: d.update(acc_z=900.0), lambda d: d.update(gyro_x=99.0), lambda d: d.update(mag_x=1e5),
    lambda d: d.update(timestamp=True)])
def test_invalid_packets_rejected(mut):
    d = dict(GOOD)
    mut(d)
    with pytest.raises(ValidationError):
        validate_packet(d, Settings())


def test_non_object_rejected():
    with pytest.raises(ValidationError):
        validate_packet([1, 2, 3], Settings())


def _feed(pre, freq_hz, seconds=4.0, fs=50.0):
    from server.sensor_processor import Sample
    out = []
    for i in range(int(seconds * fs)):
        t = i / fs
        a = np.array([math.sin(2 * math.pi * freq_hz * t), 0, 9.8])
        p = pre.process(Sample(1000 + t * 1000, a, np.zeros(3), np.zeros(3)))
        out.append(p.acc_f[0])
    return np.array(out)


def test_lowpass_attenuates_high_frequency_and_passes_low():
    cfg = Settings()
    hi = _feed(SensorPreprocessor(cfg), 20.0)[100:]
    lo = _feed(SensorPreprocessor(cfg), 1.0)[100:]
    assert np.abs(hi).max() < 0.25          # 20 Hz strongly attenuated (cutoff 10 Hz)
    assert np.abs(lo).max() > 0.9           # 1 Hz passes


def test_timestamp_handling_out_of_order_and_gap():
    from server.sensor_processor import Sample
    pre = SensorPreprocessor(Settings())
    mk = lambda t: Sample(t, np.array([0, 0, 9.8]), np.zeros(3), np.zeros(3))
    assert pre.process(mk(1000)) is not None
    assert pre.process(mk(1020)).dt == pytest.approx(0.02)
    assert pre.process(mk(1010)) is None                       # out of order
    p = pre.process(mk(3000))                                   # 2 s gap
    assert p.gap and pre.stats.gaps == 1 and pre.stats.out_of_order == 1
    pre.process(mk(3020))
    pre.process(mk(3100))                                       # 4 samples missing
    assert pre.stats.missed_samples >= 2
