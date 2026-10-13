"""Cancellation and the speed estimate.

Cancelling must never stop mid-line: a half-generated waveform is not usable
audio, and the whole point of the per-line design is that whatever finished is
kept.
"""
import numpy as np
import pytest

import common
from common import Cancelled, estimate_seconds, seconds_per_frame, synth, token_budget


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate_custom_voice(self, **kw):
        self.calls += 1
        n = len(kw["text"])
        return [np.zeros(240, dtype=np.float32) for _ in range(n)], 24000


LINES = [f"line {i}" for i in range(5)]


def test_cancel_stops_at_a_line_boundary_and_keeps_finished_audio():
    m = FakeModel()
    seen = []
    # Ask to stop once two lines are done.
    with pytest.raises(Cancelled):
        synth("preset", m, LINES, language="Auto", sampling={}, speaker="ryan",
              chunk_size=1, on_line=lambda i, w, sr: seen.append(i),
              should_cancel=lambda: len(seen) >= 2)
    assert seen == [0, 1], "the two finished lines must have been emitted"
    assert m.calls == 2, "no further model call may start after the cancel"


def test_no_cancel_callback_runs_everything():
    m = FakeModel()
    wavs, _ = synth("preset", m, LINES, language="Auto", sampling={}, speaker="ryan",
                    chunk_size=1)
    assert len(wavs) == len(LINES) and m.calls == len(LINES)


def test_a_callback_that_never_cancels_is_harmless():
    m = FakeModel()
    wavs, _ = synth("preset", m, LINES, language="Auto", sampling={}, speaker="ryan",
                    chunk_size=1, should_cancel=lambda: False)
    assert len(wavs) == len(LINES)


def test_cancel_before_the_first_line_yields_nothing():
    m = FakeModel()
    with pytest.raises(Cancelled):
        synth("preset", m, LINES, language="Auto", sampling={}, speaker="ryan",
              chunk_size=1, should_cancel=lambda: True)
    assert m.calls == 0


# --- speed calibration -------------------------------------------------------

def test_record_and_read_back_speed(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "speed.json")
    assert seconds_per_frame("float32") is None
    common.record_speed(60.0, 30.0, "float32")          # 2.0 s/frame
    assert seconds_per_frame("float32") == pytest.approx(2.0)
    common.record_speed(30.0, 30.0, "float32")          # 1.0 s/frame
    assert seconds_per_frame("float32") == pytest.approx(1.5)   # median of two


def test_outliers_do_not_capture_the_estimate(tmp_path, monkeypatch):
    """A running average is permanently skewed by a burst of bad samples -- a
    stubbed model, a swap-thrashing run. The median must shrug them off."""
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "speed.json")
    for _ in range(5):
        common.record_speed(2.0, 1.0, "float32")        # 2.0 s/frame, the truth
    for _ in range(3):
        common.record_speed(0.007, 1.0, "float32")      # absurdly fast outliers
    assert seconds_per_frame("float32") > 1.0, "outliers must not capture the median"
    for _ in range(2):
        common.record_speed(2.0, 1.0, "float32")
    assert seconds_per_frame("float32") == pytest.approx(2.0), "must re-converge"


def test_sample_window_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "speed.json")
    for _ in range(common.SPEED_WINDOW * 3):
        common.record_speed(1.0, 1.0, "float32")
    assert common.speed_sample_count("float32") == common.SPEED_WINDOW


def test_legacy_and_corrupt_files_recalibrate_instead_of_lying(tmp_path, monkeypatch):
    path = tmp_path / "speed.json"
    monkeypatch.setattr(common, "SPEED_PATH", path)
    path.write_text('{"float32": {"sec_per_frame": 1.9, "samples": 8}}', encoding="utf-8")
    assert seconds_per_frame("float32") is None      # old int-samples shape
    path.write_text("not json at all", encoding="utf-8")
    assert seconds_per_frame("float32") is None


def test_reset_speed(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "speed.json")
    common.record_speed(2.0, 1.0, "float32")
    assert seconds_per_frame("float32") is not None
    assert common.reset_speed() is True
    assert seconds_per_frame("float32") is None
    assert common.reset_speed() is False             # already gone, no raise


def test_record_speed_ignores_nonsense(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "speed.json")
    for secs, frames in [(0, 10), (10, 0), (-1, 10), (10, -1), (None, 10), (10, None)]:
        common.record_speed(secs, frames, "float32")
    assert seconds_per_frame("float32") is None


def test_record_speed_never_raises_on_an_unwritable_path(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "\0bad" / "speed.json")
    common.record_speed(10.0, 5.0, "float32")           # must not raise


def test_estimate_is_none_until_calibrated(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SPEED_PATH", tmp_path / "speed.json")
    monkeypatch.setattr(common, "seconds_per_frame", lambda *a, **k: None)
    assert estimate_seconds(["hello there"]) is None


def test_estimate_scales_with_text_and_excludes_runaway_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "seconds_per_frame", lambda *a, **k: 1.0)
    short = estimate_seconds(["x" * 40])
    long = estimate_seconds(["x" * 400])
    assert long > short
    # token_budget deliberately carries ~3x headroom as a runaway guard; the
    # user-facing estimate must not inherit it or every ETA would be 3x too long.
    assert short < token_budget("x" * 40)


# --- the machine report ------------------------------------------------------

def test_system_report_has_the_fields_the_ui_renders():
    r = common.system_report()
    for key in ("cpu", "cores", "threads", "torch_threads", "simd", "bf16_native",
                "free_gb", "total_gb", "low_memory", "dtype", "sec_per_frame"):
        assert key in r, f"system_report() is missing {key}"
    assert isinstance(r["bf16_native"], bool)
    assert r["dtype"] in ("float32", "bfloat16", "float16")
