import numpy as np
import pytest
import torch

import common
from common import (
    DELIVERY_PRESETS,
    delivery_sampling,
    loudness_normalize,
    peak_limit,
    resolve_dtype,
    sampling_kwargs,
    stitch,
    write_raw_parts,
    write_wavs,
)

SR = 24000


def _tone(seconds=1.0, freq=220.0, amp=0.2):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- delivery presets / sampling ------------------------------------------------

def test_delivery_sampling_known_and_unknown():
    assert delivery_sampling("steady") == DELIVERY_PRESETS["steady"]
    assert delivery_sampling("STEADY") == DELIVERY_PRESETS["steady"]
    assert delivery_sampling("natural") == {}
    assert delivery_sampling(None) == {}
    assert delivery_sampling("nonsense") == {}
    # returns a copy, not the module dict
    delivery_sampling("steady")["temperature"] = 999
    assert DELIVERY_PRESETS["steady"]["temperature"] != 999


def test_sampling_kwargs_explicit_overrides_preset():
    args = type("A", (), {})()
    args.delivery = "steady"
    args.temperature = 0.42
    args.top_k = None
    args.top_p = None
    args.repetition_penalty = None
    args.max_new_tokens = None
    out = sampling_kwargs(args)
    assert out["temperature"] == 0.42               # explicit wins
    assert out["repetition_penalty"] == DELIVERY_PRESETS["steady"]["repetition_penalty"]


# --- dtype / low memory -------------------------------------------------------

def test_resolve_dtype_low_memory():
    assert resolve_dtype("cpu", "0.6B", "auto", low_memory=False) is torch.float32
    assert resolve_dtype("cpu", "0.6B", "auto", low_memory=True) is torch.bfloat16
    assert resolve_dtype("cpu", "1.7B", "auto") is torch.bfloat16
    assert resolve_dtype("cpu", "0.6B", "float16") is torch.float16   # explicit request honoured


# --- loudness / peak --------------------------------------------------------------

def test_peak_limit_only_attenuates():
    loud = [_tone(amp=2.0)]
    out = peak_limit(loud, 0.97)
    assert np.max(np.abs(out[0])) == pytest.approx(0.97, abs=1e-3)
    quiet = [_tone(amp=0.1)]
    assert np.max(np.abs(peak_limit(quiet, 0.97)[0])) == pytest.approx(0.1, abs=1e-6)


def test_loudness_normalize_hits_target():
    pyln = pytest.importorskip("pyloudnorm")
    quiet = [_tone(seconds=2.0, amp=0.02)]
    out = loudness_normalize(quiet, SR, target_lufs=-16.0, peak=0.99)
    measured = pyln.Meter(SR).integrated_loudness(out[0])
    assert measured == pytest.approx(-16.0, abs=1.0)


def test_loudness_normalize_disabled_falls_back_to_peak():
    loud = [_tone(amp=3.0)]
    out = loudness_normalize(loud, SR, target_lufs=0, peak=0.97)
    assert np.max(np.abs(out[0])) == pytest.approx(0.97, abs=1e-3)


def test_loudness_normalize_survives_silence():
    silence = [np.zeros(SR, dtype=np.float32)]
    out = loudness_normalize(silence, SR, target_lufs=-16.0)
    assert np.max(np.abs(out[0])) == 0.0


# --- write_wavs / parts / stitch --------------------------------------------------

def test_write_wavs_join_uses_per_line_gaps(tmp_path):
    import soundfile as sf

    wavs = [_tone(seconds=0.5), _tone(seconds=0.5), _tone(seconds=0.5)]
    out = tmp_path / "joined.wav"
    write_wavs(wavs, SR, str(out), join=True, gaps=[0, 100, 900], lufs=0)
    data, sr = sf.read(out)
    expected = int(0.5 * SR) * 3 + int(0.1 * SR) + int(0.9 * SR)
    assert sr == SR
    assert abs(len(data) - expected) <= 2


def test_write_raw_parts_then_stitch_roundtrip(tmp_path):
    import soundfile as sf

    wavs = [_tone(seconds=0.4, freq=200), _tone(seconds=0.4, freq=400)]
    parts = write_raw_parts(wavs, SR, str(tmp_path / "parts"), "take")
    assert len(parts) == 2 and all(p.exists() for p in parts)
    # a raw part is untouched by levelling
    raw, _ = sf.read(parts[0])
    assert np.max(np.abs(raw)) == pytest.approx(0.2, abs=1e-3)

    joined = stitch([str(p) for p in parts], str(tmp_path / "out.wav"),
                    gaps=[0, 500], lufs=0)
    data, _ = sf.read(joined)
    expected = int(0.4 * SR) * 2 + int(0.5 * SR)
    assert abs(len(data) - expected) <= 2


# --- bfloat16 is only a saving where the CPU can actually do it ---------------
# Measured on this project's i5-7200U (AVX2, no AVX512/AMX): bf16 is 25x slower
# than float32 for a decode matvec and 265x slower for a prefill matmul, because
# torch emulates it. So the "save RAM" fallback must be gated on the hardware.

def test_low_memory_threshold_is_stricter_without_native_bf16(monkeypatch):
    import common

    monkeypatch.setattr(common, "cpu_bf16_is_native", lambda: True)
    native = common.low_memory_threshold_gb()
    monkeypatch.setattr(common, "cpu_bf16_is_native", lambda: False)
    emulated = common.low_memory_threshold_gb()
    assert emulated < native, "emulated bf16 must be a last resort, not the default"


def test_auto_low_memory_holds_out_for_float32_when_bf16_is_emulated(monkeypatch):
    import common

    # 3.6GB free: an emulated-bf16 CPU must stay on float32. It stays there even
    # when float32 does not fit -- measured 7.1x faster while paging -- so this
    # also holds well below the float32 weight size.
    monkeypatch.setattr(common, "cpu_bf16_is_native", lambda: False)
    for avail in (3.6, 2.5, 1.7):
        monkeypatch.setattr(common, "free_gb", lambda avail=avail: avail)
        assert common.auto_low_memory() is False, f"{avail}GB free should still use float32"
    # The same 3.6GB on a CPU with real bf16 kernels: switching is a genuine win.
    monkeypatch.setattr(common, "cpu_bf16_is_native", lambda: True)
    assert common.auto_low_memory() is True


def test_auto_low_memory_still_falls_back_when_float32_cannot_fit(monkeypatch):
    import common

    monkeypatch.setattr(common, "free_gb", lambda: 0.8)
    monkeypatch.setattr(common, "cpu_bf16_is_native", lambda: False)
    assert common.auto_low_memory() is True


def test_auto_low_memory_is_false_when_free_ram_is_unknown(monkeypatch):
    import common

    monkeypatch.setattr(common, "free_gb", lambda: None)
    assert common.auto_low_memory() is False


def test_cpu_bf16_probe_never_raises():
    import common

    assert isinstance(common.cpu_bf16_is_native(), bool)
