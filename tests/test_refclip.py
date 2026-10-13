import numpy as np
import soundfile as sf

import refclip

SR = 24000


def _speech(seconds, amp=0.3, sr=SR):
    """Speech-like signal: a tone with a syllable-rate envelope."""
    t = np.arange(int(seconds * sr)) / sr
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2
    return (amp * env * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_good_clip_has_no_issues():
    r = refclip.analyze_wave(_speech(10), SR)
    assert r.ok and r.grade == "good", r.issues


def test_silent_clip_is_unusable_and_says_why():
    r = refclip.analyze_wave(np.zeros(SR * 8, dtype=np.float32), SR)
    assert not r.ok
    assert any("silent" in m for _, m in r.issues)


def test_short_and_long_clips():
    assert not refclip.analyze_wave(_speech(2), SR).ok
    assert refclip.analyze_wave(_speech(4), SR).grade == "usable"
    assert refclip.analyze_wave(_speech(40), SR).grade == "usable"
    assert not refclip.analyze_wave(_speech(70), SR).ok


def test_clipping_is_flagged():
    wav = np.clip(_speech(8, amp=3.0), -1, 1)
    r = refclip.analyze_wave(wav, SR)
    assert any("clipped" in m for _, m in r.issues)


def test_mostly_silence_is_flagged():
    wav = np.concatenate([_speech(3), np.zeros(SR * 12, dtype=np.float32)])
    r = refclip.analyze_wave(wav, SR)
    assert any("silence" in m.lower() for _, m in r.issues)


def test_prepare_trims_levels_and_writes_mono(tmp_path):
    quiet = _speech(8, amp=0.05)
    pad = np.zeros(SR * 2, dtype=np.float32)
    stereo = np.stack([np.concatenate([pad, quiet, pad])] * 2, axis=1)
    src = tmp_path / "rec.wav"
    sf.write(src, stereo, 48000)

    dest, rep = refclip.prepare(src, tmp_path / "out" / "voice.ref.wav")
    data, sr = sf.read(dest)
    assert data.ndim == 1 and sr == 48000
    assert abs(np.max(np.abs(data)) - refclip.TARGET_PEAK) < 0.01
    # Samples were made at 24 kHz but written at 48 kHz: 4 s of speech, 1 s of
    # padding each side. Only ~0.3 s of that padding should survive the trim.
    assert rep.duration < 4.6
