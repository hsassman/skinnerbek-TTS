"""Regression tests for the security audit fixes (see SECURITY.md)."""
import asyncio
import json
import sys
import time

import numpy as np
import pytest
import soundfile as sf

import app
import refclip
from session import load_session, save_session
from textprep import _MD_LINK, load_lexicon, normalize_line


def _wav(path, seconds=0.2, sr=24000, channels=1):
    sf.write(path, np.zeros((int(sr * seconds), channels), dtype=np.float32), sr)
    return str(path)


# --- file names and paths ---------------------------------------------------

@pytest.mark.parametrize("raw", ["../../x", "C:/Windows/x", "a/b\\c", "..", "", "speak_ryan"])
def test_safe_prefix_is_a_plain_name(raw):
    out = app._safe_prefix(raw)
    assert out and "/" not in out and "\\" not in out and ":" not in out and ".." not in out


def test_recorder_value_must_be_an_upload(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    (uploads / "abc").mkdir(parents=True)
    clip = uploads / "abc" / "take.wav"
    _wav(clip)
    outside = _wav(tmp_path / "secret.wav")
    monkeypatch.setattr(app, "get_upload_folder", lambda: str(uploads))
    assert app._recorded_clip(str(clip)) == str(clip.resolve())
    assert app._recorded_clip(outside) != outside          # gr.update(), not the path
    assert app._recorded_clip(str(uploads / ".." / "secret.wav")) != outside


def _session(tmp_path, **params):
    parts = [_wav(tmp_path / f"take_{i:03d}.wav") for i in range(2)]
    state = {"kind": "preset", "params": {"size": "0.6B", "speaker": "ryan", "prefix": "take", **params},
             "lines": ["a", "b"], "gaps": [0, 300], "parts": parts, "sr": 24000, "prefix": "take"}
    return save_session(state, str(tmp_path / "joined.wav"), tmp_path)


def test_session_with_unsafe_params_prefix_is_refused(tmp_path):
    assert load_session(str(_session(tmp_path))) is not None
    assert load_session(str(_session(tmp_path, prefix="..\\..\\escape"))) is None
    assert load_session(str(_session(tmp_path, prefix="C:/escape"))) is None


def test_session_parts_must_sit_beside_the_session(tmp_path):
    p = _session(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    data["parts"][1] = _wav(elsewhere / "other.wav")
    p.write_text(json.dumps(data), encoding="utf-8")
    assert load_session(str(p)) is None


def test_legacy_voice_clip_outside_voices_is_not_served(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ref_clip_path", lambda name: tmp_path / "missing.wav")
    monkeypatch.setattr(app, "voice_meta", lambda name: {"ref_audio": _wav(tmp_path / "x.wav")})
    monkeypatch.setattr(app, "_voice_card_html", lambda v: "")
    path, _ = app.do_pick_voice("someone")
    assert path is None


def test_unknown_preset_speaker_is_refused():
    out = list(app.do_speak("preset:../../x", "Hello.", "0.6B", "Auto", "", "Natural", True, 300, 650, 0))
    assert len(out) == 1 and "Pick a voice" in str(out[0])


# --- resource limits ---------------------------------------------------------

def test_long_script_is_refused():
    with pytest.raises(ValueError, match="limit"):
        app._prep("a" * (app.MAX_SCRIPT_CHARS + 1), True, 300, 650)


def test_number_runs_stay_fast():
    for line in ("1," * 20000 + "1", "1" * 40000 + "%", "[" * 40000, "[a](" * 10000):
        t = time.perf_counter()
        normalize_line(line, lexicon=False)
        assert time.perf_counter() - t < 2.0


def test_markdown_links_still_strip():
    assert _MD_LINK.sub(r"\1", "see [the docs](https://x.y/z) now") == "see the docs now"


def test_oversized_clip_is_refused_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(refclip, "MAX_DECODE_SECONDS", 1.0)
    long_clip = _wav(tmp_path / "long.wav", seconds=2.0)
    with pytest.raises(ValueError, match="clip"):
        refclip.load_mono(long_clip)
    many = _wav(tmp_path / "many.wav", seconds=0.1, channels=refclip.MAX_CHANNELS + 1)
    with pytest.raises(ValueError):
        refclip.load_mono(many)


def test_lexicon_rule_with_bad_template_is_skipped(tmp_path):
    lex = tmp_path / "lexicon.txt"
    lex.write_text("/x/ -> \\9\nhello -> howzit\n", encoding="utf-8")
    rules = load_lexicon(lex)
    assert len(rules) == 1


# --- cancellation ------------------------------------------------------------

def test_stop_reaches_every_running_generation():
    a, b = app._new_run(), app._new_run()
    try:
        app.do_cancel()
        assert a.is_set() and b.is_set()
    finally:
        app._end_run(a)
        app._end_run(b)
    c = app._new_run()
    try:
        assert not c.is_set()          # a new run is not pre-cancelled
    finally:
        app._end_run(c)


# --- network exposure --------------------------------------------------------

def _call(guard, host):
    sent = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    mw = app.LocalGuard(inner, local_only=guard)
    asyncio.run(mw({"type": "http", "headers": [(b"host", host.encode())]}, receive, send))
    return sent[0]["status"], dict(sent[0]["headers"])


@pytest.mark.parametrize("host", ["127.0.0.1:7860", "localhost:7860", "[::1]:7860", "LOCALHOST"])
def test_local_hosts_are_served_with_frame_protection(host):
    status, headers = _call(True, host)
    assert status == 200
    assert headers[b"x-frame-options"] == b"SAMEORIGIN"
    assert b"frame-ancestors 'self'" in headers[b"content-security-policy"]


@pytest.mark.parametrize("host", ["evil.example:7860", "127.0.0.1.evil.example", "192.168.1.5:7860", ""])
def test_foreign_host_is_refused_when_local_only(host):
    assert _call(True, host)[0] == 403


def test_host_check_is_off_when_shared_with_a_login():
    assert _call(False, "studio.example")[0] == 200


@pytest.mark.parametrize("argv", [["--share"], ["--host", "0.0.0.0"], ["--host", "0.0.0.0", "--auth", "nopass"]])
def test_exposure_without_a_login_is_refused(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["app.py", *argv])
    with pytest.raises(SystemExit) as e:
        app.main()
    assert e.value.code == 2
