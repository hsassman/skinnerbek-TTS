"""Sessions make a take repairable after a restart, and cleanup stops killed
runs from leaving per-line audio behind forever."""
import json
import time

import numpy as np
import soundfile as sf

from session import SESSION_SUFFIX, cleanup_orphans, list_sessions, load_session, save_session


def _wav(path, seconds=0.2, sr=24000):
    sf.write(path, np.zeros(int(sr * seconds), dtype=np.float32), sr)
    return str(path)


def _state(tmp_path, n=3):
    parts = [_wav(tmp_path / f"take_{i:03d}.wav") for i in range(n)]
    return {
        "kind": "preset",
        "params": {"size": "0.6B", "speaker": "ryan", "prefix": "take"},
        "lines": [f"line {i}" for i in range(n)],
        "gaps": [0] + [300] * (n - 1),
        "parts": parts,
        "sr": 24000,
        "prefix": "take",
    }


def test_roundtrip(tmp_path):
    state = _state(tmp_path)
    p = save_session(state, str(tmp_path / "joined.wav"), tmp_path)
    assert p is not None and p.exists()
    back = load_session(str(p))
    assert back is not None
    for key in ("kind", "lines", "gaps", "parts", "sr", "prefix"):
        assert back[key] == state[key]


def test_session_is_unusable_once_its_audio_is_gone(tmp_path):
    state = _state(tmp_path)
    p = save_session(state, str(tmp_path / "joined.wav"), tmp_path)
    from pathlib import Path

    Path(state["parts"][1]).unlink()
    assert load_session(str(p)) is None
    assert list_sessions(tmp_path) == []


def test_listing_is_newest_first(tmp_path):
    for i in range(3):
        s = _state(tmp_path)
        s["prefix"] = f"take{i}"
        save_session(s, str(tmp_path / "j.wav"), tmp_path)
        time.sleep(0.02)
    got = list_sessions(tmp_path)
    assert len(got) == 3
    assert [g["created"] for g in got] == sorted((g["created"] for g in got), reverse=True)


def test_corrupt_and_foreign_files_are_ignored(tmp_path):
    (tmp_path / f"bad{SESSION_SUFFIX}").write_text("{not json", encoding="utf-8")
    (tmp_path / f"old{SESSION_SUFFIX}").write_text(json.dumps({"version": 999}), encoding="utf-8")
    assert list_sessions(tmp_path) == []


def test_save_never_raises_on_a_bad_path(tmp_path):
    assert save_session({"prefix": "x"}, "j.wav", tmp_path / "nope" / "\0bad") is None


def test_cleanup_keeps_referenced_and_recent_parts(tmp_path):
    state = _state(tmp_path)
    save_session(state, str(tmp_path / "joined.wav"), tmp_path)
    recent = _wav(tmp_path / "recent_001.wav")          # unreferenced but new
    orphan = _wav(tmp_path / "orphan_001.wav")          # unreferenced and old
    import os

    old = time.time() - 48 * 3600
    os.utime(orphan, (old, old))

    removed = cleanup_orphans(tmp_path, max_age_hours=24.0)
    names = {p.name for p in removed}
    assert names == {"orphan_001.wav"}
    from pathlib import Path

    assert all(Path(p).exists() for p in state["parts"]), "referenced parts must survive"
    assert Path(recent).exists(), "parts younger than the age floor must survive"


def test_cleanup_dry_run_deletes_nothing(tmp_path):
    orphan = _wav(tmp_path / "orphan_001.wav")
    import os
    from pathlib import Path

    old = time.time() - 48 * 3600
    os.utime(orphan, (old, old))
    removed = cleanup_orphans(tmp_path, max_age_hours=24.0, dry_run=True)
    assert [p.name for p in removed] == ["orphan_001.wav"]
    assert Path(orphan).exists()


def test_cleanup_on_a_missing_directory_is_a_noop(tmp_path):
    assert cleanup_orphans(tmp_path / "does-not-exist") == []
