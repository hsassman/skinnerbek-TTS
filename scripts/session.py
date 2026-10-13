"""Persist the last take so a read can be repaired after a restart.

The "Fix one line" panel needs the line list, the per-line gaps and the paths of
the raw per-line wavs. That lived only in a `gr.State`, so closing the browser
tab -- or restarting the server, which this project does often -- threw away the
ability to repair a read whose audio was still sitting on disk.

A session is a small JSON file written next to the parts it describes. It holds
no audio: just the text, the gaps and the filenames, so it stays tiny and is
safe to delete at any time.

`cleanup_orphans()` is the other half: a killed run leaves per-line wavs behind
with nothing referencing them, and those accumulate forever otherwise.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSION_SUFFIX = ".session.json"
SESSION_VERSION = 1


def sessions_dir(parts_dir: Path) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)
    return parts_dir


def save_session(state: Dict[str, Any], joined: str, parts_dir: Path) -> Optional[Path]:
    """Write `state` beside its parts. Returns the path, or None if it couldn't.

    Never raises: failing to write a convenience file must not fail a generation
    the user just waited minutes for.
    """
    try:
        d = sessions_dir(Path(parts_dir))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = d / f"{state.get('prefix', 'take')}_{stamp}{SESSION_SUFFIX}"
        payload = {
            "version": SESSION_VERSION,
            "created": time.time(),
            "created_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joined": str(joined),
            **{k: state[k] for k in
               ("kind", "params", "lines", "gaps", "parts", "sr", "prefix") if k in state},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


def list_sessions(parts_dir: Path, limit: int = 20) -> List[Dict[str, Any]]:
    """Newest-first sessions whose parts all still exist on disk."""
    out: List[Dict[str, Any]] = []
    try:
        files = sorted(Path(parts_dir).glob(f"*{SESSION_SUFFIX}"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return out
    for f in files:
        data = _read(f)
        if not data:
            continue
        parts = data.get("parts") or []
        if not parts or not all(Path(p).exists() for p in parts):
            continue          # a half-deleted take is not repairable; skip it
        data["_path"] = str(f)
        out.append(data)
        if len(out) >= limit:
            break
    return out


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("version") != SESSION_VERSION:
        return None
    if not data.get("lines") or not data.get("parts"):
        return None
    return data


def load_session(path: str) -> Optional[Dict[str, Any]]:
    """Rebuild a `gr.State` payload from a session file, or None if unusable."""
    data = _read(Path(path))
    if not data:
        return None
    if len(data["lines"]) != len(data["parts"]):
        return None           # written by an older/partial run
    if not all(Path(p).exists() for p in data["parts"]):
        return None
    return {
        "kind": data["kind"],
        "params": data["params"],
        "lines": list(data["lines"]),
        "gaps": list(data["gaps"]),
        "parts": list(data["parts"]),
        "sr": int(data["sr"]),
        "prefix": data.get("prefix", data["kind"]),
    }


def cleanup_orphans(parts_dir: Path, max_age_hours: float = 24.0,
                    dry_run: bool = False) -> List[Path]:
    """Delete per-line wavs that no session references and that are old enough.

    The age floor is what makes this safe to call while a generation is running:
    the parts of an in-flight take are seconds old, so they are never candidates.
    Returns the paths removed (or that would be, when `dry_run`).
    """
    d = Path(parts_dir)
    if not d.is_dir():
        return []
    referenced = set()
    for f in d.glob(f"*{SESSION_SUFFIX}"):
        data = _read(f)
        for p in (data or {}).get("parts", []) or []:
            referenced.add(str(Path(p).resolve()))

    cutoff = time.time() - max_age_hours * 3600
    removed: List[Path] = []
    for wav in d.glob("*.wav"):
        try:
            if str(wav.resolve()) in referenced or wav.stat().st_mtime > cutoff:
                continue
            if not dry_run:
                wav.unlink()
            removed.append(wav)
        except OSError:
            continue
    return removed
