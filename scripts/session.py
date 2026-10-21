"""Persist repairable takes and remove old, unreferenced audio parts."""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSION_SUFFIX = ".session.json"
SESSION_VERSION = 1
# Settings lists this many takes to restore; cleanup keeps exactly those.
KEEP_SESSIONS = 25
# A prefix becomes part of output file names, so it must be a plain name.
_SAFE_PREFIX = re.compile(r"^\w[\w .-]{0,80}$")


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
        path = d / f"{state.get('prefix', 'take')}_{stamp}_{uuid.uuid4().hex[:8]}{SESSION_SUFFIX}"
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
    if limit <= 0:
        return out
    try:
        files = sorted(Path(parts_dir).glob(f"*{SESSION_SUFFIX}"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return out
    for f in files:
        data = _read(f)
        if not data or not _parts_exist(data):
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
    lines, parts, gaps = (data.get(k) for k in ("lines", "parts", "gaps"))
    if not all(isinstance(v, list) and v for v in (lines, parts, gaps)):
        return None
    if len(lines) != len(parts) or len(gaps) != len(lines):
        return None
    if not all(isinstance(v, str) and v.strip() for v in lines + parts):
        return None
    if any("\0" in p for p in parts):
        return None
    if not all(type(g) in (int, float) and math.isfinite(g) and g >= 0 for g in gaps):
        return None
    if data.get("kind") not in ("preset", "design", "voice"):
        return None
    if not isinstance(data.get("params"), dict):
        return None
    if type(data.get("sr")) is not int or data["sr"] <= 0:
        return None
    prefix = data.get("prefix", data["kind"])
    if not isinstance(prefix, str) or ".." in prefix or not _SAFE_PREFIX.match(prefix):
        return None
    return data


def _parts_exist(data: Dict[str, Any]) -> bool:
    return all(Path(p).is_file() for p in data["parts"])


def load_session(path: str) -> Optional[Dict[str, Any]]:
    """Rebuild a `gr.State` payload from a session file, or None if unusable."""
    data = _read(Path(path))
    if not data or not _parts_exist(data):
        return None
    return {
        "kind": data["kind"],
        "params": data["params"],
        "lines": list(data["lines"]),
        "gaps": list(data["gaps"]),
        "parts": list(data["parts"]),
        "sr": data["sr"],
        "prefix": data.get("prefix", data["kind"]),
    }


def cleanup_orphans(parts_dir: Path, max_age_hours: float = 24.0,
                    dry_run: bool = False) -> List[Path]:
    """Delete old audio that no kept take needs.

    The newest KEEP_SESSIONS session files (what Settings offers to restore) are
    kept, and so is every part they name -- even a session too odd to restore,
    so hand-repaired files aren't lost. Older session files are removed with the
    parts only they used; otherwise output/parts would grow forever, because
    every "Fix one line" writes a new session that still names the old part.

    Only files older than `max_age_hours` are candidates, which is what makes
    this safe while a generation is running. Returns the paths removed (or that
    would be, when `dry_run`).
    """
    d = Path(parts_dir)
    if not d.is_dir():
        return []
    cutoff = time.time() - max_age_hours * 3600
    try:
        sessions = sorted(d.glob(f"*{SESSION_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        sessions = []
    kept, stale = sessions[:KEEP_SESSIONS], sessions[KEEP_SESSIONS:]

    referenced = set()
    for f in kept:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parts = data.get("parts") if isinstance(data, dict) else None
        for p in parts if isinstance(parts, list) else []:
            if isinstance(p, str) and p and "\0" not in p:
                referenced.add(str(Path(p).resolve()))

    removed: List[Path] = []
    for f in stale:
        try:
            if f.stat().st_mtime > cutoff:
                continue
            if not dry_run:
                f.unlink()
            removed.append(f)
        except OSError:
            continue
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
