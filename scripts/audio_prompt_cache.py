"""Disk cache for encoded voice-clone prompts, keyed by the input that made them.

`model.create_voice_clone_prompt()` speech-tokenizes the reference clip and
extracts a speaker embedding. On CPU that's tens of seconds, and it is *identical*
every time for the same clip + transcript + mode + model size. The named voice
library (`voice_library.py`) already caches this for voices you register; this
module does the same for ad-hoc `tts.py clone --ref-audio ...` runs, so repeated
takes of the same clip only pay for generation.

Cache entries live under `.cache/voice_prompts/` (gitignored, safe to delete).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from common import REPO_ROOT
from voice_library import payload_from_items, rebuild_items

CACHE_DIR = REPO_ROOT / ".cache" / "voice_prompts"


def cache_key(ref_audio: str, ref_text: Optional[str], x_vector_only: bool,
              model_size: str) -> str:
    """Stable hash of everything that affects the encoding.

    Hash the file *contents* when `ref_audio` is a local path (so moving/renaming
    the clip still hits, and editing it misses); otherwise hash the string (URL or
    base64 blob).
    """
    h = hashlib.sha256()
    p = Path(ref_audio)
    try:
        if p.is_file():
            h.update(b"file\0")
            h.update(p.read_bytes())
        else:
            h.update(b"str\0")
            h.update(str(ref_audio).encode("utf-8"))
    except OSError:
        h.update(b"str\0")
        h.update(str(ref_audio).encode("utf-8"))
    h.update(b"\0ref_text\0")
    h.update((ref_text or "").strip().encode("utf-8"))
    h.update(b"\0xvec\0")
    h.update(b"1" if x_vector_only else b"0")
    h.update(b"\0size\0")
    h.update(str(model_size).encode("utf-8"))
    return h.hexdigest()[:32]


def _path(key: str, cache_dir: Optional[Path] = None) -> Path:
    return Path(cache_dir or CACHE_DIR) / f"{key}.pt"


def load_cached(key: str, cache_dir: Optional[Path] = None) -> Optional[List]:
    path = _path(key, cache_dir)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return rebuild_items(payload)
    except Exception:  # noqa: BLE001 -- a corrupt entry should just miss
        return None


def save_cached(key: str, items, cache_dir: Optional[Path] = None) -> Path:
    path = _path(key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload_from_items(items), path)
    return path


def encode_prompt(
    model,
    ref_audio: str,
    ref_text: Optional[str],
    x_vector_only: bool,
    model_size: str,
    *,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List, bool]:
    """Return (prompt_items, from_cache).

    `prompt_items` is what `run_clone(..., voice_clone_prompt=...)` expects.
    """
    key = cache_key(ref_audio, ref_text, x_vector_only, model_size)
    if use_cache:
        hit = load_cached(key, cache_dir)
        if hit is not None:
            if verbose:
                print(f"Reusing cached encoding of {ref_audio} ({key}).", file=sys.stderr)
            return hit, True

    items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=(ref_text.strip() if ref_text and ref_text.strip() else None),
        x_vector_only_mode=bool(x_vector_only),
    )
    if use_cache:
        save_cached(key, items, cache_dir)
        if verbose:
            print(f"Encoded {ref_audio} and cached it ({key}) for next time.", file=sys.stderr)
    return items, False
