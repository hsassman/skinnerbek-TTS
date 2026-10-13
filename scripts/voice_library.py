"""A named library of cloned voices, stored under `voices/`.

Encoding a reference clip (speech-tokenizing it + extracting a speaker embedding)
is the slow part of cloning on CPU, and it's identical every time you reuse the
same clip. This module does it once, saves the result as `voices/<name>.pt`, and
lets you refer to the voice by name forever after.

The on-disk format matches the one used by qwen_tts's own Gradio demo
(`{"items": [asdict(VoiceClonePromptItem), ...]}`), so voices saved here load in
the UI and vice versa.
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch

from common import VOICES_DIR


def voice_path(name: str, voices_dir: Path = VOICES_DIR) -> Path:
    return Path(voices_dir) / f"{name}.pt"


def ref_clip_path(name: str, voices_dir: Path = VOICES_DIR) -> Path:
    """The cleaned reference clip a voice was built from (see refclip.prepare).
    Kept beside the .pt so the voice can be auditioned and re-encoded later."""
    return Path(voices_dir) / f"{name}.ref.wav"


def delete_voice(name: str, voices_dir: Path = VOICES_DIR) -> bool:
    """Remove a voice and its reference clip. False if there was no such voice."""
    path = voice_path(name, voices_dir)
    if not path.exists():
        return False
    path.unlink()
    ref_clip_path(name, voices_dir).unlink(missing_ok=True)
    return True


def payload_from_items(items) -> dict:
    """Serialisable `{"items": [...]}` for `torch.save` (no meta). Shared with the
    ad-hoc reference-clip cache in audio_prompt_cache.py."""
    return {"items": [asdict(it) for it in items]}


def rebuild_items(payload: dict) -> List:
    """Turn a saved `{"items": [...]}` payload back into VoiceClonePromptItem objects."""
    from qwen_tts import VoiceClonePromptItem

    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError("not a valid voice payload (missing 'items')")

    items = []
    for d in payload["items"]:
        ref_code = d.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding")
        if ref_spk is None:
            raise ValueError("payload item is missing ref_spk_embedding")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        xvec_only = bool(d.get("x_vector_only_mode", False))
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=xvec_only,
                icl_mode=bool(d.get("icl_mode", not xvec_only)),
                ref_text=d.get("ref_text"),
            )
        )
    return items


def list_voices(voices_dir: Path = VOICES_DIR) -> List[str]:
    voices_dir = Path(voices_dir)
    if not voices_dir.exists():
        return []
    return sorted(p.stem for p in voices_dir.glob("*.pt"))


def save_voice(
    name: str,
    items,
    ref_audio: str,
    model_size: str = "0.6B",
    voices_dir: Path = VOICES_DIR,
) -> Path:
    """Persist the prompt items from `model.create_voice_clone_prompt` under `name`."""
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)

    payload = payload_from_items(items)
    # Not read back by the demo, but makes `voices show` useful.
    payload["meta"] = json.dumps(
        {
            "name": name,
            "ref_audio": str(ref_audio),
            "ref_text": items[0].ref_text,
            "x_vector_only_mode": bool(items[0].x_vector_only_mode),
            # The speaker embedding is extracted by the Base model, so a voice is tied to
            # the size that made it -- see the mismatch warning in load_voice.
            "model_size": model_size,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
    )
    path = voice_path(name, voices_dir)
    torch.save(payload, path)
    return path


def load_voice(
    name: str,
    voices_dir: Path = VOICES_DIR,
    expect_model_size: Optional[str] = None,
) -> List:
    """Rebuild VoiceClonePromptItem objects for `generate_voice_clone(voice_clone_prompt=...)`."""
    path = voice_path(name, voices_dir)
    if not path.exists():
        known = list_voices(voices_dir)
        hint = f" Known voices: {', '.join(known)}" if known else " No voices saved yet."
        raise SystemExit(f"No saved voice named '{name}' at {path}.{hint}")

    payload = torch.load(path, map_location="cpu", weights_only=True)

    if expect_model_size:
        saved_size = (voice_meta(name, voices_dir) or {}).get("model_size")
        if saved_size and saved_size != expect_model_size:
            print(
                f"WARNING: voice '{name}' was registered with the {saved_size} Base model but "
                f"you're loading {expect_model_size}. The speaker embedding comes from the model "
                f"that encoded it, so the clone may be off. Re-register with "
                f"`voices add {name} --force --model-size {expect_model_size}` to match.",
                file=sys.stderr,
            )

    try:
        return rebuild_items(payload)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")


def voice_meta(name: str, voices_dir: Path = VOICES_DIR) -> Optional[dict]:
    path = voice_path(name, voices_dir)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw = payload.get("meta") if isinstance(payload, dict) else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
