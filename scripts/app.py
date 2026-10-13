"""Skinnerbek Studio: one web app for every Qwen3-TTS capability on this machine.

`launch_ui.py` runs the demo that ships with `qwen_tts`: one capability per
process, a bilingual UI, no notion of a script or a saved-voice library. This
app is the friendlier front door:

    * Speak       -- pick a voice, type a script, watch it render line by line,
                     redo any line you don't like.
    * Design      -- invent a voice from a written description (VoiceDesign).
    * Voices      -- record or upload a clip, check its quality, keep it as a voice.
    * Recordings  -- play back and download anything generated earlier.
    * Settings    -- machine report, pronunciation lexicon, housekeeping.

Every generation runs through the *same* helpers the command line uses
(`common.synth` over `run_preset` / `run_design` / `run_clone`, then
`write_wavs` / `stitch`), so a result from this app matches the equivalent
`tts.py` command at the same `--chunk-size`.

    python scripts/app.py                # http://127.0.0.1:7860
    python scripts/app.py --offline      # skip Hugging Face network checks
    python scripts/app.py --port 8000 --share
    python scripts/app.py --mock         # fast stand-in model, for UI work

Styles and browser code live in ui/ (studio.css, studio.js, recorder.js).

One model fits in RAM at a time on an 8 GB box, so switching voice type or model
size swaps the loaded model. The first generation of each kind loads the
checkpoint -- minutes on CPU. Later ones reuse it.
"""
import argparse
import gc
import queue
import sys
import threading
import time
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Import qwen_tts lazily (it prints banners and is slow); std / local first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_LUFS,
    OUTPUT_DIR,
    SAMPLES_DIR,
    VOICES_DIR,
    Cancelled,
    auto_low_memory,
    delivery_sampling,
    estimate_seconds,
    load_model,
    prepare_script,
    record_speed,
    reset_speed,
    resolve_dtype,
    speed_sample_count,
    stitch,
    synth,
    system_report,
    warn_if_instruct_ignored,
)
from session import cleanup_orphans, list_sessions, load_session, save_session  # noqa: E402
from textprep import LEXICON_PATH  # noqa: E402
import refclip  # noqa: E402
from voice_library import (  # noqa: E402
    delete_voice, list_voices, load_voice, ref_clip_path, save_voice, voice_meta, voice_path,
)

# ---------------------------------------------------------------------------
# Static facts about the built-in speakers. Names come straight from the
# CustomVoice checkpoint config (talker_config.spk_id); the two dialect notes
# come from talker_config.spk_is_dialect. No gender/language is asserted for the
# others -- the Qwen3-TTS model card is the reference for those.
# ---------------------------------------------------------------------------
PRESET_SPEAKERS: List[Tuple[str, str]] = [
    ("ryan", "Ryan"),
    ("aiden", "Aiden"),
    ("serena", "Serena"),
    ("vivian", "Vivian"),
    ("sohee", "Sohee"),
    ("ono_anna", "Ono Anna"),
    ("uncle_fu", "Uncle Fu"),
    ("dylan", "Dylan  (Chinese, Beijing accent)"),
    ("eric", "Eric  (Chinese, Sichuan accent)"),
]
PRESET_NOTE = "Ryan and Aiden are the English voices. The others are tuned for other languages."

LANGUAGES = [
    "Auto", "English", "Chinese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
]

# Runtime settings, filled in by main() from the command line.
CFG = SimpleNamespace(device="cpu", offline=False, low_memory=False, chunk_size=1,
                      threads=None, dtype="auto", mock=False)

# One loaded model at a time (8 GB box). Keyed by (kind, size, dtype_name).
_CACHE: dict = {}

# Cancellation. Generation is strictly serial here -- only one model fits in RAM
# -- so a single process-wide flag is the whole story, and it saves the Cancel
# button from needing per-session plumbing.
_CANCEL = threading.Event()


# ---------------------------------------------------------------------------
# Design system. The stylesheet and the browser-side behaviour live in ui/ as
# real .css/.js files so they can be read, linted and diffed like any other code.
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).resolve().parent / "ui"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
CSS = (UI_DIR / "studio.css").read_text(encoding="utf-8")
STUDIO_JS = (UI_DIR / "studio.js").read_text(encoding="utf-8")
RECORDER_JS = (UI_DIR / "recorder.js").read_text(encoding="utf-8")

APP_NAME = "Skinnerbek Studio"
APP_VERSION = "0.2"

# Lucide-style stroke icons, inlined so the app needs no network or icon font.
_ICON_PATHS = {
    "script": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
    "voice": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "render": '<path d="M3 12h1M7 8v8M11 5v14M15 9v6M19 7v10"/>',
    "spark": '<path d="M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.9 2.1 2.1.9-2.1.9L19 21l-.9-2.1-2.1-.9 2.1-.9z"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
    "play": '<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5z"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>',
    "broom": '<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>',
    "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    "lock": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "wifi_off": '<path d="M2 2l20 20M8.5 16.5a5 5 0 0 1 7 0M5 13a10 10 0 0 1 5.2-2.8M19 13a10 10 0 0 0-2.4-1.7M12 20h.01"/>',
}


def _icon(name: str) -> str:
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>')


# The mark: a speech bubble whose top edge is Table Mountain, sound bars inside.
# Drawn as SVG (not the generated bitmap) so it is crisp at every size and its
# bars can animate. assets/favicon.svg is the static copy of the same shape.
LOGO_SVG = (
    '<svg viewBox="0 0 64 64" aria-hidden="true" focusable="false">'
    '<defs><linearGradient id="tts-logo-g" x1="0" y1="1" x2="1" y2="0">'
    '<stop offset="0" stop-color="#ff5a5f"/><stop offset=".55" stop-color="#ff8f50"/>'
    '<stop offset="1" stop-color="#ffb547"/></linearGradient></defs>'
    '<rect width="64" height="64" rx="16" fill="#0f1424"/>'
    '<path d="M13 27q0-6.5 6-7.5l3-4.5 4 1.2 3-1.6 6 .9 4-1 4 .8 3 4.2q6 1 6 7.5v9.5q0 7-7 7H28l-11 8 2.6-8.2'
    'Q13 43 13 36.5z" fill="none" stroke="url(#tts-logo-g)" stroke-width="4.4" stroke-linejoin="round"/>'
    '<g fill="#fff4e3">'
    '<rect class="bar" x="21.2" y="27" width="3.6" height="8" rx="1.8"/>'
    '<rect class="bar" x="26.6" y="23.5" width="3.6" height="15" rx="1.8"/>'
    '<rect class="bar" x="32" y="21" width="3.6" height="20" rx="1.8"/>'
    '<rect class="bar" x="37.4" y="23.5" width="3.6" height="15" rx="1.8"/>'
    '<rect class="bar" x="42.8" y="27" width="3.6" height="8" rx="1.8"/>'
    '</g></svg>'
)


def _mountain_wave(offset: float = 0.0, amp: float = 1.0) -> str:
    """Table Mountain's skyline (Devil's Peak, the flat table, Lion's Head) as an
    audio waveform: the silhouette plus a deterministic ripple along it."""
    import math

    profile = [(0, 168), (110, 158), (150, 132), (195, 70), (222, 96), (250, 82), (470, 80),
               (500, 118), (528, 122), (552, 98), (572, 120), (620, 166)]

    def base(x: float) -> float:
        for (x0, y0), (x1, y1) in zip(profile, profile[1:]):
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                t = t * t * (3 - 2 * t)            # smoothstep between control points
                return y0 + (y1 - y0) * t
        return profile[-1][1]

    pts = []
    for i in range(0, 621, 3):
        x = float(i)
        env = 2.2 + 5.5 * math.exp(-((x - 330) / 70) ** 2) + 3.5 * math.exp(-((x - 430) / 30) ** 2)
        ripple = amp * env * math.sin(x * 0.42) * (0.6 + 0.4 * math.sin(x * 0.07))
        pts.append(f"{x:.0f},{base(x) + offset + ripple:.1f}")
    return "M" + " L".join(pts)


def _asset_uri(stem: str) -> Optional[str]:
    """A bitmap from assets/ as a data: URI (no static-file route needed), or
    None so the caller falls back to drawn SVG art."""
    import base64

    for ext, mime in (("webp", "image/webp"), ("png", "image/png"), ("jpg", "image/jpeg")):
        p = ASSETS_DIR / f"{stem}.{ext}"
        if p.exists():
            return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return None


def _hero_css() -> str:
    """The generated banner (assets/banner.*) becomes the masthead's own
    background, so it fills the header edge to edge and is clipped by its
    rounded corners instead of floating as a box inside it."""
    uri = _asset_uri("banner")
    return f'\n.gradio-container .tts-head {{ --tts-hero-img: url("{uri}"); }}\n' if uri else ""


def _hero_art() -> str:
    """Drawn skyline-waveform, used only when there is no banner bitmap."""
    if _asset_uri("banner"):
        return ""
    stars = "".join(f'<circle class="star" cx="{x}" cy="{y}" r="{r}"/>' for x, y, r in
                    ((300, 22, 1.2), (372, 40, 1.6), (455, 18, 1.1), (520, 46, 1.4), (590, 26, 1.2),
                     (250, 44, 1), (410, 30, .9)))
    rings = "".join(f'<circle class="ring" cx="360" cy="80" r="{r}"/>' for r in (48, 88, 132))
    return (
        '<svg class="tts-hero__art" viewBox="0 0 620 200" preserveAspectRatio="xMaxYMid slice" aria-hidden="true">'
        '<defs><linearGradient id="tts-sunset" x1="0" x2="1"><stop offset="0" stop-color="#ff5a5f" stop-opacity="0"/>'
        '<stop offset=".25" stop-color="#ff6b5a"/><stop offset=".7" stop-color="#ff9a4d"/>'
        '<stop offset="1" stop-color="#ffb547"/></linearGradient></defs>'
        f'{rings}{stars}'
        f'<path class="wave wave--2" style="--len:1500" d="{_mountain_wave(16, .6)}"/>'
        f'<path class="wave" style="--len:1500" d="{_mountain_wave()}"/>'
        '</svg>'
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _low_memory() -> bool:
    return bool(CFG.low_memory) or auto_low_memory()


def get_model(kind: str, size: str):
    """Return a loaded Qwen3TTSModel, reusing the cached one when possible.

    Swapping kind or size evicts the previous model first -- loading a second
    one alongside it would blow the RAM budget on this machine.
    """
    dtype = resolve_dtype(CFG.device, size, CFG.dtype, low_memory=_low_memory())
    key = (kind, size, str(dtype))
    if key in _CACHE:
        return _CACHE[key]
    if CFG.mock:
        from mock_model import MockTTSModel

        _CACHE.clear()
        _CACHE[key] = MockTTSModel(kind, size)
        return _CACHE[key]

    if _CACHE:
        _CACHE.clear()
        gc.collect()

    args = SimpleNamespace(
        device=CFG.device,
        model_size=size,
        dtype=CFG.dtype,
        offline=CFG.offline,
        low_memory=CFG.low_memory,
        threads=CFG.threads,
    )
    # load_model runs the RAM preflight (common.check_memory), which raises
    # SystemExit when a load physically can't fit -- callers catch that too.
    model = load_model(kind, args)
    _CACHE[key] = model
    return model


# ---------------------------------------------------------------------------
# The machine report
#
# This exists because the worst performance trap in this project was invisible:
# a CPU with no hardware bfloat16 silently taking a ~25x slower path. Putting
# the dtype, the bf16 capability and the measured speed on screen turns an
# hour-long mystery into something you can read before you commit to a run.
# ---------------------------------------------------------------------------
def _stat(label: str, value: str, tone: str = "") -> str:
    """One machine-report card: a small label over its value. Tone is also
    spelt out in the value text (e.g. "slow here"), never colour alone."""
    cls = f"tts-hstat tts-hstat--{tone}" if tone else "tts-hstat"
    return f'<div class="{cls}"><span>{escape(label)}</span><b>{escape(value)}</b></div>'


def _speed_phrase(spf: Optional[float]) -> str:
    """Seconds-per-frame as something a person can plan around."""
    if not spf:
        return "not measured yet"
    per_sec_audio = spf * 12.0          # 12 codec frames per second of speech
    if per_sec_audio < 1:
        return f"~{1 / per_sec_audio:.1f}x faster than real time"
    return f"~{per_sec_audio:.0f}s of compute per second of audio"


def _system_detail_html() -> str:
    """The same facts as the header chips, laid out for reading rather than
    glancing -- Settings is where you go to understand a number, not spot it."""
    r = system_report()
    spf = r.get("sec_per_frame")
    free, total = r.get("free_gb"), r.get("total_gb")
    rows = [
        ("Processor", str(r.get("cpu", "unknown"))),
        ("Cores / threads", f"{r.get('cores') or '?'} physical, {r.get('threads') or '?'} logical"),
        ("torch threads", str(r.get("torch_threads", "?"))),
        ("Vector instructions", str(r.get("simd", "unknown"))),
        ("Hardware bfloat16", "yes" if r.get("bf16_native") else
         "no - bfloat16 is emulated and far slower than float32 here"),
        ("Weight precision in use", str(r.get("dtype", "?"))),
        ("Memory", f"{free:.2f} GB free of {total:.2f} GB" if free and total else "unknown"),
        ("Measured speed", _speed_phrase(spf) +
         (f" (median of {speed_sample_count()} lines)" if spf else "")),
    ]
    body = "".join(
        f'<tr><th scope="row">{escape(k)}</th><td>{escape(v)}</td></tr>' for k, v in rows
    )
    return f'<table class="tts-kv">{body}</table>'


def _system_html() -> str:
    r = system_report()
    free, total = r.get("free_gb"), r.get("total_gb")
    dtype = r.get("dtype", "?")

    # Amber, not red: low RAM and the bf16 path make runs slow, not broken.
    ram_tone = "warn" if (free or 0) < 3.2 else "ok"
    chips = [
        _stat("CPU", f"{r.get('cores') or '?'} cores / {r.get('threads') or '?'} threads"),
        _stat("Vector unit", str(r.get("simd", "?"))),
        _stat("Torch threads", str(r.get("torch_threads", "?"))),
        _stat("RAM free",
              f"{free:.1f} of {total:.1f} GB" if free and total else "unknown",
              ram_tone),
        _stat("Precision", dtype if dtype == "float32" else f"{dtype} (slow here)",
              "ok" if dtype == "float32" else "warn"),
        _stat("Speed", _speed_phrase(r.get("sec_per_frame"))),
    ]
    html = f'<div class="tts-hstats" role="group" aria-label="Machine status">{"".join(chips)}</div>'

    # The one warning that actually costs hours if missed.
    if dtype != "float32" and not r.get("bf16_native"):
        need = max(0.0, 3.2 - (free or 0))
        html += (
            '<details class="tts-note tts-note--warn">'
            f"<summary><strong>Running on the slow path.</strong> Free {need:.1f} GB of RAM "
            "to speed up the next take.</summary>"
            "<p>This CPU has no hardware bfloat16, so torch emulates it &mdash; about "
            "<strong>25&times; slower per step</strong> than float32. Closing a browser usually "
            "frees enough, and float32 is picked automatically on the next generation.</p></details>"
        )
    elif ram_tone == "warn":
        html += (
            '<div class="tts-note tts-note--warn">'
            "<strong>RAM is tight.</strong> float32 wants about 3&nbsp;GB, so this run "
            "will page to disk and be slower than it needs to be. Closing a few browser "
            "tabs before generating will speed it up.</div>"
        )
    return html


# ---------------------------------------------------------------------------
# Text prep / output helpers
# ---------------------------------------------------------------------------
def _lang(display: str) -> str:
    """The bundled demo passes languages to the model lowercased, with 'Auto'
    left alone. Match that so a picked language behaves the same here."""
    return "Auto" if not display or display == "Auto" else display.lower()


def _seeded(seed):
    """Seed torch for a repeatable take. 0 / blank means a fresh take: Gradio 6
    turns an empty number box into 0, so 0 can't double as a real seed --
    otherwise every take, and every "Fix one line" retry, would be identical."""
    if seed is None or seed == "" or int(seed) == 0:
        import torch

        torch.seed()        # reseed from OS entropy so a fresh take really is fresh
        return
    import torch

    torch.manual_seed(int(seed))


def _prep(script, tidy, gap_ms, para_gap):
    """(lines, gaps) from the script box. `tidy` toggles the rewrite; the
    lexicon.txt pronunciation overrides are always applied."""
    return prepare_script(
        script,
        normalize=bool(tidy),
        lexicon=True,
        max_words=24,
        sentence_gap_ms=int(gap_ms),
        paragraph_gap_ms=int(para_gap),
    )


def _parts_dir() -> Path:
    d = OUTPUT_DIR / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


_CAPABILITY = {"preset": "preset", "voice": "clone", "design": "design"}


def _fmt_dur(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Live render progress
#
# `synth` already reports each finished line through `on_line`, so the panel is
# a pure view over data the generation loop was emitting anyway -- it adds no
# work to the model path.
# ---------------------------------------------------------------------------
def _progress_html(lines: List[str], done: List[Dict[str, Any]], started: float,
                   state: str = "running", note: str = "") -> str:
    """state: running | done | cancelled | error."""
    total = len(lines)
    n = len(done)
    pct = 100.0 * n / total if total else 0.0
    elapsed = time.time() - started

    # ETA from this run's own completed lines, which beats any static estimate.
    eta = None
    if state == "running" and n and n < total:
        per_char = sum(d["seconds"] for d in done) / max(1, sum(len(lines[d["i"]]) for d in done))
        eta = per_char * sum(len(lines[i]) for i in range(n, total))

    title = {
        "running": f"Rendering line {min(n + 1, total)} of {total}",
        "done": f"Finished {total} line{'s' if total != 1 else ''}",
        "cancelled": f"Stopped after {n} of {total} line{'s' if total != 1 else ''}",
        "error": "Generation failed",
    }[state]
    fill_cls = {"running": "", "done": " tts-bar__fill--done",
                "cancelled": " tts-bar__fill--err", "error": " tts-bar__fill--err"}[state]

    meta = [f"{_fmt_dur(elapsed)} elapsed"]
    if eta is not None:
        meta.append(f"about {_fmt_dur(eta)} left")
    audio = sum(d["audio"] for d in done)
    if audio:
        meta.append(f"{audio:.1f}s audio")

    rows = []
    by_index = {d["i"]: d for d in done}
    for i, text in enumerate(lines):
        d = by_index.get(i)
        if d:
            ico, cls, right = "&#10003;", " tts-line--done", f'{d["seconds"]:.0f}s &middot; {d["audio"]:.1f}s'
            aria = "done"
        elif i == n and state == "running":
            ico, cls, right = '<span class="tts-spin">&#9696;</span>', " tts-line--now", "rendering…"
            aria = "in progress"
        else:
            ico, cls, right = "&#9675;", " tts-line--todo", ""
            aria = "waiting"
        snippet = text if len(text) <= 96 else text[:95] + "…"
        rows.append(
            f'<li class="tts-line{cls}">'
            f'<span class="tts-line__ico" aria-hidden="true">{ico}</span>'
            f'<span class="tts-line__txt">{escape(snippet)}'
            f'<span class="tts-sr"> ({aria})</span></span>'
            f'<span class="tts-line__t">{right}</span></li>'
        )

    note_html = f'<div class="tts-note tts-note--warn">{note}</div>' if note else ""
    return (
        f'<div class="tts-prog" role="status" aria-live="polite">'
        f'<div class="tts-prog__top">'
        f'<span class="tts-prog__title">{_EQ if state == "running" else ""}{escape(title)}</span>'
        f'<span class="tts-prog__meta">{" &middot; ".join(meta)}</span>'
        f"</div>"
        f'<div class="tts-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{pct:.0f}" aria-label="Rendering progress">'
        f'<div class="tts-bar__fill{fill_cls}" style="width:{pct:.1f}%"></div></div>'
        f'<ol class="tts-lines">{"".join(rows)}</ol>'
        f"{note_html}</div>"
    )


_EQ = '<span class="tts-eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>'

# A small waveform-in-a-bubble for empty states, in the brand gradient.
_EMPTY_ART = (
    '<svg class="tts-empty__art" viewBox="0 0 120 72" aria-hidden="true">'
    '<defs><linearGradient id="tts-sunset-e" x1="0" x2="1"><stop offset="0" stop-color="#ff6b5a"/>'
    '<stop offset="1" stop-color="#ffb547"/></linearGradient></defs>'
    '<rect class="b" x="14" y="6" width="92" height="52" rx="16"/><path class="b" d="M30 56l-8 12 18-10z"/>'
    '<path class="s" d="M34 32v0M44 24v16M54 18v28M64 26v12M74 20v24M84 30v4"/></svg>'
)


def _idle_html(title: str = "Nothing rendering", msg: str = "", art: bool = True) -> str:
    body = f"<b>{escape(title)}</b>{escape(msg)}" if msg else f"<b>{escape(title)}</b>"
    return f'<div class="tts-empty">{_EMPTY_ART if art else ""}{body}</div>'


def _sec(title: str, level: int = 2, n: Optional[int] = None) -> str:
    """One section-heading rhythm for every tab: a small uppercase label with a
    rule running to the edge, optionally numbered as a step."""
    num = (f'<span class="tts-sec__n" aria-hidden="true">{n}</span><span class="tts-sr">Step {n}: </span>'
           if n else "")
    return f'<div class="tts-sec">{num}<h{level}>{escape(title)}</h{level}></div>'


def _title(icon: str, title: str, sub: str = "") -> str:
    """A card's heading: icon tile, title, one-line explanation."""
    p = f"<p>{sub}</p>" if sub else ""
    return (f'<div class="tts-title"><span class="tts-title__ico">{_icon(icon)}</span>'
            f"<div><h2>{escape(title)}</h2>{p}</div></div>")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _run_synth(kind: str, p: dict, lines: List[str], events: "queue.Queue"):
    """Generate `lines`, pushing a dict onto `events` as each one lands.

    Runs on a worker thread so the request handler can keep yielding progress
    to the browser. Writes each finished line to output/parts/ immediately, so
    a cancelled or crashed run still leaves usable audio behind.
    """
    capability = _CAPABILITY[kind]
    prefix = p.get("prefix", capability)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pdir = _parts_dir()
    part_paths: List[str] = []
    clock = {"t": time.time()}

    def on_line(i, wav, sr):
        now = time.time()
        seconds = now - clock["t"]
        clock["t"] = now
        path = pdir / f"{prefix}_{ts}_{i + 1:03d}.wav"
        sf.write(path, np.asarray(wav, dtype=np.float32), sr)
        part_paths.append(str(path))
        audio = len(wav) / float(sr or 24000)
        # Feed the measured rate back so future runs can be estimated up front
        # (never from the mock model, which would make every estimate absurd).
        if not CFG.mock:
            record_speed(seconds, audio * 12.0)
        events.put({"i": i, "seconds": seconds, "audio": audio, "path": str(path)})

    events.put({"phase": "loading"})
    if capability == "preset":
        model = get_model("CustomVoice", p["size"])
        kw = dict(speaker=p["speaker"], instruct=(p.get("instruct") or None))
    elif capability == "clone":
        model = get_model("Base", p["vsize"])
        kw = dict(voice_clone_prompt=load_voice(p["ref"], expect_model_size=p["vsize"]))
    else:  # design
        model = get_model("VoiceDesign", "1.7B")
        kw = dict(instruct=p["desc"])

    events.put({"phase": "generating"})
    clock["t"] = time.time()
    wavs, sr = synth(
        capability, model, lines,
        language=_lang(p["language"]),
        sampling=delivery_sampling(p["delivery"]),
        chunk_size=CFG.chunk_size,
        on_line=on_line,
        should_cancel=_CANCEL.is_set,
        **kw,
    )
    return wavs, sr, part_paths


def _finish(kind, params, lines, gaps, sr, part_paths):
    """Stitch a levelled join from the per-line parts and build the redo state."""
    prefix = params.get("prefix", kind)
    joined = OUTPUT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    stitch(part_paths, str(joined), gaps=gaps, lufs=DEFAULT_LUFS)

    state = {
        "kind": kind,
        "params": params,
        "lines": list(lines),
        "gaps": list(gaps),
        "parts": list(part_paths),
        "sr": int(sr),
        "prefix": prefix,
    }
    save_session(state, str(joined), _parts_dir())
    n = len(lines)
    redo = gr.update(
        choices=[(f"{i + 1}. {ln[:60]}", str(i)) for i, ln in enumerate(lines)],
        value="0",
        visible=(n > 1),
    )
    return str(joined), state, redo


def _err(msg: str):
    """Five outputs: audio, progress, status, state, redo-dropdown."""
    return (gr.update(), _idle_html("Nothing rendering"),
            f'<div class="tts-note tts-note--err" role="alert">{msg}</div>',
            None, gr.update(visible=False))


def _loading_html(lines: List[str], started: float) -> str:
    n = len(lines)
    return (
        f'<div class="tts-prog" role="status" aria-live="polite">'
        f'<div class="tts-prog__top">'
        f'<span class="tts-prog__title">{_EQ}Loading the voice model</span>'
        f'<span class="tts-prog__meta">{_fmt_dur(time.time() - started)} elapsed</span></div>'
        f'<div class="tts-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="0" aria-label="Rendering progress">'
        f'<div class="tts-bar__fill" style="width:4%"></div></div>'
        f'<div class="tts-prog__meta" style="margin-top:10px">'
        f'First use of each model reads it from disk &mdash; minutes on CPU. '
        f'Then {n} line{"s" if n != 1 else ""} will render one at a time.</div>'
        f"</div>"
    )


def _stitch_done(kind, params, lines, gaps, done) -> Tuple[str, dict, Any]:
    """Join whatever finished into one levelled wav. Used for both a complete
    run and a cancelled one, so a stopped render still gives you usable audio."""
    ordered = sorted(done, key=lambda d: d["i"])
    parts = [d["path"] for d in ordered]
    n = len(parts)
    sr = sf.info(parts[0]).samplerate
    return _finish(kind, params, lines[:n], gaps[:n], sr, parts)


def _stream(kind: str, params: dict, lines: List[str], gaps: List[int], seed, note: str = ""):
    """Run one generation, yielding live progress to the browser as it goes.

    The model call happens on a worker thread; this generator just drains the
    event queue and re-renders. The generation path itself is untouched.
    """
    _CANCEL.clear()
    _seeded(seed)
    events: "queue.Queue" = queue.Queue()
    box: Dict[str, Any] = {}

    def worker():
        try:
            box["result"] = _run_synth(kind, params, lines, events)
        except BaseException as exc:  # noqa: BLE001 -- reported in the UI below
            box["error"] = exc
        finally:
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    started = time.time()
    done: List[Dict[str, Any]] = []
    phase = "loading"

    def frame():
        html = (_loading_html(lines, started) if phase == "loading"
                else _progress_html(lines, done, started, "running"))
        return gr.update(), html, gr.update(), gr.update(), gr.update()

    while True:
        try:
            evt = events.get(timeout=0.5)
        except queue.Empty:
            yield frame()
            continue
        if evt is None:
            break
        if "phase" in evt:
            phase = evt["phase"]
        else:
            done.append(evt)
        yield frame()

    thread.join(timeout=2.0)
    err = box.get("error")

    if isinstance(err, Cancelled) or (_CANCEL.is_set() and done):
        if not done:
            yield (gr.update(), _idle_html("Stopped", " Nothing finished rendering, so there is no audio to keep."),
                   '<div class="tts-note tts-note--warn">Stopped. Nothing was rendered.</div>',
                   None, gr.update(visible=False))
            return
        path, state, redo = _stitch_done(kind, params, lines, gaps, done)
        yield (path,
               _progress_html(lines, done, started, "cancelled",
                              note="Joined the lines that finished. The rest were not rendered."),
               f'<div class="tts-note tts-note--warn"><strong>Stopped after '
               f'{len(done)} of {len(lines)} lines.</strong><br>Saved to <code>{escape(path)}</code></div>',
               state, redo)
        return

    if err is not None:
        yield _err(f"<strong>Could not generate.</strong><br>{escape(type(err).__name__)}: {escape(str(err))}")
        return

    _wavs, sr, parts = box["result"]
    path, state, redo = _finish(kind, params, lines, gaps, sr, parts)
    n = len(lines)
    total = time.time() - started
    extra = ("<br>Not happy with one line? Use <strong>Fix one line</strong> below to redo just that one."
             if n > 1 else "")
    yield (path,
           _progress_html(lines, done, started, "done"),
           f'<div class="tts-note tts-note--ok"><strong>Done</strong> &mdash; {n} line'
           f'{"s" if n != 1 else ""} in {_fmt_dur(total)}, {sr} Hz.<br>'
           f'Saved to <code>{escape(path)}</code>{note}{extra}</div>',
           state, redo)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def voice_choices() -> List[Tuple[str, str]]:
    out = [(label, f"preset:{key}") for key, label in PRESET_SPEAKERS]
    saved = list_voices()
    if saved:
        out += [(f"{n}  (saved)", f"voice:{n}") for n in saved]
    return out


def do_cancel():
    _CANCEL.set()
    return '<div class="tts-note tts-note--warn">Stopping after the current line…</div>'


def do_preview(script, tidy, gap_ms, para_gap):
    """Show exactly what the model will be asked to say, and what it will cost."""
    try:
        lines, gaps = _prep(script, tidy, gap_ms, para_gap)
    except Exception as e:  # noqa: BLE001
        return f'<div class="tts-note tts-note--warn">Nothing to preview: {escape(str(e))}</div>'

    eta = estimate_seconds(lines)
    n = len(lines)
    head = f"<strong>{n} line{'s' if n != 1 else ''}</strong> will be generated"
    head += f", roughly <strong>{_fmt_dur(eta)}</strong> at this machine's measured speed." if eta \
        else ". Generate once and the time estimate calibrates itself."
    if not tidy:
        head += " <em>(text cleanup off &mdash; only line splits and the lexicon applied)</em>"

    rows = []
    for i, ln in enumerate(lines):
        gap = f'<span class="tts-prev__gap"> +{gaps[i]} ms pause before</span>' if i else ""
        rows.append(f'<div class="tts-prev__row"><span class="tts-prev__n">{i + 1}</span>'
                    f'<span>{escape(ln)}{gap}</span></div>')
    return (f'<div class="tts-note">{head}</div>'
            f'<div class="tts-prev">{"".join(rows)}</div>')


def do_speak(voice_value, script, size, language, instruct, delivery, tidy, gap_ms, para_gap, seed):
    try:
        if not voice_value:
            yield _err("Pick a voice first.")
            return
        lines, gaps = _prep(script, tidy, gap_ms, para_gap)
        kind_tag, ref = voice_value.split(":", 1)

        if kind_tag == "preset":
            note = ""
            if instruct and instruct.strip():
                warn_if_instruct_ignored("CustomVoice", SimpleNamespace(model_size=size), instruct)
                if size == "0.6B":
                    note = ("<br>Note: the style instruction was ignored &mdash; the 0.6B built-in "
                            "model drops it. Switch model size to 1.7B in Advanced for style control.")
            params = {"size": size, "language": language, "speaker": ref,
                      "instruct": instruct or "", "delivery": delivery, "prefix": f"speak_{ref}"}
            yield from _stream("preset", params, lines, gaps, seed, note)
            return

        # kind_tag == "voice": a saved clone. Its own model size wins.
        vsize = (voice_meta(ref) or {}).get("model_size", size)
        params = {"vsize": vsize, "language": language, "ref": ref, "delivery": delivery,
                  "prefix": f"speak_{ref}"}
        yield from _stream("voice", params, lines, gaps, seed)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        yield _err(f"<strong>Could not generate.</strong><br>{escape(type(e).__name__)}: {escape(str(e))}")


def do_design(description, script, language, delivery, tidy, gap_ms, para_gap, seed):
    try:
        if not description or not description.strip():
            yield _err("Describe the voice you want first.")
            return
        lines, gaps = _prep(script, tidy, gap_ms, para_gap)
        params = {"language": language, "desc": description.strip(), "delivery": delivery,
                  "prefix": "design"}
        yield from _stream("design", params, lines, gaps, seed)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        yield _err(f"<strong>Could not generate.</strong><br>{escape(type(e).__name__)}: {escape(str(e))}")


def do_regen_line(state, which, seed):
    """Regenerate a single line of the last take and re-stitch the whole read."""
    try:
        if not state:
            return gr.update(), "Generate something first, or restore a take in Settings.", state
        lines = state["lines"]
        try:
            idx = int(which)
        except (TypeError, ValueError):
            return gr.update(), "Pick which line to redo.", state
        if not 0 <= idx < len(lines):
            return gr.update(), "That line isn't in the last take.", state

        _CANCEL.clear()
        _seeded(seed)
        events: "queue.Queue" = queue.Queue()
        _wavs, _sr, new_parts = _run_synth(state["kind"], state["params"], [lines[idx]], events)

        parts = list(state["parts"])
        parts[idx] = new_parts[0]
        joined = OUTPUT_DIR / f"{state['prefix']}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        stitch(parts, str(joined), gaps=state["gaps"], lufs=DEFAULT_LUFS)

        new_state = {**state, "parts": parts}
        save_session(new_state, str(joined), _parts_dir())
        return (str(joined),
                f"Redid line {idx + 1} and re-stitched {len(parts)} lines. "
                f"Saved to `{joined}`",
                new_state)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return gr.update(), f"**Could not regenerate.** {type(e).__name__}: {e}", state


def _hue(name: str) -> int:
    """A stable avatar hue per voice name, kept in the warm half of the wheel."""
    return (sum(ord(c) * (i + 7) for i, c in enumerate(name)) % 70) - 20


def _initials(label: str) -> str:
    parts = [p for p in label.replace("_", " ").replace("-", " ").split() if p]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def _voice_card_html(value: Optional[str]) -> str:
    """Explain the selected voice: what it is, how it was made, what it suits."""
    if not value or ":" not in value:
        return _idle_html("No voice selected", " Pick one from the list above.", art=False)
    kind, ref = value.split(":", 1)
    if kind == "preset":
        label = dict(PRESET_SPEAKERS).get(ref, ref).split("  (")[0]
        english = ref in ("ryan", "aiden")
        tags = ['<span class="tts-tag">Built-in</span>',
                '<span class="tts-tag tts-tag--ok">English</span>' if english
                else '<span class="tts-tag tts-tag--warn">Other languages</span>']
        desc = ("A stock Qwen3-TTS speaker. Clear and reliable, but no South African accent; "
                "use a cloned voice for that.") if english else (
                "Tuned for another language. It will read English, with that accent.")
    else:
        label = ref
        meta = voice_meta(ref) or {}
        tags = ['<span class="tts-tag tts-tag--accent">Cloned</span>',
                f'<span class="tts-tag">{escape(str(meta.get("model_size", "?")))}</span>',
                '<span class="tts-tag">timbre only</span>' if meta.get("x_vector_only_mode")
                else '<span class="tts-tag tts-tag--ok">full clone</span>']
        if meta.get("created"):
            tags.append(f'<span class="tts-tag">{escape(meta["created"])}</span>')
        said = (meta.get("ref_text") or "").strip()
        more = "…" if len(said) > 110 else ""
        desc = (f"Cloned from a reference clip: <q>{escape(said[:110])}{more}</q>"
                if said else "Cloned from a reference clip, timbre only.")
    return (f'<div class="tts-vcard"><div class="tts-avatar" aria-hidden="true" style="--h:{_hue(label)}">'
            f"{escape(_initials(label))}</div><div><h3>{escape(label)}</h3><p>{desc}</p>"
            f'<div class="tts-tags">{"".join(tags)}</div></div></div>')


def _voices_gallery_html() -> str:
    names = list_voices()
    if not names:
        return _idle_html("No saved voices yet",
                          " Record or upload a clip on the left and it will appear here.")
    return ('<div class="tts-gallery">'
            + "".join(_voice_card_html(f"voice:{n}") for n in names) + "</div>")


def _clip_report_html(path) -> str:
    """The quality report shown as soon as a clip is recorded or uploaded."""
    if not path:
        return ""
    try:
        r = refclip.analyze(path)
    except Exception as e:  # noqa: BLE001
        return ('<div class="tts-qa tts-qa--unusable"><div class="tts-qa__grade">&#10007; Could not read '
                f"this file</div><div>{escape(type(e).__name__)}: {escape(str(e))}. "
                "Try a WAV, FLAC or MP3.</div></div>")
    icon, verdict = {"good": ("&#10003;", "Great clip"), "usable": ("!", "Usable, with notes"),
                     "unusable": ("&#10007;", "This clip won't make a good voice")}[r.grade]
    span = 45.0
    mark = min(100.0, r.duration / span * 100)
    lo, hi = refclip.IDEAL_SECONDS
    stats = [("Length", f"{r.duration:.1f}s"), ("Speech", f"{r.speech_seconds:.1f}s"),
             ("Loudness", f"{r.rms_db:.0f} dBFS"), ("Peak", f"{r.peak_db:.1f} dBFS")]
    issues = "".join(
        f'<li><strong>{"Problem" if lvl == "err" else "Note"}:</strong> {escape(m)}</li>'
        for lvl, m in r.issues)
    channels = "stereo" if r.channels > 1 else "mono"
    return (
        f'<div class="tts-qa tts-qa--{r.grade}" role="status">'
        f'<div class="tts-qa__head"><span class="tts-qa__grade">{icon} {verdict}</span>'
        f'<span class="tts-tag">{r.sample_rate} Hz &middot; {channels}</span></div>'
        f'<div class="tts-lenbar__cap"><span>Length</span><span>ideal {lo:.0f}&ndash;{hi:.0f} s, '
        f'shown in green</span></div>'
        f'<div class="tts-lenbar" role="img" aria-label="Length {r.duration:.1f} seconds; ideal is '
        f'{lo:.0f} to {hi:.0f} seconds">'
        f'<div class="tts-lenbar__zone" style="left:{lo / span * 100:.1f}%;width:{(hi - lo) / span * 100:.1f}%"></div>'
        f'<div class="tts-lenbar__mark" style="left:calc({mark:.1f}% - 1px)"></div></div>'
        '<div class="tts-qa__stats">'
        + "".join(f'<div class="tts-stat"><small>{k}</small><b>{v}</b></div>' for k, v in stats)
        + "</div>"
        + (f"<ul>{issues}</ul>" if issues else
           '<div style="font-size:.84rem">Clear, well-levelled and a good length. '
           "Add the transcript and save.</div>")
        + "</div>"
    )


def _bad(msg: str) -> str:
    return f'<div class="tts-note tts-note--err" role="alert">{msg}</div>'


def do_add_voice(name, ref_audio, ref_text, xvec_only, size,
                 progress=gr.Progress(track_tqdm=True)):
    """Returns (status html, speak voice dropdown, gallery html, manage dropdown)."""
    keep = (gr.update(), gr.update(), gr.update())
    name = (name or "").strip()
    try:
        if not name:
            return (_bad("Give the voice a name."), *keep)
        if not name.replace("_", "").replace("-", "").isalnum():
            return (_bad("Use only letters, numbers, - and _ in the name."), *keep)
        if not ref_audio:
            return (_bad("Record or upload a reference clip first."), *keep)
        if not xvec_only and not (ref_text or "").strip():
            return (_bad("Type the transcript of the clip (word for word), or tick "
                         "<strong>timbre only</strong> to skip it (lower quality)."), *keep)
        if voice_path(name).exists():
            return (_bad(f"A voice called <strong>{escape(name)}</strong> already exists. "
                         "Pick another name or delete it on the right."), *keep)

        progress(0.05, desc="Checking the clip…")
        clean, report = refclip.prepare(ref_audio, ref_clip_path(name))
        if not report.ok:
            clean.unlink(missing_ok=True)
            why = "<br>".join(escape(m) for lvl, m in report.issues if lvl == "err")
            return (_bad(f"<strong>That clip can't make a usable voice.</strong><br>{why}"), *keep)

        progress(0.2, desc="Loading the cloning model…")
        model = get_model("Base", size)
        progress(0.6, desc="Learning the voice…")
        items = model.create_voice_clone_prompt(
            ref_audio=str(clean),
            ref_text=((ref_text or "").strip() or None),
            x_vector_only_mode=bool(xvec_only),
        )
        save_voice(name, items, ref_audio=str(clean), model_size=size)
        gr.Info(f"Saved the voice '{name}'.")
        names = list_voices()
        return (f'<div class="tts-note tts-note--ok"><strong>Saved {escape(name)}.</strong> '
                "It's selected on the Speak tab, ready to use.</div>",
                gr.update(choices=voice_choices(), value=f"voice:{name}"),
                _voices_gallery_html(),
                gr.update(choices=names, value=name))
    except (Exception, SystemExit) as e:  # noqa: BLE001
        # A failed encode must not leave an orphaned clip behind.
        if name and not voice_path(name).exists():
            ref_clip_path(name).unlink(missing_ok=True)
        return (_bad(f"<strong>Could not save the voice.</strong><br>"
                     f"{escape(type(e).__name__)}: {escape(str(e))}"), *keep)


def do_pick_voice(name):
    """Manage panel: audition the clip a saved voice was made from."""
    if not name:
        return None, _idle_html("No voice selected", art=False)
    ref = ref_clip_path(name)
    if not ref.exists():
        # Voices made before clips were kept beside them record wherever the
        # clip was, often relative to scripts/ (the CLI's working directory).
        old = (voice_meta(name) or {}).get("ref_audio")
        old = Path(old) if old else None
        if old is not None and not old.is_absolute():
            old = (Path(__file__).resolve().parent / old).resolve()
        ref = old if old is not None and old.exists() else None
    return (str(ref) if ref else None), _voice_card_html(f"voice:{name}")


def do_delete_voice(name, armed):
    """Two-step delete: the first press arms it, the second removes the voice
    and its reference clip. Returns (status, speak dropdown, manage dropdown,
    button, armed)."""
    try:
        if not name:
            return _bad("Pick a voice to delete."), gr.update(), gr.update(), gr.update(), False
        if not armed:
            return ('<div class="tts-note tts-note--warn">Press <strong>Confirm delete</strong> to remove '
                    f"<strong>{escape(name)}</strong> for good. Its saved clip is deleted too.</div>",
                    gr.update(), gr.update(), gr.update(value="Confirm delete"), True)
        delete_voice(name)
        gr.Info(f"Deleted the voice '{name}'.")
        remaining = list_voices()
        return ("",
                gr.update(choices=voice_choices(), value="preset:ryan"),
                gr.update(choices=remaining, value=(remaining[0] if remaining else None)),
                gr.update(value="Delete voice"), False)
    except Exception as e:  # noqa: BLE001
        return _bad(f"Could not delete: {escape(str(e))}"), gr.update(), gr.update(), gr.update(), False


def refresh_voice_lists():
    names = list_voices()
    return (gr.update(choices=voice_choices()),
            gr.update(choices=names, value=(names[0] if names else None)),
            _voices_gallery_html())


# ---------------------------------------------------------------------------
# Recordings tab
# ---------------------------------------------------------------------------
def _recording_choices() -> List[Tuple[str, str]]:
    """Finished joins in output/, newest first. Per-line parts are excluded --
    they are working files, and the joined take is what you actually want."""
    try:
        files = [p for p in OUTPUT_DIR.glob("*.wav") if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:80]:
        try:
            info = sf.info(str(p))
            when = time.strftime("%d %b %H:%M", time.localtime(p.stat().st_mtime))
            out.append((f"{p.name}   ·   {info.duration:.1f}s   ·   {when}", str(p)))
        except Exception:  # noqa: BLE001 -- a half-written wav shouldn't hide the rest
            continue
    return out


def do_refresh_recordings():
    choices = _recording_choices()
    return gr.update(choices=choices, value=(choices[0][1] if choices else None))


def do_pick_recording(path):
    """Returns (player, meta card, download button)."""
    if not path or not Path(path).exists():
        return (None, _idle_html("No recordings yet", " Finished takes appear here, newest first."),
                gr.update(visible=False))
    try:
        info = sf.info(str(path))
        when = time.strftime("%d %b %Y, %H:%M", time.localtime(Path(path).stat().st_mtime))
        meta = (f'<div class="tts-vcard"><div class="tts-avatar" aria-hidden="true" style="--h:{_hue(Path(path).stem)}">'
                f'{_icon("play")}</div><div><h3>{escape(Path(path).name)}</h3>'
                f"<p>{when}</p><div class=\"tts-tags\">"
                f'<span class="tts-tag tts-tag--accent">{info.duration:.1f}s</span>'
                f'<span class="tts-tag">{info.samplerate} Hz</span>'
                f'<span class="tts-tag">{escape(info.subtype)}</span></div></div></div>')
        return str(path), meta, gr.update(value=str(path), visible=True)
    except Exception as e:  # noqa: BLE001
        return None, _bad(f"Could not read: {escape(str(e))}"), gr.update(visible=False)


def do_delete_recording(path, armed):
    """Two-step delete of one finished read. Returns (status, list, button, armed)."""
    if not path or not Path(path).exists():
        return _bad("Pick a recording first."), gr.update(), gr.update(), False
    if not armed:
        return ('<div class="tts-note tts-note--warn">Press <strong>Confirm delete</strong> to remove '
                f"<strong>{escape(Path(path).name)}</strong>. This can't be undone.</div>",
                gr.update(), gr.update(value="Confirm delete"), True)
    try:
        Path(path).unlink()
    except OSError as e:
        return _bad(f"Could not delete: {escape(str(e))}"), gr.update(), gr.update(value="Delete"), False
    gr.Info(f"Deleted {Path(path).name}.")
    choices = _recording_choices()
    return ("", gr.update(choices=choices, value=(choices[0][1] if choices else None)),
            gr.update(value="Delete"), False)


# ---------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------
def do_load_lexicon():
    try:
        return LEXICON_PATH.read_text(encoding="utf-8")
    except OSError:
        return ("# Pronunciation overrides, one per line:  pattern -> replacement\n"
                "# Wrap the pattern in /slashes/ for a regex.\n"
                "# Nguyen -> win\n")


def do_save_lexicon(text):
    try:
        LEXICON_PATH.write_text(text or "", encoding="utf-8")
        # textprep re-reads on mtime change, so the next generation picks this up
        # with no restart.
        return ('<div class="tts-note tts-note--ok">Saved. It applies from the next '
                'generation &mdash; no restart needed.</div>')
    except OSError as e:
        return f'<div class="tts-note tts-note--err">Could not save: {escape(str(e))}</div>'


def _session_choices() -> List[Tuple[str, str]]:
    out = []
    for s in list_sessions(_parts_dir(), limit=25):
        n = len(s.get("lines", []))
        first = (s.get("lines") or [""])[0][:52]
        out.append((f"{s.get('created_human', '?')}  ·  {n} lines  ·  {first}", s["_path"]))
    return out


def do_refresh_sessions():
    choices = _session_choices()
    return gr.update(choices=choices, value=(choices[0][1] if choices else None))


def do_restore_session(path):
    """Load an earlier take back into the Speak tab so its lines can be redone."""
    if not path:
        return None, gr.update(visible=False), '<div class="tts-note">Pick a take to restore.</div>', gr.update()
    state = load_session(path)
    if not state:
        return (None, gr.update(visible=False),
                '<div class="tts-note tts-note--err">That take can no longer be restored '
                '(its per-line audio was deleted).</div>', gr.update())
    lines = state["lines"]
    redo = gr.update(choices=[(f"{i + 1}. {ln[:60]}", str(i)) for i, ln in enumerate(lines)],
                     value="0", visible=len(lines) > 1)
    return (state, redo,
            f'<div class="tts-note tts-note--ok"><strong>Restored {len(lines)} lines.</strong><br>'
            "Go to the Speak tab and use <strong>Fix one line</strong> to redo any of them.</div>",
            gr.update())


def do_cleanup(dry_run: bool = False):
    removed = cleanup_orphans(_parts_dir(), max_age_hours=24.0, dry_run=dry_run)
    if not removed:
        return ('<div class="tts-note tts-note--ok">Nothing to clean &mdash; every per-line file '
                'belongs to a take you can still restore.</div>')
    names = "".join(f"<li>{escape(p.name)}</li>" for p in removed[:12])
    more = f"<li>…and {len(removed) - 12} more</li>" if len(removed) > 12 else ""
    verb = "Would delete" if dry_run else "Deleted"
    return (f'<div class="tts-note"><strong>{verb} {len(removed)} orphaned file'
            f'{"s" if len(removed) != 1 else ""}</strong> (older than 24 h, not part of any '
            f'restorable take).<ul>{names}{more}</ul></div>')


def do_system_refresh():
    return _system_html()


def do_system_detail_refresh():
    return _system_detail_html()


def do_reset_speed():
    removed = reset_speed()
    msg = ("Speed calibration cleared. The next generation starts measuring again."
           if removed else "Nothing to clear - no calibration has been recorded yet.")
    return f'<div class="tts-note tts-note--ok">{msg}</div>'


# ---------------------------------------------------------------------------
# UI
#
# Layout follows the order of the job: what to say, who says it, then the
# result. The right-hand column is the render column -- progress first, because
# that is what you watch, then the audio it produces, then the tools for
# repairing it. Headings step h1 -> h2 -> h3 without skipping, every control
# carries a real label, and status is announced politely to screen readers.
# ---------------------------------------------------------------------------
DELIVERY_INFO = ("Steady is calmer and more consistent, best on the small (0.6B) voice. "
                 "Expressive varies more but can wander.")
TIDY_INFO = ("Spell out numbers, money, dates and symbols; strip markdown; one sentence per line. "
             "Pronunciation fixes in the lexicon always apply.")

# One-click starter scripts. Fictional places and offers, so nothing reads as a
# real advert or news claim.
SAMPLE_SCRIPTS = {
    "Howzit": ("Howzit everyone, and welcome back! It's a lekker sunny morning in Cape Town.\n\n"
               "The wind has finally calmed down, so grab a koeksister and let's get going."),
    "Radio spot": ("This weekend only at Harbour Lane Market: everything is up to fifty percent off!\n\n"
                   "Doors open at 8 a.m. on Saturday. Don't miss it, my bru."),
    "Story": ("My ouma always said the mountain keeps its secrets. One windy evening in the Bo-Kaap, "
              "I finally understood what she meant."),
    "Announcement": ("Good afternoon, everyone. The 3:15 train to Simon's Town is running about ten "
                     "minutes late.\n\nThank you for your patience, and enjoy the rest of your day."),
}

DESIGN_IDEAS = {
    "Cape Town auntie": "Warm Capetonian auntie in her fifties, kind, lively and a little cheeky, South African English",
    "Radio DJ": "Young Cape Flats radio DJ, upbeat and fast-talking, full of energy, South African English",
    "Narrator": "Calm, deep male narrator, measured pace, warm and trustworthy",
    "Help desk": "Bright young woman, friendly customer-service tone, clear and crisp",
}

# A passage to read when recording a reference clip: ~15 seconds, varied sounds,
# a natural Capetonian register. Reading it means the transcript is known.
READ_PASSAGE = ("Morning everyone! It's a lekker sunny day in Cape Town, and the wind is finally "
                "taking a break. I'm going to grab a coffee, catch the train into town, and meet my "
                "friends by the harbour at half past ten. Honestly, there's nowhere else I'd rather be.")


def _masthead_html() -> str:
    return (
        '<div class="tts-hero">'
        f"{_hero_art()}"
        '<div class="tts-hero__brand">'
        f'<span class="tts-logo">{LOGO_SVG}</span>'
        f"<div><h1>Skinnerbek <span>Studio</span></h1>"
        "<p>Natural South African voices, from Cape Flats to Constantia. Write a script, pick a "
        "voice, press Speak.</p>"
        '<div class="tts-hero__badges">'
        f'<span class="tts-badge">{_icon("lock")} Private: runs on this machine</span>'
        f'<span class="tts-badge">{_icon("wifi_off")} Works offline</span>'
        "</div></div></div>"
        '<div class="tts-hero__actions">'
        f'<button class="tts-iconbtn" type="button" data-skb="theme" title="Dark theme" '
        f'aria-label="Dark theme" aria-pressed="false">{_icon("moon")}</button>'
        "</div></div>"
    )


WELCOME_HTML = (
    '<section class="tts-welcome" aria-label="Getting started"><ol class="tts-welcome__steps">'
    '<li class="tts-step"><i aria-hidden="true">1</i><div><b>Write or paste a script</b>'
    "<span>Or start from a sample. Long text is split into sentences for you.</span></div></li>"
    '<li class="tts-step"><i aria-hidden="true">2</i><div><b>Pick a voice</b>'
    "<span>Clone a real Cape Town voice in the Voice library for an authentic accent.</span></div></li>"
    '<li class="tts-step"><i aria-hidden="true">3</i><div><b>Press Speak</b>'
    "<span>Or <kbd>Ctrl</kbd> + <kbd>Enter</kbd>. Each line renders, then joins into one take.</span></div></li>"
    "</ol>"
    '<button class="tts-welcome__x" type="button" data-skb="welcome-close" '
    'aria-label="Dismiss the getting-started steps">&times;</button>'
    "</section>"
)

FOOTER_HTML = (
    '<div class="tts-foot">'
    f"<div><b>{APP_NAME}</b> <span>v{APP_VERSION}</span><span>Everything stays on this computer</span></div>"
    "<div><span>Built on Qwen3-TTS (Apache-2.0)</span><span>Only clone voices you have permission to use</span></div>"
    "</div>"
)

# The text box's live counter, computed in the browser so typing never waits on
# the server. Mirrors the app's own estimate: ~2.6 words a second of speech.
COUNTER_JS = """(text) => {
  const t = (text || '').trim();
  const words = t ? t.split(/\\s+/).length : 0;
  const sentences = t ? (t.match(/[^.!?\\n]+([.!?]+|$)/g) || []).filter(s => s.trim()).length : 0;
  const secs = Math.round(words / 2.6);
  const dur = secs < 60 ? secs + 's' : Math.floor(secs / 60) + 'm ' + String(secs % 60).padStart(2, '0') + 's';
  const long = words > 400 ? ' class="over"' : '';
  return `<div class="tts-count"><span><b>${words}</b> words</span><span><b>${sentences}</b> sentences</span>` +
         `<span${long}>about <b>${dur}</b> of speech${words > 400 ? ' · long reads take a while on CPU' : ''}</span></div>`;
}"""


def _counter_static() -> str:
    return ('<div class="tts-count"><span><b>0</b> words</span><span><b>0</b> sentences</span>'
            "<span>about <b>0s</b> of speech</span></div>")


WAVEFORM = gr.WaveformOptions(waveform_color="#8d97ad", waveform_progress_color="#ff7a55")


def _fix_one_line_block():
    """The 'redo a single line' controls. Returns (dropdown, seed, button, status)."""
    with gr.Accordion("Fix one line", open=False, elem_classes="tts-pane"):
        gr.HTML('<div class="tts-note">Redo just one sentence of the last take. The rest is kept '
                "and the read is re-stitched. Change the seed for a different attempt.</div>")
        line = gr.Dropdown(label="Line to redo", choices=[], visible=False)
        new_seed = gr.Number(label="Seed", precision=0, value=0,
                             info="0 gives a fresh attempt each time. Any other number is repeatable.")
        with gr.Row(elem_classes="tts-actions"):
            btn = gr.Button("Regenerate this line", variant="primary")
        status = gr.Markdown(elem_classes="tts-status")
    return line, new_seed, btn, status


def _render_column(label_prefix: str, button_word: str):
    """Progress -> result -> status -> prepared text -> repair. Returns the parts."""
    with gr.Column(elem_classes=["tts-card", "tts-card--sticky"]):
        gr.HTML(_title("render", "Your take", "Each line appears here as it renders."))
        progress = gr.HTML(_idle_html(
            "Ready when you are", f" Press {button_word} and watch each line render here."))
        audio = gr.Audio(label=f"{label_prefix} result", type="filepath",
                         elem_classes=["tts-player", "tts-result"],
                         waveform_options=WAVEFORM)
        status = gr.HTML(elem_classes="tts-status")
        with gr.Accordion("Prepared text", open=False, elem_classes="tts-pane"):
            preview = gr.HTML(_idle_html(
                "No preview yet",
                " Press Preview for the exact lines the model will be given, and a time estimate.",
                art=False))
        redo_line, redo_seed, redo_btn, redo_status = _fix_one_line_block()
    return progress, audio, status, preview, redo_line, redo_seed, redo_btn, redo_status


def _sample_buttons(target, samples: Dict[str, str]):
    with gr.Row(elem_classes="tts-samples"):
        for label, text in samples.items():
            gr.Button(label, size="sm").click(lambda t=text: t, outputs=[target], queue=False)


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_NAME) as demo:
        with gr.Column(elem_classes="tts-head"):
            gr.HTML(_masthead_html())
            sysbar = gr.HTML(_system_html())

        with gr.Tabs(elem_classes="tts-tabs") as tabs:
            # ================= Speak =================
            with gr.Tab("Speak", id="speak"):
                speak_state = gr.State(None)
                gr.HTML(WELCOME_HTML)
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("script", "Script", "What should be said. Try a sample to get going."))
                            script = gr.Textbox(
                                label="Script", show_label=False, lines=9, max_lines=30,
                                placeholder=(
                                    "Type or paste what the voice should say…\n\n"
                                    "Long text is split into one sentence per line automatically. "
                                    "A blank line starts a new paragraph; lines starting with # are skipped."
                                ),
                            )
                            counter = gr.HTML(_counter_static())
                            _sample_buttons(script, SAMPLE_SCRIPTS)

                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("voice", "Voice", "Built-in speakers, plus every voice you've cloned."))
                            with gr.Row(equal_height=False, elem_classes="tts-pickrow"):
                                voice = gr.Dropdown(label="Voice", show_label=False, choices=voice_choices(),
                                                    value="preset:ryan", scale=5)
                                refresh_btn = gr.Button("Refresh", scale=0, min_width=96,
                                                        variant="secondary", elem_classes="tts-refresh")
                            voice_card = gr.HTML(_voice_card_html("preset:ryan"))
                            delivery = gr.Radio(["Steady", "Natural", "Expressive"], value="Steady",
                                                label="Delivery", info=DELIVERY_INFO,
                                                elem_classes="tts-seg")
                            tidy = gr.Checkbox(value=True, label="Tidy the text before speaking",
                                               info=TIDY_INFO)
                            with gr.Accordion("Advanced settings", open=False, elem_classes="tts-pane"):
                                size = gr.Radio(
                                    ["0.6B", "1.7B"], value="0.6B", label="Model size", elem_classes="tts-seg",
                                    info="0.6B is faster and lighter. 1.7B needs ~3.8 GB free RAM but "
                                         "enables the style instruction. Saved voices use their own size.",
                                )
                                language = gr.Dropdown(LANGUAGES, value="Auto", label="Language",
                                                       info="Auto detects it from the text.")
                                instruct = gr.Textbox(
                                    label="Style instruction (built-in voices, 1.7B only)",
                                    placeholder="e.g. Speak slowly and warmly", lines=2,
                                )
                                with gr.Row():
                                    gap_ms = gr.Slider(0, 1500, value=300, step=50,
                                                       label="Pause between sentences (ms)")
                                    para_gap = gr.Slider(0, 2000, value=650, step=50,
                                                         label="Pause between paragraphs (ms)")
                                seed = gr.Number(
                                    label="Seed", precision=0, value=0,
                                    info="0 gives a fresh take each time. Any other number makes a take repeatable.",
                                )
                            with gr.Row(elem_classes="tts-actions"):
                                speak_btn = gr.Button("Speak", variant="primary", scale=4,
                                                      elem_id="tts-speak-btn", elem_classes="tts-go")
                                preview_btn = gr.Button("Preview", scale=1, variant="secondary")
                                stop_btn = gr.Button("Stop", variant="secondary", scale=1,
                                                     elem_classes="tts-cancel")
                            gr.HTML('<div class="tts-hint">Tip: press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> '
                                    "(<kbd>&#8984;</kbd> + <kbd>Enter</kbd> on a Mac) to speak</div>")
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        (s_prog, audio, status, preview_md,
                         redo_line, redo_seed, redo_btn, redo_status) = _render_column("Speak", "Speak")

                script.input(None, inputs=[script], outputs=[counter], js=COUNTER_JS)
                script.change(None, inputs=[script], outputs=[counter], js=COUNTER_JS)
                voice.change(_voice_card_html, inputs=[voice], outputs=[voice_card], queue=False)
                speak_btn.click(
                    do_speak,
                    inputs=[voice, script, size, language, instruct, delivery, tidy,
                            gap_ms, para_gap, seed],
                    outputs=[audio, s_prog, status, speak_state, redo_line],
                )
                # queue=False so Stop is handled immediately rather than waiting
                # behind the generation it is meant to interrupt.
                stop_btn.click(do_cancel, outputs=[status], queue=False)
                preview_btn.click(do_preview, inputs=[script, tidy, gap_ms, para_gap],
                                  outputs=[preview_md])
                redo_btn.click(do_regen_line, inputs=[speak_state, redo_line, redo_seed],
                               outputs=[audio, redo_status, speak_state])

            # ================= Design a voice =================
            with gr.Tab("Design a voice", id="design"):
                d_state = gr.State(None)
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("spark", "Describe a voice",
                                           "Invent a new voice from words. Uses the 1.7B VoiceDesign model: "
                                           "the slowest, and it needs ~3.8&nbsp;GB of free RAM."))
                            d_desc = gr.Textbox(
                                label="The voice", lines=3,
                                placeholder="e.g. Warm older man, gentle and unhurried, slight gravel",
                            )
                            _sample_buttons(d_desc, DESIGN_IDEAS)
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("script", "Script", "What this voice should say."))
                            d_script = gr.Textbox(label="Script", show_label=False, lines=7,
                                                  placeholder="What should this voice say?")
                            d_counter = gr.HTML(_counter_static())
                            _sample_buttons(d_script, SAMPLE_SCRIPTS)
                            d_delivery = gr.Radio(["Steady", "Natural", "Expressive"], value="Steady",
                                                  label="Delivery", info=DELIVERY_INFO, elem_classes="tts-seg")
                            d_tidy = gr.Checkbox(value=True, label="Tidy the text before speaking",
                                                 info=TIDY_INFO)
                            with gr.Accordion("Advanced settings", open=False, elem_classes="tts-pane"):
                                d_language = gr.Dropdown(LANGUAGES, value="Auto", label="Language")
                                with gr.Row():
                                    d_gap = gr.Slider(0, 1500, value=300, step=50,
                                                      label="Pause between sentences (ms)")
                                    d_para_gap = gr.Slider(0, 2000, value=650, step=50,
                                                           label="Pause between paragraphs (ms)")
                                d_seed = gr.Number(label="Seed", precision=0, value=0,
                                                   info="0 gives a fresh take each time.")
                            with gr.Row(elem_classes="tts-actions"):
                                d_btn = gr.Button("Speak", variant="primary", scale=4,
                                                  elem_id="tts-design-btn", elem_classes="tts-go")
                                d_preview_btn = gr.Button("Preview", scale=1, variant="secondary")
                                d_stop = gr.Button("Stop", variant="secondary", scale=1,
                                                   elem_classes="tts-cancel")
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        (d_prog, d_audio, d_status, d_preview_md,
                         d_redo_line, d_redo_seed, d_redo_btn, d_redo_status) = _render_column("Design", "Speak")

                d_script.input(None, inputs=[d_script], outputs=[d_counter], js=COUNTER_JS)
                d_script.change(None, inputs=[d_script], outputs=[d_counter], js=COUNTER_JS)
                d_btn.click(
                    do_design,
                    inputs=[d_desc, d_script, d_language, d_delivery, d_tidy,
                            d_gap, d_para_gap, d_seed],
                    outputs=[d_audio, d_prog, d_status, d_state, d_redo_line],
                )
                d_stop.click(do_cancel, outputs=[d_status], queue=False)
                d_preview_btn.click(do_preview, inputs=[d_script, d_tidy, d_gap, d_para_gap],
                                    outputs=[d_preview_md])
                d_redo_btn.click(do_regen_line, inputs=[d_state, d_redo_line, d_redo_seed],
                                 outputs=[d_audio, d_redo_status, d_state])

            # ================= Voice library =================
            with gr.Tab("Voice library", id="voices") as voices_tab:
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("mic", "Clone a voice",
                                           "Record or upload 10&ndash;20 seconds of clear speech. Only use "
                                           "voices you have the speaker's permission to clone."))
                            gr.HTML(_sec("Name it", 3, 1))
                            a_name = gr.Textbox(label="Voice name", placeholder="e.g. auntie_faiza",
                                                show_label=False)
                            gr.HTML(_sec("Record or upload a clip", 3, 2))
                            gr.HTML('<div class="tts-readme"><small>Read this aloud</small>'
                                    f"<q>{escape(READ_PASSAGE)}</q></div>")
                            recorder = gr.HTML(value="", html_template='<div class="tts-rec"></div>',
                                               js_on_load=RECORDER_JS, elem_classes="tts-recorder")
                            a_audio = gr.Audio(
                                label="Reference clip (record above, or drop a file here)",
                                type="filepath", sources=["upload"], elem_classes="tts-player",
                                waveform_options=WAVEFORM,
                            )
                            a_report = gr.HTML()
                            gr.HTML(_sec("Transcript", 3, 3))
                            a_text = gr.Textbox(
                                label="Exactly what is said in the clip", lines=3,
                                placeholder="Type exactly what is said in the clip, word for word.",
                            )
                            use_passage = gr.Button("I read the passage above", size="sm",
                                                    elem_classes="tts-pill")
                            gr.HTML(_sec("Save", 3, 4))
                            with gr.Row():
                                a_size = gr.Radio(
                                    ["0.6B", "1.7B"], value="0.6B", label="Model size", elem_classes="tts-seg",
                                    info="A voice is tied to the size that made it. 1.7B needs ~3.8 GB free RAM.",
                                )
                                a_xvec = gr.Checkbox(
                                    label="Timbre only (skip the transcript: quicker, lower quality)",
                                    value=False,
                                )
                            with gr.Row(elem_classes="tts-actions"):
                                a_btn = gr.Button("Save this voice", variant="primary", elem_classes="tts-go")
                            a_status = gr.HTML(elem_classes="tts-status")
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("grid", "Your voices", "Everything you've cloned."))
                            gallery = gr.HTML(_voices_gallery_html())
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("play", "Manage a voice", "Hear the clip it was made from, use it, or remove it."))
                            v_pick = gr.Dropdown(label="Voice", choices=list_voices(),
                                                 value=(list_voices()[0] if list_voices() else None))
                            v_card = gr.HTML()
                            v_audio = gr.Audio(label="Reference clip", type="filepath", interactive=False,
                                               elem_classes="tts-player", waveform_options=WAVEFORM)
                            with gr.Row(elem_classes="tts-actions"):
                                v_use = gr.Button("Use in Speak", variant="primary")
                                v_del = gr.Button("Delete voice", variant="stop")
                            v_armed = gr.State(False)
                            v_status = gr.HTML(elem_classes="tts-status")

                recorder.change(lambda p: p or gr.update(), inputs=[recorder], outputs=[a_audio])
                a_audio.change(_clip_report_html, inputs=[a_audio], outputs=[a_report])
                use_passage.click(lambda: READ_PASSAGE, outputs=[a_text], queue=False)
                a_btn.click(do_add_voice, inputs=[a_name, a_audio, a_text, a_xvec, a_size],
                            outputs=[a_status, voice, gallery, v_pick])
                v_pick.change(do_pick_voice, inputs=[v_pick], outputs=[v_audio, v_card]).then(
                    lambda: (False, gr.update(value="Delete voice"), ""), outputs=[v_armed, v_del, v_status])
                v_use.click(lambda n: (gr.update(value=f"voice:{n}") if n else gr.update(),
                                       gr.Tabs(selected="speak")),
                            inputs=[v_pick], outputs=[voice, tabs])
                v_del.click(do_delete_voice, inputs=[v_pick, v_armed],
                            outputs=[v_status, voice, v_pick, v_del, v_armed]).then(
                    lambda: _voices_gallery_html(), outputs=[gallery])

            # ================= Recordings =================
            with gr.Tab("Recordings", id="recordings") as rec_tab:
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("history", "Recordings", "Everything you've generated, newest first."))
                            with gr.Row(equal_height=False, elem_classes="tts-pickrow"):
                                rec_list = gr.Dropdown(label="Recording", choices=_recording_choices(),
                                                       value=None, scale=5)
                                rec_refresh = gr.Button("Refresh", scale=0, min_width=96,
                                                        variant="secondary", elem_classes="tts-refresh")
                            rec_meta = gr.HTML()
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("play", "Playback"))
                            rec_audio = gr.Audio(label="Selected recording", type="filepath",
                                                 interactive=False, elem_classes="tts-player",
                                                 waveform_options=WAVEFORM)
                            with gr.Row(elem_classes="tts-actions"):
                                rec_dl = gr.DownloadButton("Download", variant="primary", visible=False)
                                rec_del = gr.Button("Delete", variant="stop")
                            rec_armed = gr.State(False)
                            rec_status = gr.HTML(elem_classes="tts-status")

                rec_list.change(do_pick_recording, inputs=[rec_list],
                                outputs=[rec_audio, rec_meta, rec_dl]).then(
                    lambda: (False, gr.update(value="Delete"), ""), outputs=[rec_armed, rec_del, rec_status])
                rec_refresh.click(do_refresh_recordings, outputs=[rec_list])
                rec_del.click(do_delete_recording, inputs=[rec_list, rec_armed],
                              outputs=[rec_status, rec_list, rec_del, rec_armed])

            # ================= Settings =================
            with gr.Tab("Settings", id="settings") as settings_tab:
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("chip", "This machine", "What the app has to work with, and how fast it runs."))
                            sys_detail = gr.HTML(_system_detail_html())
                            with gr.Row(elem_classes="tts-actions"):
                                sys_btn = gr.Button("Re-check", variant="secondary")
                                speed_btn = gr.Button("Reset speed calibration", variant="secondary")
                            speed_status = gr.HTML(elem_classes="tts-status")

                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("history", "Restore an earlier take",
                                           "Bring a finished read back so you can redo a single line, even "
                                           "after a restart."))
                            sess_list = gr.Dropdown(label="Saved takes", choices=_session_choices(), value=None)
                            with gr.Row(elem_classes="tts-actions"):
                                sess_refresh = gr.Button("Refresh", variant="secondary")
                                sess_restore = gr.Button("Restore into Speak", variant="primary")
                            sess_status = gr.HTML(elem_classes="tts-status")

                    with gr.Column(scale=2, elem_classes="tts-col"):
                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("book", "Pronunciation lexicon",
                                           "One rule per line: <code>pattern -&gt; replacement</code>. Wrap "
                                           "the pattern in <code>/slashes/</code> for a regex. Applies from "
                                           "the next generation, no restart."))
                            lex_box = gr.Textbox(label="lexicon.txt", lines=14, max_lines=40,
                                                 value=do_load_lexicon(), interactive=True,
                                                 elem_classes="tts-mono")
                            with gr.Row(elem_classes="tts-actions"):
                                lex_reload = gr.Button("Reload from disk", variant="secondary")
                                lex_save = gr.Button("Save lexicon", variant="primary")
                            lex_status = gr.HTML(elem_classes="tts-status")

                        with gr.Column(elem_classes="tts-card"):
                            gr.HTML(_title("broom", "Housekeeping",
                                           "A stopped or crashed run leaves per-line audio behind. This removes "
                                           "only files older than 24 hours that no restorable take uses."))
                            with gr.Row(elem_classes="tts-actions"):
                                clean_dry = gr.Button("Preview cleanup", variant="secondary")
                                clean_btn = gr.Button("Clean up now", variant="stop")
                            clean_status = gr.HTML(elem_classes="tts-status")

                sys_btn.click(do_system_detail_refresh, outputs=[sys_detail]).then(
                    do_system_refresh, outputs=[sysbar])
                speed_btn.click(do_reset_speed, outputs=[speed_status]).then(
                    do_system_detail_refresh, outputs=[sys_detail]).then(
                    do_system_refresh, outputs=[sysbar])
                sess_refresh.click(do_refresh_sessions, outputs=[sess_list])
                sess_restore.click(do_restore_session, inputs=[sess_list],
                                   outputs=[speak_state, redo_line, sess_status, sess_list])
                lex_reload.click(do_load_lexicon, outputs=[lex_box])
                lex_save.click(do_save_lexicon, inputs=[lex_box], outputs=[lex_status])
                clean_dry.click(lambda: do_cleanup(dry_run=True), outputs=[clean_status])
                clean_btn.click(lambda: do_cleanup(dry_run=False), outputs=[clean_status]).then(
                    do_refresh_sessions, outputs=[sess_list])

        gr.HTML(FOOTER_HTML)

        refresh_btn.click(refresh_voice_lists, outputs=[voice, v_pick, gallery])
        demo.load(do_system_refresh, outputs=[sysbar])
        # Hidden tabs mount lazily in Gradio 6: an event that targets one of
        # their components before it exists leaves it stuck on "pending". So
        # each tab fills itself when it is opened, which also keeps it fresh.
        voices_tab.select(refresh_voice_lists, outputs=[voice, v_pick, gallery]).then(
            do_pick_voice, inputs=[v_pick], outputs=[v_audio, v_card])
        # Refreshing sets the dropdown, and its .change loads the player; loading
        # here too raced it (two fetches of the same file, the first a 503).
        rec_tab.select(do_refresh_recordings, outputs=[rec_list])
        settings_tab.select(do_refresh_sessions, outputs=[sess_list]).then(
            do_system_detail_refresh, outputs=[sys_detail])

    return demo


THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        name="coral", c50="#fff3f1", c100="#ffe3de", c200="#ffc6bc", c300="#ffa192",
        c400="#ff7d6b", c500="#ff6b5a", c600="#e8503f", c700="#c23d2f", c800="#9c3026",
        c900="#7a261f", c950="#4a1611"),
    secondary_hue="amber",
    neutral_hue=gr.themes.Color(
        name="ink", c50="#f6f3ef", c100="#eee8e1", c200="#e4ddd4", c300="#c9c3bb",
        c400="#8d97ad", c500="#5d6578", c600="#3a445c", c700="#253048", c800="#151c2d",
        c900="#111725", c950="#0a0e18"),
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "Segoe UI", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "ui-monospace", "monospace"],
    radius_size="lg",
).set(
    # Every built-in component aligned with the brand tokens in ui/studio.css and
    # checked for WCAG AA: text >= 4.5:1, control boundaries >= 3:1. Light first,
    # then the _dark variant. Gradio has no _dark for color_accent or the shadows;
    # studio.css sets --color-accent per theme instead.
    body_background_fill="#f6f3ef", body_background_fill_dark="#0a0e18",
    body_text_color="#171a26", body_text_color_dark="#eef1f8",
    body_text_color_subdued="#5d6578", body_text_color_subdued_dark="#8d97ad",
    background_fill_primary="#ffffff", background_fill_primary_dark="#0a0e18",
    background_fill_secondary="#f6f3ef", background_fill_secondary_dark="#111725",
    panel_background_fill="#f8f5f1", panel_background_fill_dark="#151c2d",
    panel_border_color="#e4ddd4", panel_border_color_dark="#253048",
    border_color_primary="#e4ddd4", border_color_primary_dark="#253048",
    border_color_accent="#bf3626", border_color_accent_dark="#ff8a78",
    color_accent="#bf3626", color_accent_soft="#fdeee9", color_accent_soft_dark="#2a1a1d",
    block_background_fill="#ffffff", block_background_fill_dark="#111725",
    block_border_color="#e4ddd4", block_border_color_dark="#253048",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_text_color="#555d70", block_label_text_color_dark="#a7b0c4",
    block_title_text_color="#171a26", block_title_text_color_dark="#eef1f8",
    block_info_text_color="#5d6578", block_info_text_color_dark="#8d97ad",
    accordion_text_color="#171a26", accordion_text_color_dark="#eef1f8",
    input_background_fill="#f8f5f1", input_background_fill_dark="#0a0e18",
    input_background_fill_focus="#ffffff", input_background_fill_focus_dark="#0a0e18",
    input_background_fill_hover="#f8f5f1", input_background_fill_hover_dark="#0a0e18",
    input_border_width="1px", input_border_width_dark="1px",
    input_border_color="#8c8273", input_border_color_dark="#6b7591",
    input_border_color_hover="#6f6658", input_border_color_hover_dark="#8d97ad",
    input_border_color_focus="#bf3626", input_border_color_focus_dark="#ff8a78",
    input_placeholder_color="#616980", input_placeholder_color_dark="#8d97ad",
    input_shadow="0 1px 2px rgba(23,26,38,.05)", input_shadow_dark="0 1px 2px rgba(0,0,0,.45)",
    input_shadow_focus="0 0 0 3px rgba(191,54,38,.18)", input_shadow_focus_dark="0 0 0 3px rgba(255,138,120,.22)",
    checkbox_background_color="#ffffff", checkbox_background_color_dark="#151c2d",
    checkbox_background_color_hover="#f8f5f1", checkbox_background_color_hover_dark="#1c2539",
    checkbox_background_color_focus="#ffffff", checkbox_background_color_focus_dark="#151c2d",
    checkbox_background_color_selected="#bf3626", checkbox_background_color_selected_dark="#ff6b5a",
    checkbox_border_width="1px", checkbox_border_width_dark="1px",
    checkbox_border_color="#8c8273", checkbox_border_color_dark="#6b7591",
    checkbox_border_color_hover="#6f6658", checkbox_border_color_hover_dark="#8d97ad",
    checkbox_border_color_focus="#bf3626", checkbox_border_color_focus_dark="#ff8a78",
    checkbox_border_color_selected="#bf3626", checkbox_border_color_selected_dark="#ff6b5a",
    checkbox_label_background_fill="#ffffff", checkbox_label_background_fill_dark="#151c2d",
    checkbox_label_background_fill_hover="#f8f5f1", checkbox_label_background_fill_hover_dark="#1c2539",
    checkbox_label_text_color="#171a26", checkbox_label_text_color_dark="#eef1f8",
    slider_color="#bf3626", slider_color_dark="#ff8a78",
    loader_color="#bf3626", loader_color_dark="#ff8a78",
    link_text_color="#bf3626", link_text_color_dark="#ff8a78",
    link_text_color_hover="#a02a21", link_text_color_hover_dark="#ffa192",
    link_text_color_visited="#a02a21", link_text_color_visited_dark="#ffa192",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#111725",
    button_secondary_background_fill_hover="#fdeee9", button_secondary_background_fill_hover_dark="#2a1a1d",
    button_secondary_border_color="#8c8273", button_secondary_border_color_dark="#6b7591",
    button_secondary_border_color_hover="#bf3626", button_secondary_border_color_hover_dark="#ff8a78",
    button_secondary_text_color="#171a26", button_secondary_text_color_dark="#eef1f8",
    button_secondary_text_color_hover="#bf3626", button_secondary_text_color_hover_dark="#ff8a78",
    button_cancel_background_fill="#ffffff", button_cancel_background_fill_dark="#111725",
    button_cancel_background_fill_hover="#fdeeec", button_cancel_background_fill_hover_dark="#2e1816",
    button_cancel_border_color="#a02a21", button_cancel_border_color_dark="#ff9f96",
    button_cancel_text_color="#a02a21", button_cancel_text_color_dark="#ff9f96",
    button_cancel_text_color_hover="#a02a21", button_cancel_text_color_hover_dark="#ff9f96",
    table_border_color="#e4ddd4", table_border_color_dark="#253048",
    table_even_background_fill="#ffffff", table_even_background_fill_dark="#111725",
    table_odd_background_fill="#f8f5f1", table_odd_background_fill_dark="#151c2d",
    table_row_focus="#fdeee9", table_row_focus_dark="#2a1a1d",
    code_background_fill="#efe9e2", code_background_fill_dark="#151c2d",
    error_background_fill="#fdeeec", error_background_fill_dark="#2e1816",
    error_border_color="#a02a21", error_border_color_dark="#ff9f96",
    error_text_color="#a02a21", error_text_color_dark="#ff9f96",
    shadow_drop="0 1px 2px rgba(23,26,38,.07)",
    shadow_drop_lg="0 1px 2px rgba(23,26,38,.05), 0 8px 24px rgba(23,26,38,.07)",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--offline", action="store_true",
                        help="Load from the local Hugging Face cache only (skip network checks)")
    parser.add_argument("--low-memory", action="store_true",
                        help="Load 0.6B as bfloat16. On a CPU without hardware bf16 this is far "
                             "SLOWER, not faster -- only use it when float32 will not fit.")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Lines generated per model call. 1 (default) = one at a time, lowest "
                             "peak RAM (needed on 8GB). 0 = all in one batch, faster on a big machine.")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch intra-op threads (default: torch's own choice)")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"],
                        default="auto",
                        help="Weight dtype. auto picks float32 unless the CPU has hardware "
                             "bfloat16 and RAM is tight (see common.resolve_dtype)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab on start")
    parser.add_argument("--mock", action="store_true",
                        help="Development aid: replace the model with a fast stand-in that returns tones, "
                             "so every screen can be exercised without loading a checkpoint")
    args = parser.parse_args()

    CFG.device = args.device
    CFG.offline = args.offline
    CFG.low_memory = args.low_memory
    CFG.chunk_size = args.chunk_size
    CFG.threads = args.threads
    CFG.dtype = args.dtype
    CFG.mock = args.mock
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Serving on http://{args.host}:{args.port}  (device: {args.device}"
          f"{', offline' if args.offline else ''}"
          f"{', low-memory' if (args.low_memory or auto_low_memory()) else ''}"
          f"{', chunked' if args.chunk_size == 1 else f', chunk={args.chunk_size}'}"
          f"{', MOCK MODEL' if args.mock else ''})",
          file=sys.stderr)
    print("First generation of each kind loads its model -- minutes on CPU. Be patient.",
          file=sys.stderr)

    build_app().launch(
        # Gradio 6 takes css/theme/js here, not in the Blocks constructor.
        css=CSS + _hero_css(),
        theme=THEME,
        head=f"<script>{STUDIO_JS}</script>",
        favicon_path=str(ASSETS_DIR / "favicon.svg"),
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        # output/, voices/ and samples/ live outside the launch dir, so the players can't
        # read a take or a reference clip without this.
        allowed_paths=[str(OUTPUT_DIR), str(VOICES_DIR), str(SAMPLES_DIR)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
