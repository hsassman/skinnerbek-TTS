"""One web app for every Qwen3-TTS capability on this machine.

`launch_ui.py` runs the demo that ships with `qwen_tts`: one capability per
process, a bilingual UI, no notion of a script or a saved-voice library. This
app is the friendlier front door:

    * Speak       -- pick a voice, type a script, watch it render line by line,
                     redo any line you don't like.
    * Describe    -- invent a voice from a written description (VoiceDesign).
    * Voices      -- turn a recording into a reusable named voice.
    * Recordings  -- play back and download anything generated earlier.
    * Settings    -- machine report, pronunciation lexicon, housekeeping.

Every generation runs through the *same* helpers the command line uses
(`common.synth` over `run_preset` / `run_design` / `run_clone`, then
`write_wavs` / `stitch`), so a result from this app matches the equivalent
`tts.py` command at the same `--chunk-size`.

    python scripts/app.py                # http://127.0.0.1:7860
    python scripts/app.py --offline      # skip Hugging Face network checks
    python scripts/app.py --port 8000 --share

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
from voice_library import list_voices, load_voice, save_voice, voice_meta, voice_path  # noqa: E402

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
                      threads=None, dtype="auto")

# One loaded model at a time (8 GB box). Keyed by (kind, size, dtype_name).
_CACHE: dict = {}

# Cancellation. Generation is strictly serial here -- only one model fits in RAM
# -- so a single process-wide flag is the whole story, and it saves the Cancel
# button from needing per-session plumbing.
_CANCEL = threading.Event()


# ---------------------------------------------------------------------------
# Design system
#
# Tokens first, components second, so light/dark and density change in one
# place. Colours are chosen for WCAG AA against their own surface, and every
# status is carried by an icon and a word as well as a hue, so nothing depends
# on colour vision alone.
# ---------------------------------------------------------------------------
CSS = """
/* ===========================================================================
   Design tokens
   Light is the default on :root. Dark mode follows GRADIO, not the OS: Gradio 6
   toggles a `.dark` class on an ancestor and ships no prefers-color-scheme
   rules at all, so keying these off the media query left every panel on its
   light surface while Gradio's own components went dark. Custom properties
   inherit, so redefining them under `.dark` re-themes everything below it.
   =========================================================================== */
:root {
  --tts-surface:     #ffffff;
  --tts-surface-2:   #f4f6f9;
  --tts-surface-3:   #eaeef3;
  --tts-border:      #dde2e9;
  --tts-border-soft: #eaeef3;
  --tts-text:        #12151a;
  --tts-muted:       #5b6573;
  /* Used on .tts-sec headings (0.74rem bold) and .tts-line--todo, neither of
     which is WCAG "large text", so this needs 4.5:1 not 3:1. */
  --tts-faint:       #606a79;
  --tts-accent:      #2f5bd0;
  --tts-accent-weak: #eaf0fd;
  --tts-ok:          #10673b;
  --tts-ok-weak:     #e4f5ec;
  --tts-warn:        #85500b;
  --tts-warn-weak:   #fdf1de;
  --tts-err:         #a02a21;
  --tts-err-weak:    #fdeeec;

  --tts-r:    14px;
  --tts-r-sm: 9px;
  --tts-r-xs: 6px;

  --tts-s1: 6px;
  --tts-s2: 10px;
  --tts-s3: 14px;
  --tts-s4: 20px;
  --tts-s5: 28px;

  --tts-shadow:    0 1px 2px rgba(14,18,26,.05), 0 6px 18px rgba(14,18,26,.06);
  --tts-shadow-sm: 0 1px 2px rgba(14,18,26,.07);

  --tts-ease: cubic-bezier(.22,.61,.36,1);
  --tts-dur:  .26s;
}
.dark {
  --tts-surface:     #161a20;
  --tts-surface-2:   #1c222a;
  --tts-border:      #2b323c;
  --tts-surface-3:   #232a34;
  --tts-border-soft: #232a34;
  --tts-text:        #eef1f6;
  --tts-muted:       #a4aebc;
  --tts-faint:       #87919e;   /* 4.5:1 on every dark surface, see light note */
  --tts-accent:      #8fb0ff;
  --tts-accent-weak: #1a2740;
  --tts-ok:          #6ed49b;
  --tts-ok-weak:     #10281d;
  --tts-warn:        #f0bd6c;
  --tts-warn-weak:   #2a2013;
  --tts-err:         #ff9f96;
  --tts-err-weak:    #2e1816;

  --tts-shadow:    0 1px 2px rgba(0,0,0,.45), 0 6px 18px rgba(0,0,0,.34);
  --tts-shadow-sm: 0 1px 2px rgba(0,0,0,.5);
}

/* ===========================================================================
   Motion. One vocabulary, used everywhere, and every bit of it is switched off
   under prefers-reduced-motion at the bottom of this file.
   =========================================================================== */
@keyframes tts-rise   { from { opacity:0; transform: translateY(8px);  } to { opacity:1; transform:none; } }
@keyframes tts-fade   { from { opacity:0; } to { opacity:1; } }
@keyframes tts-slide  { from { opacity:0; transform: translateX(-6px); } to { opacity:1; transform:none; } }
@keyframes tts-pop    { 0% { transform: scale(.94); opacity:0; } 60% { transform: scale(1.02); } 100% { transform:none; opacity:1; } }
@keyframes tts-spin   { to { transform: rotate(360deg); } }
@keyframes tts-shimmer{ 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
@keyframes tts-breathe{ 0%,100% { opacity:1; } 50% { opacity:.55; } }

.gradio-container { max-width: 1320px !important; }

/* ===========================================================================
   Masthead
   =========================================================================== */
.gradio-container .tts-head {
  position: relative; overflow: hidden;
  display: flex; flex-direction: column; gap: var(--tts-s3);
  background: var(--tts-surface);
  border: 1px solid var(--tts-border);
  border-radius: var(--tts-r);
  box-shadow: var(--tts-shadow);
  /* Full-bleed: pull into the app's side padding by exactly the horizontal
     inset so the h1, the tagline and the chips all start on the same left
     edge as every tab's first section heading. */
  padding: var(--tts-s4) var(--tts-s3);
  margin: 0 calc(-1 * var(--tts-s3)) var(--tts-s4);
  animation: tts-rise .4s var(--tts-ease) both;
}
.tts-head::after {
  content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
  background: linear-gradient(90deg, var(--tts-accent), transparent 70%);
  opacity: .55;
}
.tts-head > * { width: 100%; }
.tts-head__lead h1 {
  font-size: 1.5rem; font-weight: 660; letter-spacing: -.02em;
  margin: 0 0 5px; color: var(--tts-text); line-height: 1.2;
}
.tts-head__lead p { margin: 0; color: var(--tts-muted); font-size: .93rem; line-height: 1.55; max-width: 68ch; }
.tts-head .tts-chips { margin-top: 0; }
.tts-head .tts-chips + .tts-note { margin-top: var(--tts-s2); }

/* ===========================================================================
   Status chips
   =========================================================================== */
.tts-chips { display: flex; flex-wrap: wrap; gap: var(--tts-s1); margin-top: var(--tts-s3); }
.tts-chip {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 5px 11px; border-radius: 999px;
  font-size: .79rem; font-weight: 500; line-height: 1.5;
  border: 1px solid var(--tts-border);
  background: var(--tts-surface-2); color: var(--tts-text);
  transition: transform var(--tts-dur) var(--tts-ease), box-shadow var(--tts-dur) var(--tts-ease);
  animation: tts-pop .34s var(--tts-ease) both;
}
.tts-chips .tts-chip:nth-child(1) { animation-delay: .02s; }
.tts-chips .tts-chip:nth-child(2) { animation-delay: .06s; }
.tts-chips .tts-chip:nth-child(3) { animation-delay: .10s; }
.tts-chips .tts-chip:nth-child(4) { animation-delay: .14s; }
.tts-chips .tts-chip:nth-child(5) { animation-delay: .18s; }
.tts-chips .tts-chip:nth-child(6) { animation-delay: .22s; }
.tts-chip:hover { transform: translateY(-1px); box-shadow: var(--tts-shadow-sm); }
.tts-chip b { font-weight: 640; }
.tts-chip .k { color: var(--tts-muted); font-weight: 500; }
.tts-chip--ok   { background: var(--tts-ok-weak);   border-color: transparent; color: var(--tts-ok); }
.tts-chip--warn { background: var(--tts-warn-weak); border-color: transparent; color: var(--tts-warn); }
.tts-chip--err  { background: var(--tts-err-weak);  border-color: transparent; color: var(--tts-err); }
.tts-chip--err  { animation: tts-pop .34s var(--tts-ease) both, tts-breathe 2.6s ease-in-out 1s 3; }

/* ===========================================================================
   Callouts
   =========================================================================== */
.tts-note {
  border: 1px solid var(--tts-border);
  background: var(--tts-surface-2);
  border-radius: var(--tts-r-sm);
  padding: 11px 14px;
  font-size: .865rem; line-height: 1.6; color: var(--tts-text);
  animation: tts-rise .3s var(--tts-ease) both;
}
.tts-note + .tts-note { margin-top: var(--tts-s2); }
.tts-note--warn { border-color: var(--tts-warn); background: var(--tts-warn-weak); }
.tts-note--err  { border-color: var(--tts-err);  background: var(--tts-err-weak); }
.tts-note--ok   { border-color: var(--tts-ok);   background: var(--tts-ok-weak); }
.tts-note strong { font-weight: 640; }
.tts-note code {
  background: var(--tts-surface-3); padding: 1.5px 5px;
  border-radius: var(--tts-r-xs); font-size: .9em; overflow-wrap: anywhere;
}
.tts-note ul { margin: 8px 0 0; padding-left: 18px; }
.tts-note li { margin: 2px 0; }

/* ===========================================================================
   Section headings. One rhythm for every tab.
   =========================================================================== */
.tts-sec {
  display: flex; align-items: center; gap: var(--tts-s2);
  margin: var(--tts-s5) 0 var(--tts-s2);
  animation: tts-fade .3s var(--tts-ease) both;
}
.tts-col > *:first-child .tts-sec { margin-top: 0; }
.tts-sec h2, .tts-sec h3 {
  margin: 0; font-size: .74rem; font-weight: 680;
  text-transform: uppercase; letter-spacing: .075em;
  color: var(--tts-faint); white-space: nowrap;
}
.tts-sec::after {
  content: ""; flex: 1 1 auto; height: 1px; background: var(--tts-border-soft);
}

/* ===========================================================================
   Render progress
   =========================================================================== */
.tts-prog {
  border: 1px solid var(--tts-border);
  border-radius: var(--tts-r);
  background: var(--tts-surface);
  box-shadow: var(--tts-shadow-sm);
  padding: var(--tts-s3) var(--tts-s3) var(--tts-s3);
  animation: tts-rise .3s var(--tts-ease) both;
}
.tts-prog__top {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: var(--tts-s2); flex-wrap: wrap; margin-bottom: var(--tts-s2);
}
.tts-prog__title { font-weight: 645; font-size: .95rem; color: var(--tts-text); letter-spacing: -.01em; }
.tts-prog__meta  { font-size: .8rem; color: var(--tts-muted); font-variant-numeric: tabular-nums; }
.tts-bar {
  position: relative; height: 8px; border-radius: 999px; overflow: hidden;
  background: var(--tts-surface-3);
}
.tts-bar__fill {
  height: 100%; border-radius: 999px; background: var(--tts-accent);
  transition: width .5s var(--tts-ease), background-color var(--tts-dur) var(--tts-ease);
  position: relative; overflow: hidden;
}
.tts-bar__fill::after {
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.42), transparent);
  background-size: 200% 100%;
  animation: tts-shimmer 1.6s linear infinite;
}
.tts-bar__fill--done { background: var(--tts-ok); }
.tts-bar__fill--err  { background: var(--tts-err); }
.tts-bar__fill--done::after, .tts-bar__fill--err::after { display: none; }

.tts-lines { list-style: none; margin: var(--tts-s3) 0 0; padding: 0; display: grid; gap: 3px; }
.tts-line {
  display: grid; grid-template-columns: 18px 1fr auto; align-items: baseline;
  gap: var(--tts-s2); padding: 7px 9px; border-radius: var(--tts-r-sm);
  font-size: .845rem; line-height: 1.5; color: var(--tts-text);
  border: 1px solid transparent;
  transition: background-color var(--tts-dur) var(--tts-ease), border-color var(--tts-dur) var(--tts-ease);
  animation: tts-slide .3s var(--tts-ease) both;
}
.tts-line--done { background: var(--tts-surface-2); }
.tts-line--now  { background: var(--tts-accent-weak); border-color: var(--tts-accent); }
.tts-line--todo { color: var(--tts-faint); }
.tts-line__ico  { font-size: .88rem; text-align: center; line-height: 1.5; }
.tts-line--done .tts-line__ico { color: var(--tts-ok); }
.tts-line--now  .tts-line__ico { color: var(--tts-accent); }
.tts-line__txt  { overflow-wrap: anywhere; }
.tts-line__t    { font-size: .76rem; color: var(--tts-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.tts-spin { display: inline-block; animation: tts-spin 1.1s linear infinite; }

/* ===========================================================================
   Prepared-text preview
   =========================================================================== */
.tts-prev { display: grid; gap: 4px; margin-top: var(--tts-s2); }
.tts-prev__row {
  display: grid; grid-template-columns: 24px 1fr; gap: var(--tts-s2);
  padding: 8px 10px; border-radius: var(--tts-r-sm);
  background: var(--tts-surface-2); border: 1px solid var(--tts-border-soft);
  font-size: .845rem; line-height: 1.55; color: var(--tts-text);
  animation: tts-slide .28s var(--tts-ease) both;
  transition: border-color var(--tts-dur) var(--tts-ease);
}
.tts-prev__row:hover { border-color: var(--tts-border); }
.tts-prev__n { color: var(--tts-faint); font-variant-numeric: tabular-nums; text-align: right; }
.tts-prev__gap { color: var(--tts-faint); font-size: .77rem; }

/* ===========================================================================
   Key/value table + empty state
   =========================================================================== */
.tts-empty {
  padding: 26px 18px; text-align: center; color: var(--tts-muted);
  font-size: .87rem; line-height: 1.6;
  border: 1px dashed var(--tts-border);
  border-radius: var(--tts-r); background: var(--tts-surface-2);
  animation: tts-fade .3s var(--tts-ease) both;
}
.tts-empty b { display: block; color: var(--tts-text); font-weight: 620; margin-bottom: 3px; }
.tts-kv {
  width: 100%; border-collapse: collapse; font-size: .855rem;
  border: 1px solid var(--tts-border); border-radius: var(--tts-r-sm);
  overflow: hidden; background: var(--tts-surface-2);
}
.tts-kv td { padding: 8px 11px; border-bottom: 1px solid var(--tts-border-soft); color: var(--tts-text); }
.tts-kv td:first-child { color: var(--tts-muted); width: 46%; }
.tts-kv tr:last-child td { border-bottom: 0; }
.tts-kv tr { transition: background-color var(--tts-dur) var(--tts-ease); }
.tts-kv tr:hover { background: var(--tts-surface-3); }

/* ===========================================================================
   Layout rhythm
   Three classes carry every tab's spacing so the tabs cannot drift apart:
   .tts-split  the two-column shell (work column | render column)
   .tts-col    a stacked column of blocks
   .tts-actions a row of buttons
   Gradio lays Rows and Columns out with flexbox `gap`, so overriding gap on
   the element our elem_classes lands on is enough to re-space a whole tab.
   =========================================================================== */
.gradio-container .tts-split { gap: var(--tts-s5) !important; align-items: flex-start !important; }
.gradio-container .tts-col   { gap: var(--tts-s3) !important; }

/* Buttons in a row: one height, one radius, bottom-aligned so a button next to
   a labelled input lines up with that input's box rather than with its label. */
.gradio-container .tts-actions {
  gap: var(--tts-s2) !important;
  align-items: flex-end !important;
  flex-wrap: wrap !important;
  margin: var(--tts-s1) 0 var(--tts-s1) !important;
}
.gradio-container .tts-actions > button {
  min-height: 42px !important; border-radius: var(--tts-r-sm) !important;
  font-weight: 560 !important; letter-spacing: -.005em;
  transition: transform .12s var(--tts-ease), box-shadow var(--tts-dur) var(--tts-ease),
              filter var(--tts-dur) var(--tts-ease) !important;
}
.gradio-container button:not(:disabled):hover { filter: brightness(1.04); }
.gradio-container button:not(:disabled):active { transform: translateY(1px); }

.gradio-container .tabitem { animation: tts-rise .32s var(--tts-ease) both; padding-top: var(--tts-s4) !important; }
.gradio-container .block { border-radius: var(--tts-r-sm) !important; }
.gradio-container label > span { font-weight: 560 !important; }

/* Accordions are section dividers, not raised cards, and they need the same
   air above them that a section heading gets. */
.gradio-container .tts-pane {
  border-radius: var(--tts-r) !important;
  margin-top: var(--tts-s2) !important;
}
.gradio-container .tts-pane > .label-wrap { padding: var(--tts-s2) var(--tts-s3) !important; }

/* A lead note -- the one-line intro under a tab or a section heading. Same
   full-bleed trick as the masthead so its text sits on the section-heading
   grid instead of 14px inboard of it. */
.gradio-container .tts-lead {
  margin-left: calc(-1 * var(--tts-s3)) !important;
  margin-right: calc(-1 * var(--tts-s3)) !important;
  margin-bottom: var(--tts-s4) !important;
  padding-left: var(--tts-s3) !important;
  padding-right: var(--tts-s3) !important;
}
.tts-sec + .tts-lead { margin-top: var(--tts-s1) !important; }

/* Status lines sit under the control that produces them, not glued to it. */
.gradio-container .tts-status { margin-top: calc(var(--tts-s1) * -1) !important; }

/* ===========================================================================
   Accessibility
   =========================================================================== */
:where(button, a, input, textarea, select, summary, [tabindex]):focus-visible {
  outline: 3px solid var(--tts-accent) !important;
  outline-offset: 2px !important;
  border-radius: var(--tts-r-xs);
}
.tts-sr {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}

@media (prefers-reduced-motion: reduce) {
  .tts-head, .tts-chip, .tts-note, .tts-sec, .tts-prog, .tts-line,
  .tts-prev__row, .tts-empty, .gradio-container .tabitem {
    animation: none !important;
  }
  .tts-bar__fill, .tts-line, .tts-chip, .tts-prev__row, .tts-kv tr,
  .gradio-container .tts-actions > button { transition: none !important; }
  .tts-bar__fill::after, .tts-spin { animation: none !important; }
  .tts-bar__fill::after { display: none; }
  .gradio-container button:not(:disabled):active { transform: none; }
}
"""


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
def _chip(label: str, value: str, tone: str = "") -> str:
    cls = f"tts-chip tts-chip--{tone}" if tone else "tts-chip"
    return (f'<span class="{cls}"><span class="k">{escape(label)}</span>'
            f'<b>{escape(value)}</b></span>')


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
        f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in rows
    )
    return f'<table class="tts-kv">{body}</table>'


def _system_html() -> str:
    r = system_report()
    free, total = r.get("free_gb"), r.get("total_gb")
    dtype = r.get("dtype", "?")

    ram_tone = "err" if (free or 0) < 1.5 else "warn" if (free or 0) < 3.2 else "ok"
    chips = [
        _chip("CPU", f"{r.get('cores') or '?'} cores / {r.get('threads') or '?'} threads"),
        _chip("SIMD", str(r.get("simd", "?"))),
        _chip("torch threads", str(r.get("torch_threads", "?"))),
        _chip("RAM free",
              f"{free:.1f} of {total:.1f} GB" if free and total else "unknown",
              ram_tone),
        _chip("Precision", dtype, "ok" if dtype == "float32" else "err"),
        _chip("Speed", _speed_phrase(r.get("sec_per_frame"))),
    ]
    html = f'<div class="tts-chips" role="group" aria-label="Machine status">{"".join(chips)}</div>'

    # The one warning that actually costs hours if missed.
    if dtype != "float32" and not r.get("bf16_native"):
        need = max(0.0, 3.2 - (free or 0))
        html += (
            '<div class="tts-note tts-note--err" role="alert">'
            "<strong>Running on the slow path.</strong> This CPU has no hardware "
            "bfloat16, so torch emulates it &mdash; about <strong>25&times; slower "
            "per step</strong> than float32. Free "
            f"<strong>{need:.1f} GB</strong> more RAM (closing a browser usually does it) "
            "and float32 is picked automatically on the next generation.</div>"
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
    if seed is None or seed == "":
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
        meta.append(f"~{_fmt_dur(eta)} left")
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
            ico, cls, right = '<span class="tts-spin">&#9696;</span>', " tts-line--now", "rendering"
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
        f'<span class="tts-prog__title">{escape(title)}</span>'
        f'<span class="tts-prog__meta">{" &middot; ".join(meta)}</span>'
        f"</div>"
        f'<div class="tts-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{pct:.0f}" aria-label="Rendering progress">'
        f'<div class="tts-bar__fill{fill_cls}" style="width:{pct:.1f}%"></div></div>'
        f'<ol class="tts-lines">{"".join(rows)}</ol>'
        f"{note_html}</div>"
    )


def _idle_html(title: str = "Nothing rendering", msg: str = "") -> str:
    body = f"<b>{escape(title)}</b>{escape(msg)}" if msg else f"<b>{escape(title)}</b>"
    return f'<div class="tts-empty">{body}</div>'


def _sec(title: str, level: int = 2) -> str:
    """One section-heading rhythm for every tab: a small uppercase label with a
    rule running to the edge. Kept as a helper so no tab can drift."""
    return f'<div class="tts-sec"><h{level}>{escape(title)}</h{level}></div>'


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
        # Feed the measured rate back so future runs can be estimated up front.
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
        f'<span class="tts-prog__title">Loading the model'
        f'<span class="tts-spin" aria-hidden="true"> &#9696;</span></span>'
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


def do_add_voice(name, ref_audio, ref_text, xvec_only, size,
                 progress=gr.Progress(track_tqdm=True)):
    try:
        name = (name or "").strip()
        if not name:
            return "Give the voice a name.", gr.update()
        if not name.replace("_", "").replace("-", "").isalnum():
            return "Use only letters, numbers, - and _ in the name.", gr.update()
        if not ref_audio:
            return "Upload or record a reference clip first.", gr.update()
        if not xvec_only and not (ref_text or "").strip():
            return ("Type the transcript of the clip (word for word), or tick "
                    "'timbre only' to skip it (lower quality)."), gr.update()
        if voice_path(name).exists():
            return f"A voice called '{name}' already exists. Pick another name or delete it below.", gr.update()

        progress(0, desc="Loading the cloning model...")
        model = get_model("Base", size)
        progress(0.5, desc="Encoding the reference clip...")
        items = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=((ref_text or "").strip() or None),
            x_vector_only_mode=bool(xvec_only),
        )
        path = save_voice(name, items, ref_audio=str(ref_audio), model_size=size)
        return (f"Saved '{name}' -> `{path}`. It's now in the voice list on the Speak tab.",
                gr.update(choices=voice_choices()))
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return f"**Could not save the voice.**\n\n{type(e).__name__}: {e}", gr.update()


def do_delete_voice(name):
    try:
        if not name:
            return "Pick a voice to delete.", gr.update(), gr.update()
        p = voice_path(name)
        if not p.exists():
            return f"No saved voice called '{name}'.", gr.update(), gr.update()
        p.unlink()
        remaining = list_voices()
        return (f"Deleted '{name}'.",
                gr.update(choices=voice_choices()),
                gr.update(choices=remaining, value=(remaining[0] if remaining else None)))
    except Exception as e:  # noqa: BLE001
        return f"Could not delete: {type(e).__name__}: {e}", gr.update(), gr.update()


def refresh_voice_lists():
    names = list_voices()
    return (gr.update(choices=voice_choices()),
            gr.update(choices=names, value=(names[0] if names else None)))


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
    if not path or not Path(path).exists():
        return None, ""
    try:
        info = sf.info(str(path))
        meta = (f'<div class="tts-note"><strong>{escape(Path(path).name)}</strong><br>'
                f"{info.duration:.1f} s &middot; {info.samplerate} Hz &middot; {info.subtype}<br>"
                f"<code>{escape(str(path))}</code></div>")
        return str(path), meta
    except Exception as e:  # noqa: BLE001
        return None, f'<div class="tts-note tts-note--err">Could not read: {escape(str(e))}</div>'


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
DELIVERY_INFO = ("Steady is calmer and more consistent -- best on the small (0.6B) voice. "
                 "Expressive varies more but can wander.")
TIDY_INFO = ("Spell out numbers, money, dates and symbols; strip markdown; put each sentence "
             "on its own line and break very long ones. Turn off only if it mispronounces "
             "something. Pronunciation fixes in lexicon.txt always apply.")


def _fix_one_line_block():
    """The 'redo a single line' controls. Returns (dropdown, seed, button, status)."""
    with gr.Accordion("Fix one line", open=False, elem_classes="tts-pane"):
        gr.Markdown(
            "Redo just one sentence of the last take -- the rest is kept and the "
            "read is re-stitched. Change the seed for a different attempt."
        )
        line = gr.Dropdown(label="Line to redo", choices=[], visible=False)
        new_seed = gr.Number(label="New seed (optional)", precision=0)
        with gr.Row(elem_classes="tts-actions"):
            btn = gr.Button("Regenerate this line", variant="primary")
        status = gr.Markdown(elem_classes="tts-status")
    return line, new_seed, btn, status


def _render_column(label_prefix: str):
    """Progress -> result -> status -> prepared text -> repair. Returns the parts."""
    gr.HTML(_sec('Render'))
    progress = gr.HTML(_idle_html("Ready", " Press Speak and each line will appear here as it renders."))
    audio = gr.Audio(label=f"{label_prefix} result", type="filepath")
    status = gr.HTML()
    with gr.Accordion("Prepared text", open=False, elem_classes="tts-pane"):
        preview = gr.HTML(_idle_html(
            "No preview yet",
            " Press Preview text for the exact lines the model will be given, and a time estimate."))
    redo_line, redo_seed, redo_btn, redo_status = _fix_one_line_block()
    return progress, audio, status, preview, redo_line, redo_seed, redo_btn, redo_status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS Studio") as demo:
        with gr.Column(elem_classes="tts-head"):
            gr.HTML(
                '<div class="tts-head__lead">'
                "<h1>Qwen3-TTS Studio</h1>"
                "<p>Local text to speech. Nothing leaves this machine &mdash; the model runs on "
                "the CPU here, so a read takes minutes rather than seconds.</p>"
                "</div>"
            )
            sysbar = gr.HTML(_system_html())

        with gr.Tabs():
            # ================= Speak =================
            with gr.Tab("Speak"):
                speak_state = gr.State(None)
                gr.HTML(
                    '<div class="tts-note tts-lead">Turn a script into speech with a built-in '
                    "or a saved voice. Long text is split into one line per sentence "
                    "automatically; each line is rendered on its own and the takes are "
                    "stitched together.</div>"
                )
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        gr.HTML(_sec('Script'))
                        script = gr.Textbox(
                            label="What should be said",
                            lines=10,
                            placeholder=(
                                "Type or paste what the voice should say.\n\n"
                                "Long text is split into one sentence per line automatically; "
                                "you can also put each line where you want it. Blank lines start "
                                "a new paragraph; lines starting with # are skipped."
                            ),
                        )
                        gr.HTML(_sec('Voice'))
                        voice = gr.Dropdown(
                            label="Voice",
                            choices=voice_choices(),
                            value="preset:ryan",
                            info=PRESET_NOTE,
                        )
                        with gr.Row(elem_classes="tts-actions"):
                            speak_btn = gr.Button("Speak", variant="primary", scale=3)
                            stop_btn = gr.Button("Stop", variant="stop", scale=1)
                            preview_btn = gr.Button("Preview text", scale=1)
                            refresh_btn = gr.Button("Refresh voices", scale=1)

                        delivery = gr.Radio(
                            ["Steady", "Natural", "Expressive"], value="Steady",
                            label="Delivery", info=DELIVERY_INFO,
                        )
                        tidy = gr.Checkbox(value=True, label="Tidy the text before speaking",
                                           info=TIDY_INFO)

                        with gr.Accordion("Advanced settings", open=False, elem_classes="tts-pane"):
                            size = gr.Radio(
                                ["0.6B", "1.7B"], value="0.6B", label="Model size",
                                info="0.6B is faster and lighter. 1.7B is slower and needs "
                                     "~3.8 GB free RAM, but is needed for the style instruction "
                                     "below. Saved voices use whatever size they were made with.",
                            )
                            language = gr.Dropdown(LANGUAGES, value="Auto", label="Language",
                                                   info="Auto detects it from the text.")
                            instruct = gr.Textbox(
                                label="Style instruction (built-in voices, 1.7B only)",
                                placeholder="e.g. Speak slowly and warmly", lines=2,
                            )
                            gap_ms = gr.Slider(0, 1500, value=300, step=50,
                                               label="Gap between sentences (ms)")
                            para_gap = gr.Slider(0, 2000, value=650, step=50,
                                                 label="Gap between paragraphs (ms)",
                                                 info="A blank line in your script starts a new paragraph.")
                            seed = gr.Number(
                                label="Seed (optional)", precision=0,
                                info="Set a number to make a take repeatable. Leave blank for a "
                                     "fresh take each time.",
                            )
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        (s_prog, audio, status, preview_md,
                         redo_line, redo_seed, redo_btn, redo_status) = _render_column("Speak")

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

            # ================= Describe a voice =================
            with gr.Tab("Describe a voice"):
                d_state = gr.State(None)
                gr.HTML(
                    '<div class="tts-note tts-lead"><strong>Invent a voice from a description.</strong> '
                    "This uses the VoiceDesign model, which is published only at 1.7B &mdash; the "
                    "slowest to load and run, and it needs about 3.8&nbsp;GB of free RAM.</div>"
                )
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        gr.HTML(_sec('The voice'))
                        d_desc = gr.Textbox(
                            label="Describe the voice", lines=3,
                            placeholder="e.g. Warm older man, gentle and unhurried, slight gravel",
                        )
                        gr.HTML(_sec('Script'))
                        d_script = gr.Textbox(
                            label="What should be said", lines=8,
                            placeholder="What should this voice say? One sentence per line.",
                        )
                        with gr.Row(elem_classes="tts-actions"):
                            d_btn = gr.Button("Generate", variant="primary", scale=3)
                            d_stop = gr.Button("Stop", variant="stop", scale=1)
                            d_preview_btn = gr.Button("Preview text", scale=1)
                        d_delivery = gr.Radio(["Steady", "Natural", "Expressive"], value="Steady",
                                              label="Delivery", info=DELIVERY_INFO)
                        d_tidy = gr.Checkbox(value=True, label="Tidy the text before speaking",
                                             info=TIDY_INFO)
                        with gr.Accordion("Advanced settings", open=False, elem_classes="tts-pane"):
                            d_language = gr.Dropdown(LANGUAGES, value="Auto", label="Language")
                            d_gap = gr.Slider(0, 1500, value=300, step=50,
                                              label="Gap between sentences (ms)")
                            d_para_gap = gr.Slider(0, 2000, value=650, step=50,
                                                   label="Gap between paragraphs (ms)")
                            d_seed = gr.Number(label="Seed (optional)", precision=0)
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        (d_prog, d_audio, d_status, d_preview_md,
                         d_redo_line, d_redo_seed, d_redo_btn, d_redo_status) = _render_column("Design")

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

            # ================= Voices =================
            with gr.Tab("Voices"):
                gr.HTML(
                    '<div class="tts-note tts-lead">Turn a real recording into a reusable voice. Do this '
                    "once per clip; afterwards the voice appears in the <strong>Speak</strong> "
                    "tab's voice list. Use recordings you have the right to use.</div>"
                )
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        gr.HTML(_sec('Add a voice'))
                        a_name = gr.Textbox(label="Name for this voice", placeholder="e.g. my_narrator")
                        a_audio = gr.Audio(
                            label="Reference recording (5-30 seconds of clear speech)",
                            type="filepath", sources=["upload", "microphone"],
                        )
                        a_text = gr.Textbox(
                            label="Transcript of the recording (word for word)", lines=3,
                            placeholder="Type exactly what is said in the clip.",
                        )
                        a_xvec = gr.Checkbox(
                            label="Timbre only -- skip the transcript (faster to set up, lower quality)",
                            value=False,
                        )
                        a_size = gr.Radio(
                            ["0.6B", "1.7B"], value="0.6B", label="Model size",
                            info="The voice is tied to the size that made it. 1.7B needs ~3.8 GB free RAM.",
                        )
                        with gr.Row(elem_classes="tts-actions"):
                            a_btn = gr.Button("Save this voice", variant="primary")
                        a_status = gr.Markdown(elem_classes="tts-status")
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        gr.HTML(_sec('Saved voices'))
                        d_list = gr.Dropdown(
                            label="Delete a voice", choices=list_voices(),
                            value=(list_voices()[0] if list_voices() else None),
                        )
                        with gr.Row(elem_classes="tts-actions"):
                            del_btn = gr.Button("Delete selected voice", variant="stop")
                        del_status = gr.Markdown(elem_classes="tts-status")

                a_btn.click(do_add_voice, inputs=[a_name, a_audio, a_text, a_xvec, a_size],
                            outputs=[a_status, voice]).then(refresh_voice_lists,
                                                            outputs=[voice, d_list])
                del_btn.click(do_delete_voice, inputs=[d_list],
                              outputs=[del_status, voice, d_list])

            # ================= Recordings =================
            with gr.Tab("Recordings"):
                gr.HTML(
                    '<div class="tts-note tts-lead">Everything generated so far, newest first. Pick one '
                    "to play it, or use the player's download button to save a copy.</div>"
                )
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(scale=3, elem_classes="tts-col"):
                        gr.HTML(_sec('Library'))
                        rec_list = gr.Dropdown(label="Recording", choices=_recording_choices(),
                                               value=None)
                        with gr.Row(elem_classes="tts-actions"):
                            rec_refresh = gr.Button("Refresh list")
                        rec_meta = gr.HTML()
                    with gr.Column(scale=2, elem_classes="tts-col"):
                        gr.HTML(_sec('Playback'))
                        rec_audio = gr.Audio(label="Selected recording", type="filepath")

                rec_list.change(do_pick_recording, inputs=[rec_list], outputs=[rec_audio, rec_meta])
                rec_refresh.click(do_refresh_recordings, outputs=[rec_list]).then(
                    do_pick_recording, inputs=[rec_list], outputs=[rec_audio, rec_meta])

            # ================= Settings =================
            with gr.Tab("Settings"):
                gr.HTML(
                    '<div class="tts-note tts-lead">The machine readout and speed '
                    "calibration, restoring an earlier take, the pronunciation lexicon, and "
                    "cleanup of leftover per-line audio.</div>"
                )
                with gr.Row(equal_height=False, elem_classes="tts-split"):
                    with gr.Column(elem_classes="tts-col"):
                        gr.HTML(_sec('This machine'))
                        sys_detail = gr.HTML(_system_detail_html())
                        with gr.Row(elem_classes="tts-actions"):
                            sys_btn = gr.Button("Re-check")
                            speed_btn = gr.Button("Reset speed calibration")
                        speed_status = gr.HTML(elem_classes="tts-status")

                        gr.HTML(_sec('Restore an earlier take')
                                + '<div class="tts-note tts-lead">A take is restorable while its per-line '
                                "audio is still on disk. Restoring one lets you redo a single "
                                "line of it from the Speak tab, even after a restart.</div>")
                        sess_list = gr.Dropdown(label="Saved takes", choices=_session_choices(),
                                                value=None)
                        with gr.Row(elem_classes="tts-actions"):
                            sess_refresh = gr.Button("Refresh")
                            sess_restore = gr.Button("Restore into Speak", variant="primary")
                        sess_status = gr.HTML(elem_classes="tts-status")

                    with gr.Column(elem_classes="tts-col"):
                        gr.HTML(_sec('Pronunciation lexicon')
                                + '<div class="tts-note tts-lead">One rule per line: '
                                "<code>pattern -&gt; replacement</code>. Wrap the pattern in "
                                "<code>/slashes/</code> for a regex. Applied before numbers and "
                                "abbreviations, and even when &ldquo;Tidy the text&rdquo; is off. "
                                "Saved rules apply to the next generation with no restart.</div>")
                        lex_box = gr.Textbox(label="lexicon.txt", lines=14,
                                             value=do_load_lexicon())
                        with gr.Row(elem_classes="tts-actions"):
                            lex_reload = gr.Button("Reload from disk")
                            lex_save = gr.Button("Save lexicon", variant="primary")
                        lex_status = gr.HTML(elem_classes="tts-status")

                        gr.HTML(_sec('Housekeeping')
                                + '<div class="tts-note tts-lead">A cancelled or crashed run leaves per-line '
                                "audio behind. This removes only files older than 24 hours that no "
                                "restorable take refers to.</div>")
                        with gr.Row(elem_classes="tts-actions"):
                            clean_dry = gr.Button("Preview cleanup")
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

        refresh_btn.click(lambda: gr.update(choices=voice_choices()), outputs=[voice])
        demo.load(do_system_refresh, outputs=[sysbar])
        # .then so the newest recording is actually loaded into the player, not
        # just selected in the dropdown.
        demo.load(do_refresh_recordings, outputs=[rec_list]).then(
            do_pick_recording, inputs=[rec_list], outputs=[rec_audio, rec_meta])
        demo.load(do_refresh_sessions, outputs=[sess_list])

    return demo


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
    args = parser.parse_args()

    CFG.device = args.device
    CFG.offline = args.offline
    CFG.low_memory = args.low_memory
    CFG.chunk_size = args.chunk_size
    CFG.threads = args.threads
    CFG.dtype = args.dtype
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Serving on http://{args.host}:{args.port}  (device: {args.device}"
          f"{', offline' if args.offline else ''}"
          f"{', low-memory' if (args.low_memory or auto_low_memory()) else ''}"
          f"{', chunked' if args.chunk_size == 1 else f', chunk={args.chunk_size}'})",
          file=sys.stderr)
    print("First generation of each kind loads its model -- minutes on CPU. Be patient.",
          file=sys.stderr)

    build_app().launch(
        # Gradio 6 takes css/theme here, not in the Blocks constructor. Soft gives
        # the base components rounded corners and real elevation; CSS above layers
        # this project's own tokens on top.
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        # output/ lives outside the launch dir, so the player can't read a take
        # without this.
        allowed_paths=[str(OUTPUT_DIR)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
