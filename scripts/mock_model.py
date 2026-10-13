"""A stand-in for Qwen3TTSModel, for working on the app without loading a model.

`python scripts/app.py --mock` swaps every model load for this. Each line comes
back as a short, audible tone pattern after a delay proportional to its length,
so progress, cancel, stitching, sessions, "Fix one line" and the voice library
can all be exercised in seconds on any machine. Nothing it produces is speech.
"""
import time
from typing import List

import numpy as np

SR = 24000
SECONDS_PER_CHAR = 0.03     # render delay, so the progress UI has something to show


class MockTTSModel:
    def __init__(self, kind: str, size: str):
        self.kind = kind
        self.size = size

    def _render(self, text) -> tuple:
        lines: List[str] = text if isinstance(text, list) else [text]
        out = []
        for i, line in enumerate(lines):
            time.sleep(min(4.0, 0.4 + SECONDS_PER_CHAR * len(line)))
            dur = max(0.6, 0.065 * len(line))
            t = np.arange(int(SR * dur)) / SR
            # A gentle two-tone "syllable" pattern, pitched per line so takes differ.
            f = 180 + 25 * (hash(line) % 7)
            env = np.clip(np.sin(np.pi * t / dur), 0, 1) * (0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t) ** 2)
            wav = 0.25 * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t))
            out.append(wav.astype(np.float32))
        return out, SR

    def generate_custom_voice(self, text=None, **_kw):
        return self._render(text)

    def generate_voice_design(self, text=None, **_kw):
        return self._render(text)

    def generate_voice_clone(self, text=None, **_kw):
        return self._render(text)

    def create_voice_clone_prompt(self, ref_audio=None, ref_text=None, x_vector_only_mode=False):
        import torch
        from qwen_tts import VoiceClonePromptItem

        time.sleep(1.5)
        return [VoiceClonePromptItem(
            ref_code=None if x_vector_only_mode else torch.zeros(12, 16, dtype=torch.long),
            ref_spk_embedding=torch.zeros(1024),
            x_vector_only_mode=bool(x_vector_only_mode),
            icl_mode=not x_vector_only_mode,
            ref_text=ref_text,
        )]
