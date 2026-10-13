"""The run_preset/run_design/run_clone helpers are the single seam between the
CLI, the app and qwen_tts. A fake model records what it was called with so we can
pin the mapping without loading a checkpoint.
"""
import numpy as np

from common import run_clone, run_design, run_preset


class FakeModel:
    def __init__(self):
        self.calls = []

    def _fake_audio(self, text):
        n = len(text) if isinstance(text, list) else 1
        return [np.zeros(1000, dtype=np.float32) for _ in range(n)], 24000

    def generate_custom_voice(self, **kw):
        self.calls.append(("custom_voice", kw))
        return self._fake_audio(kw["text"])

    def generate_voice_design(self, **kw):
        self.calls.append(("voice_design", kw))
        return self._fake_audio(kw["text"])

    def generate_voice_clone(self, **kw):
        self.calls.append(("voice_clone", kw))
        return self._fake_audio(kw["text"])


def test_run_preset_passes_through():
    m = FakeModel()
    wavs, sr = run_preset(m, ["a", "b"], language="english", speaker="ryan",
                          instruct="", sampling={"temperature": 0.7})
    name, kw = m.calls[0]
    assert name == "custom_voice"
    assert kw["text"] == ["a", "b"]
    assert kw["language"] == "english"
    assert kw["speaker"] == "ryan"
    assert kw["instruct"] is None          # "" is coerced to None
    assert kw["temperature"] == 0.7
    assert len(wavs) == 2 and sr == 24000


def test_run_preset_keeps_real_instruct():
    m = FakeModel()
    run_preset(m, ["x"], language="Auto", speaker="vivian",
               instruct="Speak warmly", sampling={})
    assert m.calls[0][1]["instruct"] == "Speak warmly"


def test_run_design_passes_through():
    m = FakeModel()
    run_design(m, ["hello"], language="Auto", instruct="gravelly older man",
               sampling={"top_k": 20})
    name, kw = m.calls[0]
    assert name == "voice_design"
    assert kw["instruct"] == "gravelly older man"
    assert kw["top_k"] == 20


def test_run_clone_with_prompt_vs_ref_audio():
    m = FakeModel()
    run_clone(m, ["one"], language="Auto", voice_clone_prompt=["PROMPT"], sampling={})
    assert m.calls[-1][1]["voice_clone_prompt"] == ["PROMPT"]
    assert "ref_audio" not in m.calls[-1][1]

    run_clone(m, ["two"], language="Auto", ref_audio="clip.wav", ref_text="two",
              x_vector_only=False, sampling={})
    kw = m.calls[-1][1]
    assert kw["ref_audio"] == "clip.wav"
    assert kw["ref_text"] == "two"
    assert kw["x_vector_only_mode"] is False
