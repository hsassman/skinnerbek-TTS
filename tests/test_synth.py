"""synth() is the per-line generation loop that keeps peak RAM to one line on the
8GB box. A fake model records how it was called so we can pin the chunking without
loading a checkpoint.
"""
import numpy as np

from common import MAX_NEW_TOKENS_CAP, MIN_NEW_TOKENS, synth, token_budget


class FakeModel:
    def __init__(self):
        self.batch_sizes = []
        self.kwargs = []

    def _audio(self, text):
        n = len(text) if isinstance(text, list) else 1
        self.batch_sizes.append(n)
        return [np.zeros(100, dtype=np.float32) for _ in range(n)], 24000

    def generate_custom_voice(self, **kw):
        self.kwargs.append(kw)
        return self._audio(kw["text"])

    def generate_voice_design(self, **kw):
        self.kwargs.append(kw)
        return self._audio(kw["text"])

    def generate_voice_clone(self, **kw):
        self.kwargs.append(kw)
        return self._audio(kw["text"])


LINES = [f"line {i}" for i in range(5)]


def test_chunk_size_1_is_one_line_per_call():
    m = FakeModel()
    wavs, sr = synth("preset", m, LINES, language="Auto", sampling={},
                     speaker="ryan", chunk_size=1)
    assert m.batch_sizes == [1, 1, 1, 1, 1]
    assert len(wavs) == 5 and sr == 24000


def test_chunk_size_0_is_a_single_batch():
    m = FakeModel()
    wavs, _ = synth("preset", m, LINES, language="Auto", sampling={},
                    speaker="ryan", chunk_size=0)
    assert m.batch_sizes == [5]
    assert len(wavs) == 5


def test_chunk_size_2_batches_in_twos():
    m = FakeModel()
    synth("design", m, LINES, language="Auto", sampling={}, instruct="warm", chunk_size=2)
    assert m.batch_sizes == [2, 2, 1]


def test_on_line_and_on_progress_callbacks():
    m = FakeModel()
    seen_lines, seen_progress = [], []
    synth("clone", m, LINES, language="Auto", sampling={},
          voice_clone_prompt=["P"], chunk_size=2,
          on_line=lambda i, w, sr: seen_lines.append(i),
          on_progress=lambda done, total: seen_progress.append((done, total)))
    assert seen_lines == [0, 1, 2, 3, 4]
    assert seen_progress == [(2, 5), (4, 5), (5, 5)]


def test_output_count_matches_lines_regardless_of_chunking():
    for cs in (1, 2, 3, 0):
        m = FakeModel()
        wavs, _ = synth("preset", m, LINES, language="Auto", sampling={},
                        speaker="ryan", chunk_size=cs)
        assert len(wavs) == len(LINES)


def test_unknown_capability_raises():
    import pytest

    with pytest.raises(ValueError):
        synth("bogus", FakeModel(), LINES, language="Auto", sampling={})


# --- runaway guard -----------------------------------------------------------
# qwen_tts defaults max_new_tokens to 4096 codec frames (~5.7 min of audio at
# 12Hz) no matter how short the line is, so a line that never emits EOS runs for
# hours. synth must always pass a ceiling derived from the text.

def test_every_call_gets_a_token_ceiling_well_under_the_library_default():
    m = FakeModel()
    synth("preset", m, LINES, language="Auto", sampling={}, speaker="ryan", chunk_size=1)
    caps = [kw["max_new_tokens"] for kw in m.kwargs]
    assert len(caps) == 5
    assert all(0 < c < 4096 for c in caps)


def test_explicit_max_new_tokens_is_not_overridden():
    m = FakeModel()
    synth("preset", m, LINES, language="Auto", sampling={"max_new_tokens": 77},
          speaker="ryan", chunk_size=1)
    assert [kw["max_new_tokens"] for kw in m.kwargs] == [77] * 5


def test_budget_is_sized_from_the_longest_line_in_the_chunk():
    m = FakeModel()
    synth("preset", m, ["hi", "x" * 400], language="Auto", sampling={},
          speaker="ryan", chunk_size=0)
    assert m.kwargs[0]["max_new_tokens"] == token_budget("x" * 400)


def test_token_budget_scales_with_length_and_stays_bounded():
    assert token_budget("") == MIN_NEW_TOKENS
    assert token_budget("a" * 50) <= token_budget("a" * 500)
    assert token_budget("a" * 1_000_000) == MAX_NEW_TOKENS_CAP


def test_token_budget_leaves_generous_headroom_for_a_real_line():
    # A 107-char line measured at 9.28s of speech = ~111 frames at 12Hz. The
    # budget must comfortably clear that or it would truncate a good read.
    assert token_budget("x" * 107) > 111 * 2
