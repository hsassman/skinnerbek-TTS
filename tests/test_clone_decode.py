"""Cloned lines decode with a short tail of the reference, not the whole clip."""
from types import SimpleNamespace

import numpy as np

import common

PER_FRAME = 4


class _Tokenizer:
    def __init__(self):
        self.decoded = []

    def decode(self, encoded):
        self.decoded.append([len(e["audio_codes"]) for e in encoded])
        # Each frame becomes PER_FRAME samples carrying its own value.
        return [np.repeat(np.asarray(e["audio_codes"], dtype=np.float32), PER_FRAME)
                for e in encoded], 24000


def _model():
    return SimpleNamespace(model=SimpleNamespace(speech_tokenizer=_Tokenizer()))


def _prompt(ref_len):
    return [SimpleNamespace(ref_code=np.zeros((ref_len, 16)))]


def test_only_the_tail_of_the_reference_is_decoded():
    model, tok = _model(), None
    tok = model.model.speech_tokenizer
    codes = np.arange(100 + 30, dtype=np.float32)          # 100 reference frames, 30 new
    with common.short_reference_decode(model, _prompt(100), context=20):
        wavs, sr = tok.decode([{"audio_codes": codes}])
    assert tok.decoded == [[50]]                            # 20 of context + 30 new
    wav = wavs[0]
    assert len(wav) == 130 * PER_FRAME                      # the caller's cut still lands right
    assert np.array_equal(wav[100 * PER_FRAME:], np.repeat(codes[100:], PER_FRAME))
    assert "decode" not in vars(tok)                        # restored afterwards


def test_short_references_and_other_decodes_are_untouched():
    model = _model()
    tok = model.model.speech_tokenizer
    with common.short_reference_decode(model, _prompt(15), context=20):
        assert "decode" not in vars(tok)                    # reference shorter than the context
    with common.short_reference_decode(model, _prompt(100), context=0):
        assert "decode" not in vars(tok)                    # switched off
    with common.short_reference_decode(model, _prompt(100), context=20):
        tok.decode([{"audio_codes": np.zeros(40)}])         # not reference + new: passed through
    assert tok.decoded == [[40]]
    with common.short_reference_decode(SimpleNamespace(), _prompt(100)):
        pass                                                # a stand-in model has no tokenizer
