"""The parts of transcribe.py that don't need the recogniser: scoring, passage
matching, hotwords, and the delivery read on a signal with known answers."""
import numpy as np
import soundfile as sf

import transcribe as asr
from transcribe import Transcript, Word


def test_wer_ignores_number_formatting_and_punctuation():
    ref = "Access your money with 24 hours notice. Twenty four bit audio."
    hyp = "access your money with twenty-four hours' notice, 24-bit audio"
    assert asr.word_error_rate(ref, hyp) == 0.0


def test_transcripts_spell_numbers_but_keep_codes():
    import textprep

    out = textprep.spell_out_numbers("Save 50,000 rand on 24-bit audio, the 4th time. The new HTD-189 plays MP3.")
    assert "fifty thousand" in out and "twenty-four-bit" in out and "fourth" in out
    assert "HTD-189" in out and "MP3" in out


def test_wer_counts_real_errors():
    assert asr.word_error_rate("save fifty thousand rand", "win fifty thousand rand") == 0.25
    assert asr.word_error_rate("one two three four", "one three four") == 0.25


def test_match_passage_recognises_a_read_with_slips():
    passage = "Morning everyone! It's a lekker sunny day in Cape Town, and the wind is finally taking a break."
    heard = "morning everyone its a lekker sunny day in cape town and the wind is finally taking a break"
    assert asr.match_passage(heard, passage) > 0.9
    assert asr.match_passage("something else entirely was said here", passage) < 0.3


def test_lexicon_hotwords_takes_plain_patterns_only(tmp_path):
    lex = tmp_path / "lexicon.txt"
    lex.write_text("# comment\nGqeberha -> ge-kwe-ber-ha\n/\\bkm\\b/ -> kilometres\n"
                   "Nguyen   -> win   # trailing comment\n\n", encoding="utf-8")
    assert asr.lexicon_hotwords(lex) == "Gqeberha Nguyen"
    assert asr.lexicon_hotwords(tmp_path / "missing.txt") == ""


def _voiced(f0, seconds, sr=16000):
    """A buzzy, speech-like tone at `f0` Hz with a syllable-rate envelope."""
    t = np.arange(int(seconds * sr)) / sr
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3 * t) ** 2
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return (0.2 * env * sig).astype(np.float32)


def test_profile_reads_pitch_pace_pauses_and_fillers(tmp_path):
    sr = 16000
    clip = np.concatenate([_voiced(140, 3.0), np.zeros(int(0.6 * sr), np.float32), _voiced(140, 3.0)])
    path = tmp_path / "clip.wav"
    sf.write(path, clip, sr)

    # 16 words over ~6.6 s with one 0.6 s pause after a sentence, and one "um".
    words = [Word(f"w{i}", i * 0.37, i * 0.37 + 0.3, 0.95) for i in range(8)]
    words[-1].text = "end."
    words += [Word("um", 3.6, 3.8, 0.5)]
    words += [Word(f"v{i}", 3.9 + i * 0.37, 3.9 + i * 0.37 + 0.3, 0.9) for i in range(8)]
    t = Transcript(text="", words=words, language="en", language_prob=0.99, duration=6.6, model="test")

    p = asr.profile(str(path), t)
    assert p.pitch_hz is not None and abs(p.pitch_hz - 140) < 8
    assert p.register == "medium"
    assert p.expressiveness == "even and steady" and p.delivery == "Steady"   # a flat tone
    assert p.fillers == 1
    assert p.pauses >= 1 and p.longest_pause >= 0.3
    assert 120 < p.wpm < 175
    assert p.contour and any(v for v in p.contour)
    meta = p.to_meta()
    assert "contour" not in meta and meta["delivery"] == "Steady"


def test_profile_without_transcript_still_reads_pitch(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, _voiced(220, 4.0), 16000)
    p = asr.profile(str(path))
    assert p.register == "high" and p.wpm is None


def test_offline_without_model_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "MODEL_DIR", tmp_path)
    try:
        asr.model_path("small", offline=True)
    except asr.TranscriberUnavailable as e:
        assert "--offline" in str(e)
    else:
        raise AssertionError("expected TranscriberUnavailable")
