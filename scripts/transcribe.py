"""Local reference-clip transcription, passage matching and delivery profiles.

Whisper loads lazily and runs under a lock. Prompts preserve disfluencies;
South African vocabulary and lexicon hotwords guide spelling. Uncertain words
are exposed for review. unload() frees the recogniser before TTS model loads.
"""
from __future__ import annotations

import difflib
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# "large-v3-turbo" is Whisper large-v3 with a 4-layer decoder: near large-v3
# accuracy at a fraction of the cost. int8 keeps it inside the 8 GB budget.
# Measured on this repo's four hand-transcribed voice clips -- see README.
DEFAULT_MODEL = "large-v3-turbo"
FAST_MODEL = "small"

# Words a generic recogniser mangles in a Cape Town studio. Spelt as users write them.
SA_HOTWORDS = (
    "lekker howzit koeksister Bo-Kaap bru boet ouma oupa braai bakkie robot eish ja "
    "Constantia Cape Flats Simon's Town Gqeberha Stellenbosch Khayelitsha Muizenberg "
    "Woodstock rand Mzansi shame sharp now-now just-now"
)

# Priming text: Whisper copies the style of the prompt, so a prompt with
# disfluencies keeps them instead of tidying them away.
_VERBATIM_PROMPT = "Umm, so, uh, I was like, hmm, you know... Okay. Right, ja."
_CLEAN_PROMPT = "Hello, welcome. It's a lekker day in Cape Town."

FILLERS = {"um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm"}

# A word below this confidence is highlighted for the user to check.
UNSURE_BELOW = 0.6

# Plain folders, not the Hugging Face cache: the cache symlinks blobs, which
# Windows refuses without Developer Mode (WinError 1314). Outside the repo, so
# the weights can never be committed.
MODEL_DIR = Path.home() / ".cache" / "skinnerbek" / "whisper"


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    text: str
    words: List[Word]
    language: str
    language_prob: float
    duration: float
    model: str

    @property
    def unsure(self) -> List[Word]:
        return [w for w in self.words if w.prob < UNSURE_BELOW]

    @property
    def confidence(self) -> float:
        """Duration-weighted mean word probability, 0..1."""
        if not self.words:
            return 0.0
        dur = np.array([max(0.02, w.end - w.start) for w in self.words])
        return float(np.average([w.prob for w in self.words], weights=dur))


@dataclass
class VoiceProfile:
    """Plain-language facts about how someone speaks, plus the raw numbers."""
    pitch_hz: Optional[float] = None       # median fundamental frequency
    pitch_lo: Optional[float] = None       # 10th percentile
    pitch_hi: Optional[float] = None       # 90th percentile
    pitch_range_st: Optional[float] = None # 10th-90th percentile span, semitones (expressiveness)
    wpm: Optional[float] = None            # words per minute while speaking
    pauses: int = 0                        # gaps >= PAUSE_S between words
    pause_ms: Optional[int] = None         # median pause length
    longest_pause: float = 0.0
    loudness_var_db: Optional[float] = None
    fillers: int = 0
    clarity: Optional[float] = None        # recogniser confidence, 0..1
    register: str = ""
    pace: str = ""
    expressiveness: str = ""
    delivery: str = "Natural"              # suggested delivery preset
    gap_ms: int = 300                      # suggested pause between sentences
    notes: List[str] = field(default_factory=list)
    # Pitch over time, ~160 points (None where unvoiced), for the on-screen
    # contour. Display only: dropped from what is saved with a voice.
    contour: List[Optional[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_meta(self) -> dict:
        d = self.to_dict()
        d.pop("contour", None)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


PAUSE_S = 0.30


class TranscriberUnavailable(RuntimeError):
    """faster-whisper missing, or the model isn't downloaded and we're offline."""


# ---------------------------------------------------------------------------
# Model cache. One recogniser at a time, behind a lock: Gradio handles requests
# on worker threads and loading twice would double the RAM.
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_MODEL = {"name": None, "obj": None}


def unload() -> None:
    """Free the recogniser (called before a TTS model loads on a small machine)."""
    with _LOCK:
        if _MODEL["obj"] is not None:
            _MODEL["obj"] = None
            _MODEL["name"] = None
            import gc

            gc.collect()


def is_loaded() -> bool:
    return _MODEL["obj"] is not None


def _get_model(name: str, offline: bool):
    if _MODEL["name"] == name and _MODEL["obj"] is not None:
        return _MODEL["obj"]
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - depends on the install
        raise TranscriberUnavailable(
            "Speech recognition isn't installed. Run: pip install faster-whisper") from e
    _MODEL["obj"] = None
    model = WhisperModel(model_path(name, offline), device="cpu", compute_type="int8")
    _MODEL.update(name=name, obj=model)
    return model


def model_path(name: str, offline: bool = False) -> str:
    """Local folder holding the CTranslate2 weights for `name`, downloading
    them on first use. `local_dir` copies real files instead of the cache's
    symlinks, which Windows refuses without Developer Mode (WinError 1314)."""
    from faster_whisper.utils import _MODELS

    dest = MODEL_DIR / name
    if (dest / "model.bin").exists():
        return str(dest)
    if offline:
        raise TranscriberUnavailable(
            f"The speech-recognition model ({name}) isn't downloaded yet, and the app is in "
            "offline mode. Start it once without --offline to fetch it.")
    from huggingface_hub import snapshot_download

    snapshot_download(_MODELS.get(name, name), local_dir=str(dest),
                      allow_patterns=["config.json", "preprocessor_config.json", "model.bin",
                                      "tokenizer.json", "vocabulary.*"])
    return str(dest)


def lexicon_hotwords(lexicon_path: Optional[Path]) -> str:
    """The left-hand side of each plain (non-regex) lexicon rule: words the user
    cares enough about to have written a pronunciation fix for."""
    if not lexicon_path or not Path(lexicon_path).exists():
        return ""
    out = []
    for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "->" not in line:
            continue
        pat = line.split("->", 1)[0].strip()
        if pat.startswith("/") and pat.endswith("/"):
            continue
        out.append(pat)
    return " ".join(out[:40])


def transcribe(path: str, *, verbatim: bool = True, model_name: str = DEFAULT_MODEL,
               language: Optional[str] = None, extra_hotwords: str = "",
               offline: bool = False, spoken_numbers: bool = True) -> Transcript:
    """Transcribe one clip. `language=None` lets Whisper detect it.
    `spoken_numbers` writes "24" as "twenty-four", matching what was said."""
    hot = " ".join(x for x in (SA_HOTWORDS, extra_hotwords) if x).strip()
    audio = load_16k(path)
    with _LOCK:
        model = _get_model(model_name, offline)
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            initial_prompt=_VERBATIM_PROMPT if verbatim else _CLEAN_PROMPT,
            hotwords=hot or None,
        )
        words: List[Word] = []
        texts: List[str] = []
        for seg in segments:          # a generator: decoding happens here
            texts.append(seg.text.strip())
            for w in seg.words or []:
                words.append(Word(w.word.strip(), float(w.start), float(w.end), float(w.probability)))
    text = _tidy(" ".join(t for t in texts if t))
    if spoken_numbers:
        import textprep

        text = textprep.spell_out_numbers(text)
    return Transcript(text=text, words=words, language=info.language,
                      language_prob=float(info.language_probability),
                      duration=float(info.duration), model=model_name)


def load_16k(path) -> np.ndarray:
    """Mono float32 at 16 kHz, decoded by the same reader the clip checks use.
    Passing samples (not a path) keeps faster-whisper off PyAV, whose newer
    releases changed the open() signature it calls."""
    import refclip

    y, sr, _channels = refclip.load_mono(path)
    if sr != 16000:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(y, dtype=np.float32)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


# ---------------------------------------------------------------------------
# Passage matching. When the user reads the on-screen passage, its exact wording
# is a better transcript than any recogniser's.
# ---------------------------------------------------------------------------
def _norm_words(text: str) -> List[str]:
    """Lower-case words with numbers spelt out, so "24-bit" and "twenty four
    bit" compare equal -- formatting is not a recognition error."""
    import textprep

    text = textprep.spell_out_numbers(text.replace("%", " percent"))
    text = text.lower().replace("-", " ")
    return [w.strip("'") for w in re.sub(r"[^a-z0-9' ]+", " ", text).split() if w.strip("'")]


def match_passage(text: str, passage: str) -> float:
    """Word-level similarity (0..1) between a transcript and a known passage."""
    a, b = _norm_words(text), _norm_words(passage)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER (substitutions + insertions + deletions) / reference words."""
    r, h = _norm_words(reference), _norm_words(hypothesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


# ---------------------------------------------------------------------------
# Delivery profile
# ---------------------------------------------------------------------------
def _pitch_track(y: np.ndarray, sr: int) -> np.ndarray:
    """F0 per 20 ms frame (Hz), NaN where unvoiced. YIN plus an energy and
    periodicity gate: fast enough for a 45 s clip and steady on speech."""
    import librosa

    hop = int(sr * 0.02)
    f0 = librosa.yin(y, fmin=60, fmax=420, sr=sr, frame_length=int(sr * 0.064), hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.064), hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    gate = rms > max(1e-4, np.percentile(rms, 60) * 0.35)
    f0 = np.where(gate, f0, np.nan)
    # YIN reports fmax/fmin on noise; drop the rails and octave jumps.
    f0[(f0 <= 62) | (f0 >= 415)] = np.nan
    med = np.nanmedian(f0) if np.isfinite(f0).any() else np.nan
    if np.isfinite(med):
        f0[(f0 > med * 1.9) | (f0 < med / 1.9)] = np.nan
    return f0


def profile(path: str, t: Optional[Transcript] = None) -> VoiceProfile:
    """How this person speaks: pitch, range, pace, pauses, fillers, clarity."""
    y, sr = load_16k(path), 16000
    p = VoiceProfile()

    f0 = _pitch_track(y, sr)
    if f0.size:
        bins = np.array_split(f0, min(160, f0.size))
        p.contour = [(round(float(np.nanmedian(b)), 1) if np.isfinite(b).any() else None) for b in bins]
    voiced = f0[np.isfinite(f0)]
    if voiced.size > 20:
        p.pitch_hz = float(np.median(voiced))
        p.pitch_lo, p.pitch_hi = (float(x) for x in np.percentile(voiced, [10, 90]))
        st = 12 * np.log2(voiced / p.pitch_hz)
        # A percentile span, not a standard deviation: creaky phrase ends and the odd
        # octave slip make the SD swing wildly, but they barely move the span.
        p.pitch_range_st = float(np.percentile(st, 90) - np.percentile(st, 10))
        p.register = ("low" if p.pitch_hz < 120 else "medium" if p.pitch_hz < 185 else "high")

    if t and t.words:
        words = t.words
        speaking = max(0.5, words[-1].end - words[0].start)
        n_words = len([w for w in words if w.text.strip(" ,.!?").lower() not in FILLERS])
        p.wpm = 60.0 * n_words / speaking
        gaps = [b.start - a.end for a, b in zip(words, words[1:])]
        long_gaps = [g for g in gaps if g >= PAUSE_S]
        p.pauses = len(long_gaps)
        p.longest_pause = float(max(long_gaps, default=0.0))
        if long_gaps:
            p.pause_ms = int(np.median(long_gaps) * 1000)
        # Pauses that follow the end of a sentence are what "pause between
        # sentences" in Speak controls; match them when there are a few.
        sent = [g for a, g in zip(words, gaps) if a.text.rstrip().endswith((".", "!", "?")) and g > 0.08]
        if len(sent) >= 2:
            p.gap_ms = int(min(1200, max(150, round(float(np.median(sent)) * 1000 / 50) * 50)))
        p.fillers = sum(1 for w in words if w.text.strip(" ,.!?").lower() in FILLERS)
        p.clarity = t.confidence

        # Loudness consistency across words.
        db = []
        for w in words:
            seg = y[int(w.start * sr):int(w.end * sr)]
            if seg.size > sr * 0.05:
                db.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
        if len(db) > 4:
            p.loudness_var_db = float(np.std(db))

    if p.wpm is not None:
        p.pace = ("slow and deliberate" if p.wpm < 120 else "relaxed" if p.wpm < 150
                  else "conversational" if p.wpm < 175 else "brisk" if p.wpm < 200 else "fast")
    if p.pitch_range_st is not None:
        # Bands checked against Praat on this repo's clips: read speech spans
        # roughly 8-13 semitones; a flat read less, an energetic ad read more.
        r = p.pitch_range_st
        p.expressiveness = "even and steady" if r < 8 else "natural" if r < 13 else "animated"
        p.delivery = "Steady" if r < 8 else "Natural" if r < 13 else "Expressive"

    if p.fillers >= 3:
        p.notes.append(f"{p.fillers} filler sounds (um, uh). They stay in the transcript because "
                       "they are in the audio; a cleaner take makes a cleaner voice.")
    if p.longest_pause > 1.5:
        p.notes.append(f"One pause runs {p.longest_pause:.1f}s. Long silences can make the "
                       "cloned voice hesitate.")
    if p.clarity is not None and p.clarity < 0.7:
        p.notes.append("Some words were hard to make out. Check the highlighted ones, or record "
                       "closer to the microphone.")
    return p
