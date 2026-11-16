"""Local reference-clip transcription, passage matching and delivery profiles.

Whisper loads lazily and runs under a lock. Prompts preserve disfluencies;
South African vocabulary and lexicon hotwords guide spelling. Uncertain words
are exposed for review. language=SPLIT transcribes English and Afrikaans parts
of a clip separately, each in its own language. unload() frees the recogniser before TTS model loads.
"""
from __future__ import annotations

import difflib
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# large-v3-turbo: near large-v3 accuracy, much cheaper; int8 fits in 8 GB. See README.
DEFAULT_MODEL = "large-v3-turbo"
FAST_MODEL = "small"

# Words a generic recogniser mangles in a Cape Town studio. Spelt as users write them.
SA_HOTWORDS = (
    "lekker howzit koeksister Bo-Kaap bru boet ouma oupa braai bakkie robot eish ja "
    "Constantia Cape Flats Simon's Town Gqeberha Stellenbosch Khayelitsha Muizenberg "
    "Woodstock rand Mzansi shame sharp now-now just-now"
)

# Whisper copies the prompt's style, so disfluencies here keep them in the transcript.
_VERBATIM_PROMPT = "Umm, so, uh, I was like, hmm, you know... Okay. Right, ja."
_CLEAN_PROMPT = "Hello, welcome. It's a lekker day in Cape Town."
_PROMPTS = {"af": ("Uhm, so, ek was, hmm, jy weet... Okay. Reg, ja.",
                   "Hallo, welkom. Dis 'n lekker dag in Kaapstad.")}

# Split mode: the clip is written out once as English and once as Afrikaans,
# and each sentence keeps the version in its own language.
SPLIT = "split"
SPLIT_LANGUAGES = ("en", "af")
# Below this whole-clip Afrikaans score a short clip is treated as English only
# and the Afrikaans pass is skipped. A real clip that was half Afrikaans scored
# 0.44, an all-English one under 0.005.
AF_POSSIBLE = 0.02
# The whole-clip score only covers the first 30 seconds of speech.
LANGUAGE_WINDOW_S = 28.0
# With no telling words in a sentence, Afrikaans has to be this much more
# confident than English to win.
AF_MARGIN = 0.10

# Small words that only one of the two languages uses. Shared spellings
# (is, was, in, of, as, by, my, so, want, word, hand, water) are left out.
_AF_WORDS = frozenset("""
'n die nie en het van om te wat vir ek jy jou ons dit met op aan sal kan moet ook maar
hulle baie nou hier daar sy hy dan toe nog uit oor na tot wees gaan kom weet dink maak goed
mense jaar dag tyd elke alles niks iets julle hierdie daardie waar hoe wie hoekom omdat
terwyl tussen onder agter voor sonder deur saam ja nee dis net soos geen meer jul haar hom
""".split())
_EN_WORDS = frozenset("""
the and to a that it you for with this have are not but they from your we there their what
will would about which when can more an been has had were who do does did i i'm it's don't
just know get like how our out up them than then into only every because behind there's
make makes she he his her its these those where why while between under after before
without through together yes no isn't doesn't can't that's here all very now people good
""".split())

FILLERS = {"um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm"}

# A word below this confidence is highlighted for the user to check.
UNSURE_BELOW = 0.6

# Plain folders outside the repo (the HF cache's symlinks fail on Windows, WinError 1314).
MODEL_DIR = Path.home() / ".cache" / "skinnerbek" / "whisper"


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float


@dataclass
class Part:
    """One stretch of a clip in one language."""
    language: str
    start: float
    end: float
    text: str
    words: List["Word"] = field(default_factory=list)


@dataclass
class Transcript:
    text: str
    words: List[Word]
    language: str
    language_prob: float
    duration: float
    model: str
    parts: List[Part] = field(default_factory=list)

    @property
    def languages(self) -> List[str]:
        """Languages heard, in the order they first come up."""
        return list(dict.fromkeys(p.language for p in self.parts)) or [self.language]

    def text_for(self, language: str) -> str:
        return " ".join(p.text for p in self.parts if p.language == language and p.text)

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
    # ~160 pitch points (None where unvoiced), for display only; not saved.
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


# --- Model cache: one recogniser, behind a lock (Gradio uses worker threads) ---
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
    _remember_last_encode(model)
    _MODEL.update(name=name, obj=model)
    return model


def _remember_last_encode(model) -> None:
    """Keep recent encoder passes so the same audio isn't encoded twice.

    The encoder is most of the cost on a CPU. A language check, an English
    pass and an Afrikaans pass all encode the same 30-second windows, so the
    repeats return the first one's result. Three windows cover a 60 s clip."""
    import hashlib

    real = model.encode
    kept: dict = {}

    def encode(features):
        key = (features.shape, hashlib.sha1(np.ascontiguousarray(features).tobytes()).digest())
        if key not in kept:
            while len(kept) >= 3:
                kept.pop(next(iter(kept)))
            kept[key] = real(features)
        return kept[key]

    encode.forget = kept.clear
    model.encode = encode


def model_path(name: str, offline: bool = False) -> str:
    """Folder with the weights for `name`, downloaded on first use."""
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


def _plain(word: str) -> str:
    """A word as the language lists spell it: lower case, bare, 'n kept."""
    w = word.strip().lower().replace("\u2019", "'").replace("\u02bc", "'").replace("\u0149", "'n")
    w = w.strip(".,!?;:\"()[]\u2026\u201c\u201d-")
    return w


def text_language(text: str) -> Optional[str]:
    """"af", "en", or None when the small words don't settle it."""
    af = en = 0
    for raw in text.split():
        w = _plain(raw)
        af += w in _AF_WORDS
        en += w in _EN_WORDS
    if af == en:
        return None
    return "af" if af > en else "en"


def sentences(words: List[Word], max_gap: float = 1.0) -> List[List[Word]]:
    """Words grouped into sentences: a full stop, question or exclamation mark
    ends one, and so does a pause longer than `max_gap` seconds."""
    out: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        if cur and w.start - cur[-1].end > max_gap:
            out.append(cur)
            cur = []
        cur.append(w)
        if w.text.rstrip("\"')\u201d").endswith((".", "!", "?", "\u2026")):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _between(words: List[Word], start: float, end: float) -> List[Word]:
    """The words whose middle falls in [start, end)."""
    return [w for w in words if start <= (w.start + w.end) / 2 < end]


def _sure(words: List[Word]) -> float:
    return float(np.mean([w.prob for w in words])) if words else 0.0


def sentence_language(af_sentence: List[Word], en_words: List[Word]) -> Optional[str]:
    """Which language one sentence is in, from the two write-ups of it.

    `af_sentence` is the sentence as the Afrikaans pass wrote it; `en_words`
    is what the English pass wrote for the same seconds. The Afrikaans pass
    writes English speech as English, so its small words usually settle it.
    None means undecided (the caller uses the neighbouring sentence)."""
    said_af = " ".join(w.text for w in af_sentence)
    said_en = " ".join(w.text for w in en_words)
    lang = text_language(said_af)
    if lang == "en":
        return "en"
    sure_af, sure_en = _sure(af_sentence), _sure(en_words)
    if lang == "af":
        # Guard: Afrikaans words for English speech (a translation) while the
        # English pass heard plain English just as confidently.
        if text_language(said_en) == "en" and sure_en >= sure_af + AF_MARGIN:
            return "en"
        return "af"
    if sure_af >= sure_en + AF_MARGIN:
        return "af"
    if sure_en >= sure_af + AF_MARGIN:
        return "en"
    return None


def split_by_sentence(af_words: List[Word], en_words: List[Word]) -> List[Part]:
    """Parts in spoken order, each one language, from the two passes' words.

    Afrikaans parts take the Afrikaans pass's words, English parts the English
    pass's words for the same seconds."""
    sents = sentences(af_words)
    if not sents:
        return []
    langs: List[Optional[str]] = []
    for i, sent in enumerate(sents):
        start = sent[0].start if i else 0.0
        end = sents[i + 1][0].start if i + 1 < len(sents) else float("inf")
        langs.append(sentence_language(sent, _between(en_words, start, end)))
    known = [lg for lg in langs if lg]
    fill = known[0] if known else "en"
    for i, lg in enumerate(langs):
        if lg is None:
            langs[i] = langs[i - 1] if i else fill

    # Runs of one language, as (language, first sentence, last sentence).
    runs: List[list] = []
    for i, lg in enumerate(langs):
        if runs and runs[-1][0] == lg:
            runs[-1][2] = i
        else:
            runs.append([lg, i, i])
    parts: List[Part] = []
    for n, (lg, first, last) in enumerate(runs):
        own = [w for sent in sents[first:last + 1] for w in sent]
        start = own[0].start if n else 0.0
        end = sents[last + 1][0].start if last + 1 < len(sents) else float("inf")
        chosen = own if lg == "af" else (_between(en_words, start, end) or own)
        said = _tidy(" ".join(w.text for w in chosen))
        if lg == "af":
            # Whisper writes the article 'n as "A"; Afrikaans has no word "a".
            said = re.sub(r"(?<![\w'])[Aa](?=\s+\w)", "'n", said)
        parts.append(Part(lg, chosen[0].start, chosen[-1].end, said, chosen))
    return parts


def _decode(model, audio, language, verbatim, hot, offset=0.0):
    prompts = _PROMPTS.get(language, (_VERBATIM_PROMPT, _CLEAN_PROMPT))
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(_VAD),
        condition_on_previous_text=False,
        initial_prompt=prompts[0] if verbatim else prompts[1],
        hotwords=hot or None,
    )
    words: List[Word] = []
    texts: List[str] = []
    for seg in segments:          # a generator: decoding happens here
        texts.append(seg.text.strip())
        for w in seg.words or []:
            words.append(Word(w.word.strip(), float(w.start) + offset, float(w.end) + offset,
                              float(w.probability)))
    return _tidy(" ".join(t for t in texts if t)), words, info


_VAD = {"min_silence_duration_ms": 500}


def _clip_language(model, audio) -> dict:
    """Language probabilities for the clip's first 30 seconds of speech.

    Builds the window exactly as model.transcribe() will (same VAD, same
    trimming), so the transcription that follows reuses this encoder pass
    (see _remember_last_encode) instead of paying for a second one."""
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    chunks = get_speech_timestamps(audio, VadOptions(**_VAD))
    if chunks:
        pieces, _meta = collect_chunks(audio, chunks)
        audio = np.concatenate(pieces, axis=0)
    features = model.feature_extractor(audio)
    window = pad_or_trim(features[:, :min(model.feature_extractor.nb_max_frames,
                                          features.shape[-1] - 1)])
    results = model.model.detect_language(model.encode(window))[0]
    return {token[2:-2]: float(prob) for token, prob in results}


def transcribe(path: str, *, verbatim: bool = True, model_name: str = DEFAULT_MODEL,
               language: Optional[str] = None, extra_hotwords: str = "",
               offline: bool = False, spoken_numbers: bool = True) -> Transcript:
    """Transcribe one clip. `language=None` lets Whisper detect it; SPLIT
    writes each sentence in English or Afrikaans, whichever it is spoken in,
    and fills `parts`. `spoken_numbers` writes "24" as "twenty-four" (English only)."""
    hot = " ".join(x for x in (SA_HOTWORDS, extra_hotwords) if x).strip()
    audio = load_16k(path)
    sr = 16000
    with _LOCK:
        model = _get_model(model_name, offline)
        detected = None
        both = False
        try:
            if language == SPLIT:
                probs = _clip_language(model, audio)
                both = probs.get("af", 0.0) >= AF_POSSIBLE or len(audio) / sr > LANGUAGE_WINDOW_S
                language = "en"
            elif language is None:
                probs = _clip_language(model, audio)
                language = max(probs, key=probs.get)
                detected = probs[language]
        except Exception as e:  # noqa: BLE001 -- a failed check must not lose the transcript
            print(f"language check: {type(e).__name__}: {e}", file=sys.stderr)
            language = "en" if language == SPLIT else language
        text, words, info = _decode(model, audio, language, verbatim, hot)
        duration, lang = float(info.duration), info.language
        prob = float(info.language_probability) if detected is None else detected
        parts = [Part(lang, 0.0, duration, text, words)]
        if both:
            # Same audio again as Afrikaans (no English hotwords): the encoder
            # pass is shared, so this costs one more decode.
            _af_text, af_words, _info = _decode(model, audio, "af", verbatim, extra_hotwords)
            split = split_by_sentence(af_words, words)
            if len({p.language for p in split}) > 1 or (split and split[0].language == "af"):
                parts = split
                words = [w for p in parts for w in p.words]
                spoken: dict = {}
                for p in parts:
                    spoken[p.language] = spoken.get(p.language, 0.0) + p.end - p.start
                lang, prob = max(spoken, key=spoken.get), 1.0
        model.encode.forget()
    if spoken_numbers:
        import textprep

        for p in parts:
            if p.language == "en":
                p.text = textprep.spell_out_numbers(p.text)
    text = " ".join(p.text for p in parts if p.text)
    return Transcript(text=text, words=words, language=lang, language_prob=prob,
                      duration=duration, model=model_name, parts=parts)


def load_16k(path) -> np.ndarray:
    """Mono float32 at 16 kHz. Samples, not a path: newer PyAV breaks faster-whisper's open()."""
    import refclip

    y, sr, _channels = refclip.load_mono(path)
    if sr != 16000:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(y, dtype=np.float32)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


# --- Passage matching: a read of the on-screen passage uses its exact text ----
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


# --- Delivery profile ------------------------------------------------------------
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
        # Percentile span, not SD: robust to creak and octave slips.
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
        # Pauses after sentence ends set Speak's "pause between sentences".
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
        # Checked against Praat: read speech spans about 8-13 semitones.
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
