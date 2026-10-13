"""Check and tidy a reference clip before it becomes a voice.

Cloning quality is set almost entirely by the reference clip, and the failure
modes are mundane: a muted or privacy-blocked microphone records pure silence,
a hot mic clips, a long clip slows encoding for no gain. None of that surfaced
before -- the clip went straight into the model and a bad voice came out the
other end, minutes later.

`analyze()` reports those problems up front, in words, and `prepare()` writes
the copy the voice is actually built from: mono, leading/trailing silence
trimmed, peak-levelled, stored next to the voice. That copy also replaces the
browser recording's temp path, which Gradio deletes when its cache is cleared.

    python scripts/refclip.py some_clip.wav      # print the report
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

MIN_SECONDS = 3.0           # below this the speaker embedding is unreliable
IDEAL_SECONDS = (5.0, 30.0)
MAX_SECONDS = 60.0          # longer clips only slow encoding; refuse past this
TARGET_PEAK = 0.89          # -1 dBFS
SILENT_RMS_DB = -50.0       # a muted/blocked mic records this, not speech
QUIET_RMS_DB = -36.0
CLIP_LEVEL = 0.999
FRAME_S = 0.025


@dataclass
class ClipReport:
    duration: float
    sample_rate: int
    channels: int
    peak_db: float
    rms_db: float
    clipped_pct: float
    speech_seconds: float
    issues: List[Tuple[str, str]] = field(default_factory=list)   # (level, message)

    @property
    def ok(self) -> bool:
        """False when the clip can't make a usable voice ('err' issues)."""
        return not any(level == "err" for level, _ in self.issues)

    @property
    def grade(self) -> str:
        if not self.ok:
            return "unusable"
        return "good" if not self.issues else "usable"


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-9)))


def load_mono(path) -> Tuple[np.ndarray, int, int]:
    """(float32 mono waveform, sample rate, original channel count)."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        # Compressed formats libsndfile can't read (mp3 on old builds, m4a...).
        import librosa

        wav, sr = librosa.load(str(path), sr=None, mono=False)
        data = np.atleast_2d(wav).T.astype(np.float32)
    channels = data.shape[1]
    return data.mean(axis=1).astype(np.float32), int(sr), int(channels)


def _frame_rms_db(wav: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * FRAME_S))
    usable = len(wav) // n * n
    if usable == 0:
        return np.array([_db(float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0)])
    frames = wav[:usable].reshape(-1, n)
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-9))


def analyze_wave(wav: np.ndarray, sr: int, channels: int = 1) -> ClipReport:
    duration = len(wav) / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    clipped = float(np.mean(np.abs(wav) >= CLIP_LEVEL) * 100) if len(wav) else 0.0

    frames = _frame_rms_db(wav, sr)
    # Speech = frames within 35 dB of the loudest stretch, and not near-silence.
    floor = max(float(np.percentile(frames, 95)) - 35.0, SILENT_RMS_DB)
    speech = float(np.sum(frames > floor) * FRAME_S)

    rep = ClipReport(duration, sr, channels, _db(peak), _db(rms), clipped, speech)
    issues = rep.issues

    if rep.rms_db < SILENT_RMS_DB or peak < 0.01:
        issues.append(("err", "The clip is silent. The microphone may be muted, or Windows may be "
                              "blocking the browser from it (Settings > Privacy > Microphone)."))
    elif rep.rms_db < QUIET_RMS_DB:
        issues.append(("warn", "Very quiet. Move closer to the microphone or raise its input level."))

    if duration < MIN_SECONDS:
        issues.append(("err", f"Too short ({duration:.1f}s). Record at least {IDEAL_SECONDS[0]:.0f} seconds of speech."))
    elif duration < IDEAL_SECONDS[0]:
        issues.append(("warn", f"A little short ({duration:.1f}s). {IDEAL_SECONDS[0]:.0f}-{IDEAL_SECONDS[1]:.0f} seconds clones best."))
    elif duration > MAX_SECONDS:
        issues.append(("err", f"Too long ({duration:.0f}s). Trim it to under {IDEAL_SECONDS[1]:.0f} seconds."))
    elif duration > IDEAL_SECONDS[1]:
        issues.append(("warn", f"Longer than needed ({duration:.0f}s). {IDEAL_SECONDS[1]:.0f} seconds is plenty "
                               "and encodes faster."))

    if clipped > 0.05:
        issues.append(("warn", f"Distorted: {clipped:.1f}% of samples are clipped. Record a little quieter."))

    if rep.ok and duration >= MIN_SECONDS and speech < 0.4 * duration:
        issues.append(("warn", "Mostly silence. Trim the pauses or record again, speaking throughout."))
    return rep


def analyze(path) -> ClipReport:
    wav, sr, channels = load_mono(path)
    return analyze_wave(wav, sr, channels)


def trim_silence(wav: np.ndarray, sr: int, pad_s: float = 0.15) -> np.ndarray:
    """Cut leading/trailing silence, keeping `pad_s` of air either side."""
    frames = _frame_rms_db(wav, sr)
    if len(frames) < 2:
        return wav
    floor = max(float(np.percentile(frames, 95)) - 35.0, SILENT_RMS_DB)
    loud = np.nonzero(frames > floor)[0]
    if not len(loud):
        return wav
    n = int(sr * FRAME_S)
    pad = int(sr * pad_s)
    start = max(0, loud[0] * n - pad)
    end = min(len(wav), (loud[-1] + 1) * n + pad)
    return wav[start:end]


def prepare(src, dest) -> Tuple[Path, ClipReport]:
    """Write the cleaned clip a voice is built from. Returns (dest, report of the result)."""
    wav, sr, _ = load_mono(src)
    wav = trim_silence(wav, sr)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0.01:
        wav = wav * (TARGET_PEAK / peak)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), wav.astype(np.float32), sr, subtype="PCM_16")
    return dest, analyze_wave(wav, sr)


def main(argv=None) -> int:
    for p in (argv if argv is not None else sys.argv[1:]):
        r = analyze(p)
        print(f"{p}: {r.grade}  {r.duration:.1f}s  {r.sample_rate} Hz  {r.channels}ch  "
              f"peak {r.peak_db:.1f} dBFS  rms {r.rms_db:.1f} dBFS  speech {r.speech_seconds:.1f}s")
        for level, msg in r.issues:
            print(f"  [{level}] {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
