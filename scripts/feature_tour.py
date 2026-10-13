"""A guided tour + smoke test of the read-quality features.

Runs quickly and needs no model: it exercises the text preparation, the pronunciation
lexicon, the loudness normalisation, the low-memory dtype policy, the reference-clip
cache and the per-line re-stitch, printing what each one does on a worked example.

    python scripts/feature_tour.py               # the fast tour (no model)
    python scripts/feature_tour.py --generate     # + one real 0.6B generation (slow)
    python scripts/feature_tour.py --generate --online   # allow HF network checks

The `--generate` step shells out to `tts.py preset` with every feature turned on
(low-memory, steady delivery, text tidy-up, -16 LUFS) and reports the resulting wav.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import textprep  # noqa: E402
from common import (  # noqa: E402
    DELIVERY_PRESETS,
    LOW_MEMORY_GB,
    OUTPUT_DIR,
    REPO_ROOT,
    auto_low_memory,
    delivery_sampling,
    loudness_normalize,
    resolve_dtype,
    stitch,
    write_raw_parts,
)

SR = 24000
RULE = "-" * 72


def h(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


SAMPLE_SCRIPT = (
    "# a heading line that will be skipped\n"
    "Dr. Adams joined the team on 3rd March 1985, and by 2007 he was running it.\n"
    "The device costs $1,250.50, which is 25% less than last year, and ships at 10:30 tomorrow.\n"
    "\n"
    "Call 5551234567 if the SQL query fails; otherwise everything is fine and there is "
    "nothing more to do here, so you can relax and enjoy the rest of your afternoon in peace.\n"
)


def tour_textprep() -> None:
    h("1. Text preparation  (numbers, dates, abbreviations, markdown, segmentation)")
    lines, gaps = textprep.prepare_script(SAMPLE_SCRIPT, max_words=24)
    print(f"{len(lines)} lines out of the script (blank line = paragraph break):\n")
    for i, (ln, g) in enumerate(zip(lines, gaps), 1):
        pause = "" if i == 1 else f"   [+{g} ms pause before]"
        print(f"  {i}. {ln}{pause}")
    print("\nThis exact list is what the model is asked to say -- the GUI's "
          "'Preview text' button shows the same thing.")


def tour_lexicon() -> None:
    h("2. Pronunciation lexicon  (lexicon.txt at the repo root)")
    path = textprep.LEXICON_PATH
    live = textprep.load_lexicon()
    print(f"file: {path}")
    print(f"active rules right now: {len(live)}"
          + ("" if live else "  (all examples are commented out -- add your own)"))

    demo = REPO_ROOT / ".cache" / "feature_tour_lexicon.txt"
    demo.parent.mkdir(parents=True, exist_ok=True)
    demo.write_text('SQL -> sequel\nQwen -> "chwen"\n/\\bkm\\b/ -> kilometres\n', encoding="utf-8")
    rules = textprep.load_lexicon(demo)
    before = "The SQL job on the Qwen box ran for 5 km."
    print(f"\nwith an example lexicon ({len(rules)} rules):")
    print(f"  before: {before}")
    print(f"  after : {textprep.apply_lexicon(before, rules)}")
    demo.unlink()
    print("\nLexicon runs before numbers/abbreviations, and even when 'tidy' is off.")


def tour_delivery() -> None:
    h("3. Delivery presets  (common.DELIVERY_PRESETS)")
    for name in ("steady", "natural", "expressive"):
        kw = delivery_sampling(name)
        print(f"  {name:11} -> {kw or '(model defaults)'}")
    print("\nExplicit --temperature / --top-k etc. still override the preset.")
    assert delivery_sampling("steady") == DELIVERY_PRESETS["steady"]


def tour_low_memory() -> None:
    h("4. Low-memory mode  (0.6B as bfloat16)")
    import psutil

    free = psutil.virtual_memory().available / 1e9
    auto = auto_low_memory()
    print(f"free RAM: {free:.1f} GB   threshold: {LOW_MEMORY_GB:.0f} GB   "
          f"auto-low-memory: {'ON' if auto else 'off'}")
    print("\ndtype that would be loaded:")
    for size in ("0.6B", "1.7B"):
        normal = resolve_dtype("cpu", size, "auto", low_memory=False)
        low = resolve_dtype("cpu", size, "auto", low_memory=True)
        print(f"  {size}: normal={str(normal).split('.')[-1]:9}  low-memory={str(low).split('.')[-1]}")
    print("\nPass --low-memory (CLI or app.py) to force it; otherwise it turns on by itself "
          "when RAM is tight.")


def tour_loudness() -> None:
    h("5. Loudness normalisation  (-16 LUFS target, one gain across the take)")
    try:
        import pyloudnorm as pyln
    except ImportError:
        print("pyloudnorm not installed -- feature falls back to peak limiting. "
              "`pip install pyloudnorm` to enable.")
        return

    meter = pyln.Meter(SR)
    quiet = [tone(2.0, amp=0.03)]
    loud = [tone(2.0, amp=0.9)]
    for label, wav in (("quiet input", quiet), ("hot input", loud)):
        before = meter.integrated_loudness(wav[0])
        after_arr = loudness_normalize(wav, SR, target_lufs=-16.0, peak=0.97)
        after = meter.integrated_loudness(after_arr[0])
        peak = float(np.max(np.abs(after_arr[0])))
        print(f"  {label:11}: {before:7.1f} LUFS  ->  {after:7.1f} LUFS   (peak {peak:.2f})")
    off = loudness_normalize(loud, SR, target_lufs=0, peak=0.97)
    print(f"  --lufs 0     : disabled, peak-limit only -> peak {float(np.max(np.abs(off[0]))):.2f}")


def tour_prompt_cache() -> None:
    h("6. Reference-clip cache  (audio_prompt_cache: ad-hoc `clone --ref-audio`)")
    import audio_prompt_cache as apc

    tmp = REPO_ROOT / ".cache" / "feature_tour"
    tmp.mkdir(parents=True, exist_ok=True)
    clip = tmp / "clip.wav"
    import soundfile as sf

    sf.write(clip, tone(1.0), SR)

    k1 = apc.cache_key(str(clip), "the transcript", False, "0.6B")
    k2 = apc.cache_key(str(clip), "different transcript", False, "0.6B")
    print(f"key(clip, textA) = {k1}")
    print(f"key(clip, textB) = {k2}   <- different text, different key")

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            self.calls += 1
            time.sleep(0.2)  # stand-in for the real tens-of-seconds encode
            return [{"ref_text": ref_text, "x_vector_only_mode": x_vector_only_mode}]

    # trivial (de)serialise so we don't need a real dataclass here
    orig_p, orig_r = apc.payload_from_items, apc.rebuild_items
    apc.payload_from_items = lambda items: {"items": items}
    apc.rebuild_items = lambda payload: payload["items"]
    try:
        m = FakeModel()
        cdir = tmp / "prompts"
        t0 = time.time()
        _, hit1 = apc.encode_prompt(m, str(clip), "the transcript", False, "0.6B",
                                    cache_dir=cdir, verbose=False)
        t1 = time.time()
        _, hit2 = apc.encode_prompt(m, str(clip), "the transcript", False, "0.6B",
                                    cache_dir=cdir, verbose=False)
        t2 = time.time()
        print(f"\n  1st call: from_cache={hit1}  ({(t1 - t0) * 1000:.0f} ms, encoder ran)")
        print(f"  2nd call: from_cache={hit2}  ({(t2 - t1) * 1000:.0f} ms, encoder skipped)")
        print(f"  encoder invocations: {m.calls}  (the named voice library already did this "
              "for saved voices; this covers one-off clips)")
    finally:
        apc.payload_from_items, apc.rebuild_items = orig_p, orig_r


def tour_per_line() -> None:
    h("7. Per-line re-stitch  (redo one sentence, keep the rest)")
    tmp = REPO_ROOT / ".cache" / "feature_tour"
    tmp.mkdir(parents=True, exist_ok=True)

    take = [tone(0.6, freq=200), tone(0.6, freq=300), tone(0.6, freq=400)]
    parts = write_raw_parts(take, SR, str(tmp / "parts"), "demo")
    gaps = [0, 300, 650]
    joined = stitch([str(p) for p in parts], str(tmp / "joined_v1.wav"), gaps=gaps, lufs=0)
    import soundfile as sf

    d1, _ = sf.read(joined)
    print(f"  3 raw line-parts written: {[p.name for p in parts]}")
    print(f"  stitched v1: {len(d1) / SR:.2f} s  (gaps {gaps} ms)")

    # "regenerate line 2" -> swap just that part, re-stitch
    new_line2 = tone(1.1, freq=330)   # a longer, different take
    sf.write(parts[1], new_line2, SR)
    joined2 = stitch([str(p) for p in parts], str(tmp / "joined_v2.wav"), gaps=gaps, lufs=0)
    d2, _ = sf.read(joined2)
    print(f"  after redoing line 2 only, stitched v2: {len(d2) / SR:.2f} s "
          f"(lines 1 and 3 untouched)")
    print("\nThe GUI's 'Fix one line' panel does exactly this; the CLI keeps numbered "
          "per-line wavs when you drop --join.")


def tour_chunking() -> None:
    h("8. Per-line generation  (common.synth / --chunk-size: the RAM fix)")
    from common import synth

    class FakeModel:
        def __init__(self):
            self.batch_sizes = []

        def generate_custom_voice(self, **kw):
            n = len(kw["text"])
            self.batch_sizes.append(n)
            return [tone(0.1) for _ in range(n)], SR

    lines = [f"sentence {i}" for i in range(5)]
    for cs in (1, 2, 0):
        m = FakeModel()
        wavs, _ = synth("preset", m, lines, language="Auto", sampling={},
                        speaker="ryan", chunk_size=cs)
        label = "one line at a time" if cs == 1 else ("one batch" if cs == 0 else f"{cs} per call")
        print(f"  --chunk-size {cs}: model called with batches {m.batch_sizes}  ({label}); "
              f"{len(wavs)} wavs out")
    print("\nqwen_tts pads every line into one tensor per call, so chunk-size 1 keeps the peak "
          "at a single line -- what makes a long script finish on the 8GB box. On a big machine "
          "use --chunk-size 0 for the faster single batch.")


def tour_tests() -> None:
    h("9. Test suite  (pytest)")
    print("running: python -m pytest -q\n")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(REPO_ROOT))
    print(f"\npytest exit code: {r.returncode}")


def do_generate(online: bool) -> None:
    h("10. One real end-to-end generation  (0.6B, every feature on -- slow)")
    script = (
        "Dr. Adams paid $1,250.50 on the 3rd of March 1985.\n"
        "That was about 25% less than the list price at the time.\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    out = OUTPUT_DIR / "feature_tour.wav"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "tts.py"), "preset",
        "--text-file", script_path, "--speaker", "Ryan", "--join",
        "--low-memory", "--delivery", "steady", "--lufs", "-16",
        "--out", str(out),
    ]
    if not online:
        cmd.append("--offline")
    print("running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("(first run loads the 0.6B checkpoint -- minutes on CPU)\n")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    dt = time.time() - t0
    if r.returncode != 0 or not out.exists():
        print(f"\ngeneration failed (exit {r.returncode}) after {dt:.0f}s -- "
              "see the output above.")
        return

    import soundfile as sf

    data, sr = sf.read(out)
    peak = float(np.max(np.abs(data)))
    line = f"\nwrote {out}\n  {len(data) / sr:.2f} s at {sr} Hz, peak {peak:.2f}, took {dt:.0f}s"
    try:
        import pyloudnorm as pyln

        line += f", {pyln.Meter(sr).integrated_loudness(data):.1f} LUFS (target -16)"
    except ImportError:
        pass
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true",
                    help="also run one real 0.6B generation through tts.py (slow)")
    ap.add_argument("--online", action="store_true",
                    help="with --generate, allow Hugging Face network checks (default: --offline)")
    ap.add_argument("--skip-tests", action="store_true", help="don't run pytest")
    args = ap.parse_args()

    print("Qwen3-TTS feature tour")
    tour_textprep()
    tour_lexicon()
    tour_delivery()
    tour_low_memory()
    tour_loudness()
    tour_prompt_cache()
    tour_per_line()
    tour_chunking()
    if not args.skip_tests:
        tour_tests()
    if args.generate:
        do_generate(args.online)

    h("done")
    print("Fast checks above need no model. Re-run with --generate for a real synthesis,")
    print("or start the app:  python scripts/app.py --offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
